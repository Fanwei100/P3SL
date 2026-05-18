"""Main training loop (data-scientist side).

Implements the round-by-round federated split-learning procedure invoked by
:mod:`training_coordinator`. The public entry point is :func:`TrainingMain`,
which dispatches into :func:`SplitTraining` and either the sequential
(:func:`fit_split_model_Sequentialy`) or parallel
(:func:`ParralelTraining.fit_split_model_Parallelly`) per-step routine, plus
the aggregated evaluation, periodic model aggregation, and power-log
download helpers.
"""

import os,logging,datetime,torch,pandas as pd,time

from wakeonlan import send_magic_packet
from tqdm import tqdm
import numpy as np
from DataSet import getDataSet
from websocketClient import WebsocketClientWorkerCustom
from Util import dataModelSetup,modelFolder,logger,resultFolder,SplitProfilingfFolder,addCsvLog,suspendAll,evaluate_split_model
from Util import loss_fn,downloadModelParametrs,UploadModelParametrs,IP2HOST,HostPower,MakeTempDataFrame
from ParralelTraining import fit_split_model_Parallelly
from SplitPointProfiling import calculatePowerConsumptions,SplitPointProfileingLocal
from P3SLPrivacyOptimization import (build_noise_assignment_table, load_privacy_table, privacy_table_to_payload,
                                     update_noise_assignment_table)


def wakeUpAll(macaddress,logger=logger):
    logger.info("Waking Up ALL.. ")
    for maddress in macaddress:
        send_magic_packet(maddress)
    time.sleep(5)

def _optional_list(values, n, cast=None, name="value"):
    if values is None:
        return [None for _ in range(n)]
    if len(values) == 1 and n > 1:
        values = [values[0] for _ in range(n)]
    if len(values) != n:
        raise ValueError(f"Expected {n} {name} entries, got {len(values)}")
    if cast is None:
        return list(values)
    return [None if v is None else cast(v) for v in values]


def configure_p3sl_from_clients(args, clients, logger):
    """Run Alg. 1/2 initial PL/T_sigma broadcast and client split selection."""
    if not getattr(args, "p3sl_auto_config", False):
        return
    if args.mode != "P3SL":
        logger.warning("--p3sl_auto_config is intended for --mode P3SL; continuing with mode %s", args.mode)

    privacy_table = load_privacy_table(args.p3sl_privacy_table)
    privacy_payload = privacy_table_to_payload(privacy_table)
    noise_assignment = build_noise_assignment_table(
        privacy_table,
        fsim_threshold=args.p3sl_fsim_threshold,
        smax=args.p3sl_smax,
        normalise_before_threshold=args.p3sl_normalize_threshold,
    )
    args._p3sl_privacy_payload = privacy_payload
    args._p3sl_noise_assignment = noise_assignment

    if args.p3sl_min_accuracy is not None:
        args._p3sl_min_accuracy = args.p3sl_min_accuracy
    elif args.p3sl_reference_accuracy is not None:
        args._p3sl_min_accuracy = (1.0 - args.p3sl_accuracy_discount) * args.p3sl_reference_accuracy
    else:
        args._p3sl_min_accuracy = None

    n = len(clients)
    alpha_overrides = _optional_list(args.p3sl_privacy_alpha, n, float, "privacy-alpha")
    max_powers = _optional_list(args.p3sl_pmax, n, float, "Pmax")
    energy_paths = _optional_list(args.p3sl_energy_profile, n, str, "energy-profile")

    selected_splits, selected_noises = [], []
    for idx, (worker, alpha, pmax, energy_path) in enumerate(zip(clients, alpha_overrides, max_powers, energy_paths)):
        selection = worker.selectP3SLSplit(
            noise_assignment=noise_assignment,
            privacy_leakage_table=privacy_payload,
            smax=args.p3sl_smax,
            alpha=alpha,
            max_power=pmax,
            energy_profile_path=energy_path,
        )
        split, noise = int(selection["split"]), float(selection["noise"])
        selected_splits.append(split)
        selected_noises.append(noise)
        worker.setDPInfo(dpinfo={"usedp": noise > 0, "dpmethod": "Laplace", "dploc": 0.0, "dpscale": noise})
        logger.info("P3SL client %s selected split=%s sigma=%s", worker.name, split, noise)

    args.sl = selected_splits
    args.dpNoise = selected_noises
    logger.info("P3SL auto configuration selected splits=%s noises=%s", selected_splits, selected_noises)


def reassign_p3sl_noise_if_needed(args, workers, accuracy, logger, curr_epoch=None):
    """Apply Eq. (5) and re-run client-side split selection when A_t < A_min."""
    if not getattr(args, "p3sl_auto_config", False):
        return
    amin = getattr(args, "_p3sl_min_accuracy", None)
    if accuracy is None or amin is None or accuracy >= amin:
        return

    args._p3sl_noise_assignment = update_noise_assignment_table(args._p3sl_noise_assignment, amin, accuracy)
    logger.warning(
        "P3SL held-out accuracy %.4f is below A_min %.4f; updated noise assignment to %s",
        accuracy, amin, args._p3sl_noise_assignment,
    )

    n = len(workers)
    alpha_overrides = _optional_list(args.p3sl_privacy_alpha, n, float, "privacy-alpha")
    max_powers = _optional_list(args.p3sl_pmax, n, float, "Pmax")
    energy_paths = _optional_list(args.p3sl_energy_profile, n, str, "energy-profile")

    selected_splits, selected_noises = [], []
    for worker, alpha, pmax, energy_path in zip(workers, alpha_overrides, max_powers, energy_paths):
        if curr_epoch is not None and hasattr(worker, "skipEpoches") and curr_epoch in worker.skipEpoches:
            continue
        selection = worker.selectP3SLSplit(
            noise_assignment=args._p3sl_noise_assignment,
            privacy_leakage_table=args._p3sl_privacy_payload,
            smax=args.p3sl_smax,
            alpha=alpha,
            max_power=pmax,
            energy_profile_path=energy_path,
        )
        split, noise = int(selection["split"]), float(selection["noise"])
        selected_splits.append(split)
        selected_noises.append(noise)
        worker.setModelEndLayer(split)
        worker.setDPInfo(dpinfo={"usedp": noise > 0, "dpmethod": "Laplace", "dploc": 0.0, "dpscale": noise})
    args.sl = selected_splits
    args.dpNoise = selected_noises
    logger.info("P3SL reassignment selected splits=%s noises=%s", selected_splits, selected_noises)


def modelsAggregation(model, parmetererlist, device,aggmethod="AggEqual", logger=logger, expariment="Testing", sl_list=None, smax=None):
    """Aggregate client-side layers according to P3SL Eq. (1).

    For P3SL, each client uploads only W_1:s_i.  Missing layers from
    s_i+1:s_max are filled with the current server-side global weights, and
    the average is taken over exactly N clients.  The aggregated W_1:s_max is
    kept on the server and is not pushed back to clients in P3SL mode.
    """
    assert aggmethod in ["AggEqual","AggClientOnly"], str(aggmethod)+" is not allowed aggregation method"
    logger.info("Aggregation Model with %s",aggmethod)
    addCsvLog("Model aggrigation Start",expariment)

    if not parmetererlist:
        logger.info("No client parameters available for aggregation; keeping server model unchanged")
        addCsvLog("Model aggrigation Finish",expariment)
        return model

    server_state = {key: parm.detach().to(device) for key, parm in model.state_dict().items()}

    # If sl_list is provided, aggregate only W_1:smax.  This is the P3SL path.
    # Otherwise fall back to the legacy behaviour, but still do not count the
    # server weights as an extra client.
    aggregate_keys = set(server_state.keys())
    if sl_list is not None:
        if smax is None:
            smax = max(sl_list)
        prev_start, prev_end = getattr(model, "start", 0), getattr(model, "end", -1)
        model.setStartEnd(0, int(smax))
        aggregate_keys = set(model.getTrainedParameter().keys())
        try:
            model.setStartEnd(prev_start, prev_end)
        except Exception:
            model.setStartEnd(0)

    state_dict={key: value.clone() for key, value in server_state.items()}
    for key, server_value in server_state.items():
        if key not in aggregate_keys:
            continue
        client_terms=[]
        for clparam in parmetererlist:
            if key in clparam:
                client_terms.append(clparam[key].to(device))
            elif aggmethod=="AggEqual":
                # Fill W_{s_i+1:smax} with current server-trained weights, as
                # described below Eq. (1) in the manuscript.
                client_terms.append(server_value)
        if client_terms:
            state_dict[key]=torch.mean(torch.stack(client_terms).double(), dim=0).to(server_value.dtype)

    model.load_state_dict(state_dict)
    addCsvLog("Model aggrigation Finish",expariment)
    return model


def evaluate_Aggregated_model(workers, doeval, douploadaggrigate,saveModel,device,IID,dpinfo,trainingmode, dataset, testdatacount,datasize, powerSaver,savePowerLogs, batch_size, lr, modelName, model,optimizer,scheduler, curr_epoch, resultFile, sl_list,aggregationmethod,logger, expariment,experimentinfo):
    # Create and send train config
    loss_list,pred_list,target_list=[],[],[]
    aggregation_accuracy = None
    logger.info("Model Aggregation and Evaluation.....")
    if savePowerLogs: addCsvLog("Downloading Parameters Start",expariment=expariment)

    if saveModel:
        logger.info("Saving Model for %s Epoch", curr_epoch)
        torch.save(model.state_dict(), f"{modelFolder}{dataset}_{modelName}_Server_{expariment}.pt")
        torch.save({"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict() if scheduler is not None else {}},f"{modelFolder}{dataset}_{modelName}_Server_{expariment}_opt.pt")

    if trainingmode=="P3SL":
        if douploadaggrigate:
            logger.warning("P3SL keeps aggregated W_1:smax on the server and does not redistribute it to clients; ignoring upload-aggrigate for P3SL.")
            douploadaggrigate = False
        parameter_list = []
        logger.info("Downloading model parameters")
        for ci, worker in enumerate(workers):
            if curr_epoch in worker.skipEpoches: continue
            worker.connect()
            logger.info("Downloading model parameters from %s",worker.name)
            parameter=worker.getModelPatameter()
            parameter_list.append(parameter)
            if saveModel:
                torch.save(parameter,f"{modelFolder}{dataset}_{modelName}_Client_{worker.name}_{ci + 1}_{expariment}.pt")
                torch.save(worker.getOptimizerPatameter(), f"{modelFolder}{dataset}_{modelName}_Client_{worker.name}_{ci + 1}_{expariment}_opt.pt")
        if savePowerLogs: addCsvLog("Downloading Parameters Finish",expariment=expariment)
        if douploadaggrigate or doeval:
            model=modelsAggregation(model, parameter_list,device, aggmethod=aggregationmethod,logger=logger, expariment=expariment, sl_list=sl_list, smax=max(sl_list)).to(device)
        if douploadaggrigate:
            for worker,sl in zip(workers,sl_list):
                if curr_epoch in worker.skipEpoches: continue
                model.setStartEnd(0,sl)
                # model.setEnd(sl)
                if savePowerLogs: addCsvLog("Sending Parameter start", expariment=expariment)
                logger.info("Sending parameter to %s",worker.name)
                worker.uploadModelParameter({k: v.cpu() for k, v in model.getTrainedParameter().items()})
    if powerSaver: suspendAll(workers,logger)
    if doeval:
        print("Test Parameeters",datasize,testdatacount,batch_size)
        dataloader=torch.utils.data.DataLoader(getDataSet(dataset,datasize=(datasize,datasize),testsize=testdatacount,nclasses=None,IID=IID,clientName="Client1")[1],batch_size=batch_size)
        st=time.time()
        if savePowerLogs: addCsvLog("Server Model Evaluation Start",expariment=expariment)
        model.setStartEnd(start=0)
        for data,target in tqdm(dataloader):
            pred=model(data.to(device))
            pred_list.append(torch.argmax(pred,dim=1).detach().cpu().numpy())
            target_list.append(target.numpy())
            loss_list.append(loss_fn(pred,target.to(device)).detach().cpu().numpy())
        pred,target=np.concatenate(pred_list),np.concatenate(target_list)
        accuracy = np.sum(pred == target) / pred.shape[0]
        timetaken=time.time()-st
        if savePowerLogs: addCsvLog("Model Evaluation Finish",expariment=expariment)
        with open(resultFile, "a") as f:
            f.write(f"{experimentinfo},{dpinfo},Server,Server,{'_'.join([str(s) for s in sl_list])},AggEval,{curr_epoch},-1,{len(dataloader)},{lr},{np.mean(loss_list)},{accuracy},{timetaken}\n")
        logger.info("Server Evaluation Aggregation Accuracy for %s Epoch is %s",curr_epoch,np.sum(pred == target) * 100 / pred.shape[0])
        aggregation_accuracy = float(accuracy)
    return aggregation_accuracy

def fit_split_model_Sequentialy(args,workers: list,dataname:str,experimentinfo:str,train_dataset_key:str,test_dataset_key:str,experimentname:str,
                   resultFolder:str=resultFolder,stepoch=0,modelFolder=modelFolder,testdatacount=-1,logger=logger):
    # We are Excpecting each worker have same number of classes
    logger.info("Start Sequential  Training with mode %s ",args.mode)
    IID,aplha=workers[0].getIIDAndAplha()
    iid={0:"Equal800",1:"2 Class",2:"Different",3:"Equal5000",4:"Equal3200",5:"FULL/4",6:"Equal128",7:"FULL/7",8:f"Dirichlet_Full_{aplha}",10:"Same5000",11:"Diff5000",12:"Dirichlet_42_12",20:"FULL/20",50:"Test50"}[IID]
    stepoch,model,optimizer,scheduler,datasize,nclasses,batchcount,alldpinfo=dataModelSetup(args,workers,  train_dataset_key, test_dataset_key,   experimentname,  stepoch=stepoch, modelFolder=modelFolder, logger=logger)
    experimentinfo+=f",{nclasses},{iid},{args.randomNoise}"
    trainbatchcounts,evalbatchcount=batchcount
    for epoch in range(stepoch+1,args.epoches+1):
        dosavemodel = args.save_model and (epoch == args.epoches or (epoch > 0 and epoch % args.save_model_per_epoch == 0))
        doeval = args.aggrigate_evaluation and (epoch == args.epoches or (epoch > 0 and epoch % args.aggrigate_eval_per_epoch == 0))
        douploadaggrigate = args.uploadaggrigate and (epoch > 0 and epoch % args.uploadaggrigate_per_epoch == 0)
        doeval=doeval or (args.mode in ("Sequential1","Sequential2") and epoch > 0 and epoch % args.uploadperepoch == 0)
        doaggrigateeval=doeval or douploadaggrigate or dosavemodel
        for ci,(sl,worker) in enumerate(zip(args.sl,workers)):
            if epoch in worker.skipEpoches:
                logger.info("Skipping Epoch %s for %s ",epoch,worker.name)
                continue
            dpinfo=",".join([str(v) for v in alldpinfo[worker.name].values()])
            worker.connect()
            if args.savePowerLogs: worker.addWorkerCsvLog(f"Training for Epoch {epoch}  Starts")
            workername=IP2HOST[worker.name.rsplit('_', 1)[0]]+"_"+worker.name.rsplit('_', 1)[1]
            logger.info("Training Epoch: %d, with worker: %s Result File %s", epoch,workername,f"Result_{experimentname}.csv")
            if args.mode=="Sequential1": UploadModelParametrs(model, worker, sl, logger)
            model.setStartEnd(start=sl)
            for sepoch in range(args.sub_epoches):
                logger.info("Training  Worker %s Epoch %s, Client %s  Model %s Dataset %s SL %s", workername, epoch, ci + 1,model.getName(),args.dataset, sl)
                loss_list, pred_list, target_list = [], [], []
                st=time.time()
                for b in tqdm(range(trainbatchcounts[worker.name]),desc=f"Training Epoch {epoch}, Client {ci+1}:"):
                    logger.debug("Requesting for midoutput")
                    (midoutput,target),midoutsize=worker.getMiddleOutput(dataset_key=train_dataset_key,returnResponseSize=True)
                    if args.savePowerLogs: worker.addWorkerCsvLog(f"Send MidOutput Finish Received {midoutsize} MB Data")
                    midoutput=worker.Reconstruct(midoutput)
                    logger.debug("Midoutput recieved")
                    optimizer.zero_grad()
                    midoutput=midoutput.to(worker.device).requires_grad_()
                    logger.debug("Change to device")
                    pred=model(midoutput)
                    logger.debug("Process local model")
                    loss=loss_fn(pred,target.to(worker.device))
                    logger.debug("calculate loss")
                    loss.backward()
                    optimizer.step()
                    logger.debug("Update local model")
                    grads=worker.Decomposition(midoutput.grad.cpu())
                    if args.savePowerLogs: worker.addWorkerCsvLog("Send Gradient Start")
                    logger.debug("Sending Gradient")
                    _,gradsize=worker.updateGrads(grads=grads,returnSentSize=True)
                    logger.debug("Gradient Sent")
                    if args.savePowerLogs: worker.addWorkerCsvLog(f"Gradient Sent with size of {gradsize} MB Data")
                    logger.debug("Sent Gradient Back")
                    midoutput.detach()
                    loss_list.append(loss.detach().cpu().numpy())
                    pred_list.append(torch.argmax(pred,dim=1).detach().cpu().numpy())
                    target_list.append(target.numpy())
                timetaken=time.time()-st
                pred,target=np.concatenate(pred_list),np.concatenate(target_list)
                accuracy=np.sum(pred==target)/pred.shape[0]
                with open(f"{resultFolder}/Result_{experimentname}.csv", "a") as f:
                    f.write(f"{experimentinfo},{dpinfo},{IP2HOST[worker.name.rsplit('_', 1)[0]]},{worker.name},{sl},Training,{epoch},{sepoch + 1},{trainbatchcounts[worker.name]},{optimizer.param_groups[0]['lr']},{np.mean(loss_list)},{accuracy},{timetaken}\n")
                logger.info("Training Finish for Model: %s, Epoch: %s, Worker :%s, Client:%s, with  avg_loss: %s, accuracy %s", model.getName(),epoch,workername,ci+1,np.mean(loss_list),accuracy)
            if epoch == args.epoches or epoch % args.evalaute_model_per_epoch == args.evalaute_model_per_epoch-1:
                nextworkers=workers[:-1] if doaggrigateeval and ci==len(workers)-1 else [workers[(ci+1)%len(workers)]]
                evaluate_split_model(worker,nextworkers,ci,  evalbatchcount[worker.name],optimizer.param_groups[0]['lr'],dpinfo,args.savePowerLogs,test_dataset_key, model, epoch, f"{resultFolder}/Result_{experimentname}.csv", worker.device,sl,logger=logger,expariment=experimentname,experimentinfo=experimentinfo)
            if len(workers)>1 and (args.mode=="SSL" or (args.mode in ("Sequential1","Sequential2") and epoch > 0 and epoch % args.uploadperepoch == 0)):
                model=downloadModelParametrs(model,worker,logger)
                if args.mode in ("SSL","Sequential2"):
                    #Upload to next client in case current client is last one will update weights to first
                    nextwprkerindex=(ci + 1) % len(workers)
                    logger.info(f"Uploading Weights from {worker.name} To {workers[nextwprkerindex].name}")
                    UploadModelParametrs(model, workers[nextwprkerindex], args.sl[nextwprkerindex], logger)
            worker.addWorkerCsvLog(f"Training for Epoch {epoch} experiment {experimentname} Finish")
            # if args.getPowerLogs: # No Used
            #     SaveServerLogs(worker,logger,expariment=experimentname,KASA=args.PowerLogger=="KASA")
            if scheduler is not None:
                worker.stepScheduler()
            if args.powerSaver and (not doaggrigateeval or ci<len(workers)-1): worker.suspend()
        if scheduler is not  None:
            scheduler.step()
        if doaggrigateeval:
            aggregation_accuracy = evaluate_Aggregated_model(workers, doeval, douploadaggrigate,dosavemodel,workers[0].device,IID ,dpinfo,args.mode,args.dataset,testdatacount, datasize, args.powerSaver, args.savePowerLogs, args.test_batch_size, optimizer.param_groups[0]['lr'], args.modelName, model,optimizer,scheduler, epoch,
                                  f"{resultFolder}/Result_{experimentname}.csv",  args.sl,args.aggregationmethod, logger=logger,expariment=experimentname,experimentinfo=experimentinfo)
            reassign_p3sl_noise_if_needed(args, workers, aggregation_accuracy, logger, curr_epoch=epoch)
    logger.info("All Training Finish with %s",experimentname)


def SplitTrainingForClients(args,clients,logger):
    for client in clients:
        if args.savePowerLogs:
            assert client.name.rsplit('_', 1)[0] in IP2HOST, "Update Hosttoclient IP in Util.py"
            logger.info("Resting PowerConsumptionLoads for %s..", client.name)
            client.ResetPowerConsumptionAndLogs()
            client.addWorkerCsvLog(f"Training Start")

    torch.manual_seed(args.seed)

    experiment = f"{datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S')}"
    fitkwargs=dict(args=args,workers=clients,dataname=args.dataset,train_dataset_key=args.dataset+"_train",test_dataset_key=args.dataset+"_test",
        experimentname=experiment,resultFolder=resultFolder,modelFolder=modelFolder,testdatacount=args.aggtestdatacount,logger=logger,)

    if args.useExpariment is not None and os.path.exists("Results/"+args.useExpariment):
        df=pd.read_csv(resultFolder+args.useExpariment)[::-1]
        datainfo=None
        for i,d in df.iterrows():
            if d["Worker"]=="Server":
                datainfo=d
                break
        if datainfo is not None:
            expname=args.useExpariment.split(".")[0].split("Result_")[-1]
            modelname=f"Models/{args.dataset}_{datainfo['ModelName']}"
            sstround=datainfo["Epoch"]//args.save_model_per_epoch*args.save_model_per_epoch
            if not os.path.exists(f"{modelname}_Server_{expname}.pt"):
                logger.info(f"{modelname}_Server_{expname}.pt not found")
            if args.mode !="SSL":
                for ci, h, p in zip(range(1, 50), args.hosts, args.ports):
                    if not os.path.exists(f"{modelname}_Client_{'_'.join(h.split('.'))}_{p}_{ci}_{expname}.pt"):
                        print(f"{modelname}_Client_{'_'.join(h.split('.'))}_{p}_{ci}_{expname}.pt not found")
            if os.path.exists(f"{modelname}_Server_{expname}.pt") and (args.mode=="SSL" or all(os.path.exists(f"{modelname}_Client_{'_'.join(h.split('.'))}_{p}_{ci}_{expname}.pt") for ci,h,p in zip(range(1,50),args.hosts,args.ports))):
                logger.info("All Model Found")
                experiment=expname
                fitkwargs.update({"experimentname":experiment,"stepoch":int(sstround)})
                if datainfo["UseDp"]:
                    dpscales=[df.loc[i,"Dpscale"] for i in [0,2,4,6]]
                    for worker, dp in zip(clients, dpscales):
                        logger.info("Initalizing Noise %s for %s", dp, worker.name)
                        worker.setDPInfo(dpinfo={"usedp": True, "dpmethod": "Laplace", "dploc": 0.0, "dpscale": dp})
                for k,v in {"modelName":datainfo["ModelName"],"dataset":datainfo["Dataset"],"sl":[int(s) for s in datainfo["Split layer"].split('_')],"batch_size":int(datainfo["BatchSize"]),"usedecomposition":datainfo["Decomposition"],"aggrigate_evaluation":datainfo["Aggregation"],"uploadaggrigate":datainfo["AggregationUploadPerEpoch"]!=-1}.items():
                    args.__setattr__(k,v)
                if datainfo["Decomposition"]:
                    for k, v in {"decompositionmethod": datainfo["DecomMethod"], "decompositionrank": datainfo["DecomRank"]}.items():
                        args.__setattr__(k, v)
                if args.aggrigate_evaluation:
                    for k, v in {"aggrigate_eval_per_epoch": datainfo["AggregationEpoch"], "aggregationmethod": datainfo["AggregationMethod"]}.items():
                        args.__setattr__(k, v)
                if args.uploadaggrigate:
                    for k, v in {"uploadaggrigate_per_epoch": datainfo["AggregationUploadPerEpoch"], "aggregationmethod": datainfo["AggregationMethod"]}.items():
                        args.__setattr__(k, v)
                logger.info("Model and log found at %s running with arguments %s",args.useExpariment,fitkwargs)
        else:
            logger.info("No Server Worker Found")
    if "stepoch" not in fitkwargs:
        if args.useExpariment is not None:
            logger.info("Not abeling to find and load model please try without useExpariment")
            exit()
        logger.info("Model File Not Provided so start from scratch")
        with open(f"{resultFolder}Result_{experiment}.csv", "w") as f:
            f.write("ModelName,Optimizer,LR,L2,Schedular,PreTrain,Dataset,BatchSize,Decomposition,DecomMethod,DecomRank,SuspenWakeUp,Aggregation,AggregationEpoch,AggregationMethod,AggregationUploadPerEpoch,Upload_MODE,PowerLogger,SEQUploadperEpoch,SkipEpochs,Classes,Data_Partition,RandomNoise,UseDp,Dpmethod,DpLoc,Dpscale,ClientName,Worker,Split layer,Train_Mode,Epoch,Sub_Epoch,BatchCount,Learning_Rate,Loss,Accuracy,TimeTaken\n")
    fitkwargs["experimentinfo"]=(f"{args.modelName},{args.optimizer},{args.lr},{args.l2},{args.schedular},{args.preTrain},{args.dataset},{args.batch_size},{args.usedecomposition},{args.decompositionmethod},{args.decompositionrank},{args.powerSaver},{args.aggrigate_evaluation},{args.aggrigate_eval_per_epoch},{args.aggregationmethod},{args.uploadaggrigate_per_epoch},{args.mode},{args.PowerLogger},{args.uploadperepoch},{args.skipEpochs}")
    logger.info("Recording results at file %s",f"{resultFolder}Result_{experiment}.csv")
    if args.mode=="ParallellN":
        fit_split_model_Parallelly(**fitkwargs)
    else:
        fit_split_model_Sequentialy(**fitkwargs)
    for client in clients:
        client.addWorkerCsvLog(f"Training END")
        client.savePowerConsumptionAndLogs(experiment)
    return experiment

def DownloadJTOPstatPowerConsumptions(args,clients,experimentname,expariment,timetaken,exparimentlist,logger):
    # This for the case where we use Jtop for energy consumption
    logger.info(f"Downloading Energy Consumption for {expariment} With JTOP Stat")
    if args.savePowerLogs:
        EnergyInfo = []
        for index,client, sl in zip(range(100),clients, args.sl):
            exparimentlist[client.name].append((expariment, sl))
            poweconsargs = client.getPowerConsumption(expariment, sl)
            clientName = IP2HOST[client.name.rsplit('_', 1)[0]]
            dpnoise = args.dpNoise[index] if args.dpNoise is not None else 0
            EnergyInfo.append((clientName, client.name, expariment, sl,dpnoise, HostPower[clientName], *poweconsargs))
        EnergyDf = pd.DataFrame(EnergyInfo,columns=["ClientName", "ClientAddress", "Experiment", "Split Layer","DPNoise", "ClientPower",
                                         "Communication", "Computation","TotalCommunicationTime", "TotalComputationTime","TotalData(MB)","MaxEnergy"])
        EnergyDf["Total"] = EnergyDf["Communication"] + EnergyDf["Computation"]
        EnergyDf["BaseExperimentName"] = experimentname
        EnergyDf["BatchSize"] = args.batch_size
        EnergyDf["ModelName"] = args.modelName
        EnergyDf["DataName"] = args.dataset
        EnergyDf["Epoches"] = args.epoches
        EnergyDf["AggMode"] = args.mode
        EnergyDf["aggrigate"] = args.aggrigate_evaluation and args.epoches>=args.aggrigate_eval_per_epoch
        EnergyDf["TimeTaken"] = timetaken[-1]
        EnergyDf["RandomNoise"] = args.randomNoise
        dffile = f"{SplitProfilingfFolder}{expariment}.csv"
        EnergyDf.to_csv(dffile, index=None)
        logger.info("SpProfiling saved at %s", dffile)
    return exparimentlist, dffile


def DownloadKASKAPowerConsumptions(args,clients,experimentname,expariment,timetaken,exparimentlist,logger):
    try:
        clientLog=None
        logger.info(f"Downloading Energy Consumption for {expariment} KASKA")
        if args.savePowerLogs:
            EnergyInfo = []
            for index,client, sl in zip(range(100),clients, args.sl):
                exparimentlist[client.name].append((expariment, sl))
                clientLog=client.getExperimentLOGS(experiment=expariment)
                clientName = IP2HOST[client.name.rsplit('_', 1)[0]]
                poweconsargs=calculatePowerConsumptions(clientName,expariment,clientLog,sl)
                dpnoise=args.dpNoise[index] if args.dpNoise is not None else 0
                EnergyInfo.append((clientName, client.name, expariment, sl,dpnoise, HostPower[clientName], *poweconsargs))
            EnergyDf = pd.DataFrame(EnergyInfo,columns=["ClientName", "ClientAddress", "Experiment", "Split Layer","DPNoise", "ClientPower",
                                             "Communication", "Computation","TotalCommunicationTime", "TotalComputationTime","TotalData(MB)","MaxEnergy"])
            EnergyDf["Total"] = EnergyDf["Communication"] + EnergyDf["Computation"]
            EnergyDf["BaseExperimentName"] = experimentname
            EnergyDf["BatchSize"] = args.batch_size
            EnergyDf["ModelName"] = args.modelName
            EnergyDf["DataName"] = args.dataset
            EnergyDf["Epoches"] = args.epoches
            EnergyDf["AggMode"] = args.mode
            EnergyDf["aggrigate"] = args.aggrigate_evaluation and args.epoches>=args.aggrigate_eval_per_epoch
            EnergyDf["TimeTaken"] = timetaken[-1]
            EnergyDf["RandomNoise"] = args.randomNoise
            dffile = f"{SplitProfilingfFolder}{expariment}.csv"
            EnergyDf.to_csv(dffile, index=None)
            logger.info("SpProfiling saved at %s",dffile)
            return exparimentlist,dffile
    except Exception as e:
        logger.critical("Isse in DownloadKASKAPowerConsumptions %s",e)
        if clientLog is not None:
            clientLog.to_csv(f"{SplitProfilingfFolder}{expariment}_TEMP.csv")
            logger.info(f"clientLog Saved at {SplitProfilingfFolder}{expariment}_TEMP.csv")
            logger.info("clientLog" + str(clientLog))
        raise e
    return exparimentlist

def DownLoadPowerConsumption(args,clients, experimentname, expariment, timetaken, exparimentlist, logger):
    if args.savePowerLogs:
        if args.PowerLogger == "KASA":
            return DownloadKASKAPowerConsumptions(args, clients, experimentname, expariment, timetaken, exparimentlist, logger)
        else:
            return DownloadJTOPstatPowerConsumptions(args, clients, experimentname, expariment, timetaken, exparimentlist, logger)
    return None,None

def SubSplitTraing(args,clients,exparimentlist):
    experimentname = datetime.datetime.now().strftime('%y_%m_%d_%H_%M_%S')
    timetaken,DFLIST = [],[]
    if args.sllist is not None:
        for i,sll in enumerate(args.sllist):
            if args.dpNoiseList is not None:
                args.dpNoise=[float(s) for s in args.dpNoiseList[i].split("_")]
                for worker, dp in zip(clients, args.dpNoiseList[i].split("_")):
                    logger.info("Initalizing Noise from dpNoiseList %s for %s", dp, worker.name)
                    worker.setDPInfo(dpinfo={"usedp": True, "dpmethod": "Laplace", "dploc": 0.0, "dpscale": float(dp)})
            sttime = time.time()
            args.sl = [int(s) for s in sll.split("_")]
            for h, p, s in zip(args.hosts, args.ports, args.sl):
                logger.info(f"Using {h}:{p} with {s} Split layer")
            expariment = SplitTrainingForClients(args, clients, logger)
            timetaken.append(time.time() - sttime)
            _,dffile=DownLoadPowerConsumption(args, clients, experimentname, expariment, timetaken, exparimentlist, logger)
            if args.savePowerLogs: DFLIST.append(dffile)
    else:
        sttime = time.time()
        expariment = SplitTrainingForClients(args, clients, logger)
        timetaken.append(time.time() - sttime)
        _,dffile=DownLoadPowerConsumption(args, clients, experimentname, expariment, timetaken, exparimentlist, logger)
        if args.savePowerLogs: DFLIST.append(dffile)

    # clientdict={client.name:client for client in clients}
    if args.savePowerLogs and args.sllist is not None:
        logger.info("Combining all Power logs")
        for cname, slexperments in exparimentlist.items():
            expariments = [exp for exp, sl in slexperments]
            sllist = [sl for exp, sl in slexperments]
            # EnergyDf=clientdict[cname].splitPointProfileing(expariments,sllist)
            EnergyDf = SplitPointProfileingLocal(expariments, sllist)
            clientName = IP2HOST[cname.rsplit('_', 1)[0]]
            EnergyDf["ClientPower"] = HostPower[clientName]
            # if len(expariments)>0: experimentname=experimentname+"_".join([exp.split("_")[-1] for exp in expariments])
            dffile = f"{SplitProfilingfFolder}{clientName}_{cname}_{experimentname}.csv"
            EnergyDf.to_csv(dffile, index=None)
            logger.info("SplitProfiling Saved at %s", dffile)
            logger.info("expariment %s", expariments)
            logger.info("SL List %s", sllist)
            logger.info("Energy is")
            logger.info(EnergyDf)
            logger.info(f"Saved at {dffile}")
    if args.savePowerLogs:
        print("SPProfiling Saved at",DFLIST)


def SplitTraining(args, logger):
    if args.verbose: logger.setLevel(level=logging.DEBUG)
    if args.powerSaver:wakeUpAll(args.macaddress,logger=logger)
    if args.savePowerLogs: MakeTempDataFrame()
    use_cuda = args.cuda and torch.cuda.is_available()
    # use_cuda=torch.cuda.is_available()
    device = torch.device("cuda:"+",".join(str(args.cudadevice[0])) if use_cuda else "cpu")
    logger.info("Using device %s",device)
    if use_cuda:
        args.cudadevice=args.cudadevice[:torch.cuda.device_count()]
        if args.mode != "ParallellN" and len(args.cudadevice)>0:
            args.cudadevice=args.cudadevice[:1]
    else:
        args.cudadevice = [-1]

    logger.info("Creating workers")
    workerargs=dict(powerSaver=args.powerSaver,logger=logger,usecompression=args.Compression,usedecomposition=args.usedecomposition,decompositionmethod=args.decompositionmethod,decompositionrank=args.decompositionrank,verbose=args.verbose)
    clients=[WebsocketClientWorkerCustom(host=host,port=port,macaddress=mac,deviceid=args.cudadevice[index%len(args.cudadevice)],reconnect=not args.noreconnect,**workerargs) for index,host,port,mac in zip(range(100),args.hosts,args.ports,args.macaddress)]
    if args.skipEpochs:
        for clientcount,sepoches in args.skipEpochsList.items():
            if len(clients)>=clientcount:
                clients[clientcount-1].skipEpoches=sepoches
    if args.dpNoise is not None:
        for worker,dp in zip(clients,args.dpNoise):
            logger.info("Initalizing Noise %s for %s",dp,worker.name)
            worker.setDPInfo(dpinfo={"usedp": True,"dpmethod": "Laplace", "dploc": 0.0, "dpscale": dp})
    configure_p3sl_from_clients(args, clients, logger)
    # alice = WebsocketClientWorkerCustom(id=args.id, port=args.port,logger=logger, **kwargs_websocket)
    exparimentlist={clinet.name:[] for clinet in clients}
    SubSplitTraing(args, clients, exparimentlist)

    # Close all Connections
    for client in clients:
        client.close()
    time.sleep(15)


def TrainingMain(args,logger):
    for i in range(args.BaseRounds):
        print(f"Running Start for {i+1}th Round")
        if args.usedecomposition:
            for r in range(1, 31):
                print(f"Running for Round {i+1} Rank {r}")
                args.sllist=None
                args.decompositionrank = r
                SplitTraining(args, logger)
        else:
            print(f"Running for Round {i + 1}")
            SplitTraining(args, logger)
    print(f"Running Finish for {args.BaseRounds} Rounds")


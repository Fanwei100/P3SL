"""Parallel split-model training driver.

Drives several client workers concurrently using threads: each worker
performs its local forward pass and uploads activations in parallel, the
coordinator finishes the forward pass and computes gradients, and the
backward pass is dispatched back to all workers. Implements the parallel
counterpart to :func:`Training.fit_split_model_Sequentialy`.
"""

import threading

import torch,time
from tqdm import tqdm
import numpy as np
from DataSet import getDataSet

from Util import dataModelSetup,modelFolder,logger,resultFolder,addCsvLog,suspendAll,evaluate_split_model
from Util import loss_fn,SaveServerLogs,IP2HOST

def modelsAggregation(workers,ModelInfolist,  aggmethod="AggEqual", logger=logger, expariment="Testing"):
    assert aggmethod in ["AggEqual","AggClientOnly"], str(aggmethod)+" is not allowed aggregation method"
    logger.info("Aggregation Model with %s",aggmethod)
    addCsvLog("Model aggrigation Start",expariment)

    modeldict,dtypes = {},{}
    for modelinfo in ModelInfolist:
        for k,v in modelinfo["Model"].state_dict().items():
            if k in modeldict:
                modeldict[k].append(v.float().cpu())
            else:
                modeldict[k]=[v.float().cpu()]
                dtypes[k]=v.dtype
            # print(k,v)
    aggparampeter={k:torch.mean(torch.stack(v),dim=0).to(dtypes[k]) for k,v in modeldict.items()}
    for modelinfo,worker in zip(ModelInfolist,workers):
        modelinfo["Model"].load_state_dict(aggparampeter)
        modelinfo["Model"]=modelinfo["Model"].to(worker.device)
    return ModelInfolist



def evaluate_Aggregated_model(workers, doeval, douploadaggrigate,saveModel,dpinfo,trainingmode, dataset, testdatacount,datasize, powerSaver,savePowerLogs, batch_size, lr, modelName, ModelInfolist, curr_epoch, resultFile, sl_list,aggregationmethod,logger, expariment,experimentinfo):
    # Create and send train config
    loss_list,pred_list,target_list=[],[],[]
    logger.info("Model Aggregation and Evaluation.....")
    if savePowerLogs: addCsvLog("Downloading Parameters Start",expariment=expariment)
    logger.info("Saving Model for %s Epoch", curr_epoch)

    logger.info("Downloading model parameters")
    for ci, (worker,modelInfo) in enumerate(zip(workers,ModelInfolist)):
        worker.connect()
        model= modelInfo["Model"]
        logger.info("Downloading model parameters from %s",worker.name)
        model.load_state_dict(worker.getModelPatameter(),strict=False)
        if saveModel:
            logger.info("Saving Model for Worker %s Epoch %s",worker.name, curr_epoch)
            torch.save(model.state_dict(),f"{modelFolder}{dataset}_{modelName}_Client_{worker.name}_{ci + 1}_{expariment}.pt")
            torch.save(worker.getOptimizerPatameter(), f"{modelFolder}{dataset}_{modelName}_Client_{worker.name}_{ci + 1}_{expariment}_opt.pt")
            torch.save( {"optimizer":modelInfo["Optimizer"],"scheduler":modelInfo["Optimizer"]},f"{modelFolder}{dataset}_{modelName}_Server_{worker.name}_{ci + 1}_{expariment}_opt.pt")
    if savePowerLogs: addCsvLog("Downloading Parameters Finish",expariment=expariment)
    if douploadaggrigate or doeval:
        ModelInfolist=modelsAggregation(workers,ModelInfolist,  aggmethod=aggregationmethod,logger=logger, expariment=expariment)
    if douploadaggrigate:
        for worker,sl,modelinfo in zip(workers,sl_list,ModelInfolist):
            modelinfo["Model"].setStartEnd(0,sl)
            # model.setEnd(sl)
            if savePowerLogs: addCsvLog("Sending Parameter start", expariment=expariment)
            logger.info("Sending parameter to %s",worker.name)
            worker.uploadModelParameter({k: v.cpu() for k, v in modelinfo["Model"].getTrainedParameter().items()})

    if powerSaver: suspendAll(workers,logger)
    if doeval:
        dataloader=torch.utils.data.DataLoader(getDataSet(dataset,datasize=(datasize,datasize),testsize=testdatacount,nclasses=None)[1],batch_size=batch_size)
        st=time.time()
        if savePowerLogs: addCsvLog("Server Model Evaluation Start",expariment=expariment)
        model=ModelInfolist[0]["Model"]
        model.setStartEnd(0)
        for data,target in tqdm(dataloader):
            pred=model(data.to(workers[0].device))
            pred_list.append(torch.argmax(pred,dim=1).detach().cpu().numpy())
            target_list.append(target.numpy())
            loss_list.append(loss_fn(pred,target.to(workers[0].device)).detach().cpu().numpy())
        pred,target=np.concatenate(pred_list),np.concatenate(target_list)

        accuracy = np.sum(pred == target) / pred.shape[0]
        timetaken=time.time()-st
        if savePowerLogs: addCsvLog("Model Evaluation Finish",expariment=expariment)
        with open(resultFile, "a") as f:
            f.write(f"{experimentinfo},{dpinfo},Server,Server,{'_'.join([str(s) for s in sl_list])},AggEval,{curr_epoch},-1,{len(dataloader)},{lr},{np.mean(loss_list)},{accuracy},{timetaken}\n")
        logger.info("Server Evaluation Aggregation Accuracy for %s Epoch is %s",curr_epoch,np.sum(pred == target) * 100 / pred.shape[0])


def TrainEpoch(args,epoch,ci,worker,workers,dpinfo,trainbatchcounts,train_dataset_key,sl,model,optimizer,scheduler,experimentname,experimentinfo,doaggrigateeval,evalbatchcount,test_dataset_key):
    model.setStartEnd(sl)
    for sepoch in range(args.sub_epoches):
        logger.info("Worker %s Training Epoch %s, Client %s ", worker.name, epoch, ci + 1)
        loss_list, pred_list, target_list = [], [], []
        st = time.time()
        for b in tqdm(range(trainbatchcounts[worker.name]),desc=f"Training Epoch {epoch}, Client {ci + 1}:"):
            logger.debug("Requesting for midoutput")
            (midoutput, target), midoutsize = worker.getMiddleOutput(dataset_key=train_dataset_key,returnResponseSize=True)
            if args.savePowerLogs: worker.addWorkerCsvLog(f"Send MidOutput Finish Received {midoutsize} MB Data")
            midoutput = worker.Reconstruct(midoutput)
            logger.debug("Midoutput recieved")
            optimizer.zero_grad()
            midoutput = midoutput.to(worker.device).requires_grad_()
            logger.debug("Change to device")
            pred = model(midoutput)
            logger.debug("Process local model")
            loss = loss_fn(pred, target.to(worker.device))
            logger.debug("calculate loss")
            loss.backward()
            optimizer.step()
            logger.debug("Update local model")
            grads = worker.Decomposition(midoutput.grad.cpu())
            if args.savePowerLogs: worker.addWorkerCsvLog("Send Gradient Start")
            _, gradsize = worker.updateGrads(grads=grads, returnSentSize=True)
            if args.savePowerLogs: worker.addWorkerCsvLog(f"Gradient Sent with size of {gradsize} MB Data")
            logger.debug("Sent Gradient Back")
            midoutput.detach()
            loss_list.append(loss.detach().cpu().numpy())
            pred_list.append(torch.argmax(pred, dim=1).detach().cpu().numpy())
            target_list.append(target.numpy())
        timetaken = time.time() - st
        pred, target = np.concatenate(pred_list), np.concatenate(target_list)
        accuracy = np.sum(pred == target) / pred.shape[0]
        with open(f"{resultFolder}/Result_{experimentname}.csv", "a") as f:
            f.write(
                f"{experimentinfo},{dpinfo},{IP2HOST[worker.name.rsplit('_', 1)[0]]},{worker.name},{sl},Training,{epoch},{sepoch + 1},{trainbatchcounts[worker.name]},{optimizer.param_groups[0]['lr']},{np.mean(loss_list)},{accuracy},{timetaken}\n")
        logger.info("Training Finish for Model: %s, Epoch: %s, Worker :%s, Client:%s, with  avg_loss: %s, accuracy %s",
                    model.getName(), epoch, worker.name, ci + 1, np.mean(loss_list), accuracy)

    if epoch == args.epoches or epoch % args.evalaute_model_per_epoch == args.evalaute_model_per_epoch - 1:
        nextworkers = workers[:-1] if doaggrigateeval and ci == len(workers) - 1 else [workers[(ci + 1) % len(workers)]]
        evaluate_split_model(worker, nextworkers, ci, evalbatchcount[worker.name], optimizer.param_groups[0]['lr'],dpinfo,
                             args.savePowerLogs, test_dataset_key, model, epoch,
                             f"{resultFolder}/Result_{experimentname}.csv", worker.device, sl, logger=logger,
                             expariment=experimentname, experimentinfo=experimentinfo)
    worker.addWorkerCsvLog(f"Training for Epoch {epoch} experiment {experimentname} Finish")
    if args.getPowerLogs:
        SaveServerLogs(worker, logger, expariment=experimentname,KASA=args.PowerLogger=="KASA")
    worker.stepScheduler()
    if args.powerSaver and (not doaggrigateeval or ci < len(workers) - 1): worker.suspend()
    scheduler.step()

def fit_split_model_Parallelly(args,workers: list,dataname:str,experimentinfo:str,train_dataset_key:str,test_dataset_key:str,experimentname:str,
                    nchannels:int=3,resultFolder:str=resultFolder,stepoch=0,modelFolder=modelFolder,testdatacount=1600,logger=logger):
    # We are Excpecting each worker have same number of classes
    logger.info("Start Parrallel Training")
    iid=["Equal","2 Class","Different","Equal5000"][workers[0].getIID()]
    ModelInfolist,datasize,nclasses,batchcount,alldpinfo=dataModelSetup(args,workers,  train_dataset_key, test_dataset_key, experimentname,    stepoch=stepoch, modelFolder=modelFolder, logger=logger,multipleModels=True)
    experimentinfo+=f",{nclasses},{iid}"
    trainbatchcounts,evalbatchcount=batchcount
    for epoch in range(stepoch+1,args.epoches+1):
        dosavemodel = args.save_model and (epoch == args.epoches or (epoch > 0 and epoch % args.save_model_per_epoch == 0))
        doeval = args.aggrigate_evaluation and (epoch == args.epoches or (epoch > 0 and epoch % args.aggrigate_eval_per_epoch == 0))
        douploadaggrigate = args.uploadaggrigate and (epoch > 0 and epoch % args.uploadaggrigate_per_epoch == 0)
        doeval=doeval or (args.mode in ("Sequential1","Sequential2") and epoch > 0 and epoch % args.uploadperepoch == 0)
        doaggrigateeval=doeval or douploadaggrigate or dosavemodel
        threadList=[]
        for ci,(sl,worker,modelinfo) in enumerate(zip(args.sl,workers,ModelInfolist)):
            worker.connect()
            dpinfo = ",".join([str(v) for v in alldpinfo[worker.name].values()])
            model, optimizer, scheduler=modelinfo["Model"],modelinfo["Optimizer"], modelinfo["schedular"]
            if args.savePowerLogs: worker.addWorkerCsvLog(f"Training for Epoch {epoch}  Starts")
            logger.info("Training Epoch: %d, with worker: %s for Exp: %s Result File %s", epoch, worker.name,experimentname,f"Result_{experimentname}.csv")
            TrainEpoch(args, epoch, ci, worker, workers,dpinfo, trainbatchcounts, train_dataset_key, sl, model, optimizer, scheduler,
                        experimentname, experimentinfo, doaggrigateeval, evalbatchcount, test_dataset_key)
            # thread=threading.Thread(target=TrainEpoch,args=(args, epoch, ci, worker, workers, trainbatchcounts, train_dataset_key, sl, model, optimizer,scheduler,experimentname, experimentinfo, doaggrigateeval, evalbatchcount, test_dataset_key))
            # thread.start()
            # threadList.append(thread)
        # for thread in threadList: thread.join()
        if doaggrigateeval:
            evaluate_Aggregated_model(workers, doeval, douploadaggrigate,dosavemodel,dpinfo, args.mode,args.dataset,testdatacount, datasize, args.powerSaver, args.savePowerLogs, args.test_batch_size, optimizer.param_groups[0]['lr'], args.modelName, ModelInfolist, epoch,
                                  f"{resultFolder}/Result_{experimentname}.csv",  args.sl,args.aggregationmethod, logger=logger,expariment=experimentname,experimentinfo=experimentinfo)
    logger.info("All Training Finish with %s",experimentname)




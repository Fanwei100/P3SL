"""Shared utilities for the coordinator side.

Houses the configuration constants (output folders, host/IP maps, loss
function), evaluation helpers, model save/load wrappers, KASA log
post-processing routines, and the small bookkeeping helpers
(:func:`addCsvLog`, :func:`MakeTempDataFrame`, ...) used throughout
:mod:`Training` and :mod:`ParralelTraining`.
"""

import copy
import os,logging,datetime,torch,pandas as pd,time
import shutil
import threading
from pytz import timezone
import pandas

from train_config import TrainConfig
from tqdm import tqdm
import numpy as np
from Models import getModels



logger = logging.getLogger(__name__)


LOG_INTERVAL = 25
loss_fn=torch.nn.CrossEntropyLoss()

resultFolder="Results/"
modelFolder="Models/"
csvFolder=f"CSV/"
SplitProfilingfFolder="Results/SPProfiling/"
KASAEnergyLogs=os.getenv("HOME")+"/KASA/Energy_log.csv"
KASALOGS="CSV/KASALOGS/"
os.makedirs(KASALOGS,exist_ok=True)
os.makedirs(resultFolder,exist_ok=True)
os.makedirs(SplitProfilingfFolder,exist_ok=True)
os.makedirs(modelFolder,exist_ok=True)
os.makedirs(csvFolder,exist_ok=True)

KASAClientScoketID={"alice":2,"alice1":3,"alice2":5,"alice3":4,"raspberry1":0,"raspberry2":1,"adam":6,"local":-1}
HOST2IP={"local": "0.0.0.0", "alice": "192.168.4.224", "alice1": "192.168.4.177", "alice2": "192.168.4.176", "alice3": "192.168.4.225", "raspberry1": "192.168.4.223", "raspberry2": "192.168.4.220", "adam": "192.168.4.59"}
# HOST2IP={"local":"0.0.0.0","alice":"[2600:4040:a7a9:5200:3374:52f0:84a3:66bc]","alice1":"[2600:4040:a7a9:5200:4c51:a14a:1094:3af2]","alice2":"[2600:4040:a7a9:5200:642f:a48:1a82:5493]","alice3":"192.168.4.38"}
IP2HOST={v.replace(".", "_"):k for k,v in HOST2IP.items()} # Replacing . with _ because . not allowed in path and we have save files with ip
HostPower={"local":"Not Defined","alice":"10W","alice1":"10W","alice2":"10W","alice3":"10W","raspberry1":"Not Defined","raspberry2":"Not Defined","adam":"Not Defined"}

def _build_optimizer( optimizer_name: str, model, optimizer_args: dict) -> torch.optim.Optimizer:
    """Build an optimizer if needed.

    Args:
        optimizer_name: A string indicating the optimizer name.
        optimizer_args: A dict containing the args used to initialize the optimizer.
    Returns:
        A Torch Optimizer.
    """

    if optimizer_name in dir(torch.optim):
        optimizer = getattr(torch.optim, optimizer_name)
        optimizer_args.setdefault("params", model.parameters())
        optimizer = optimizer(**optimizer_args)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    return optimizer


def addCsvLog(message,expariment):
    with open(f"{csvFolder}Log_Local_{expariment}.csv","a") as f:
        f.writelines(f"{datetime.datetime.now()},{message}\n")

def suspendAll(workers,logger=logger):
    logger.info("Suspending All expect First...")
    for worker in workers[1:]:
        worker.suspend()




def evaluate_split_model(worker,nextworkers,clientid, batchcount,lr,dpinfo,savePowerLogs,test_dataset_key, model, epoch,resultFile,device,sl,logger,expariment,experimentinfo):
    # Create and send train configThis is done to keep them informed about the predicted performance in the coming days. Guidance is also referred to as 'forward-looking statements' and 'earnings guidance'.
    loss_list,pred_list,target_list=[],[],[]
    logger.info("Start Evaluation for Epoch %s Worker %s SL %s .....",epoch,worker.name,sl)
    st=time.time()
    if savePowerLogs: addCsvLog("Model Evaluation Start",expariment=expariment)
    model.setStartEnd(sl)
    for b in tqdm(range(batchcount),desc=f"Testing Epoch {epoch}, Client {clientid + 1}:"):
        if b==batchcount-1:
            for nextworker in nextworkers:
                if nextworker.powerSaver:
                    logger.info("Wakeup Next worker.. %s",nextworker.name)
                    threading.Thread(target=nextworker.wakeUp).start()
        logger.debug("Requesting for midoutput")
        (midoutput, target),midoutsize = worker.getMiddleOutput(dataset_key=test_dataset_key,train=False,returnResponseSize=True)
        if savePowerLogs: worker.addWorkerCsvLog(f"Send MidOutput Finish Received {midoutsize} MB Data")
        midoutput = worker.Reconstruct(midoutput)
        pred=model(midoutput.to(device))
        pred_list.append(torch.argmax(pred,dim=1).detach().cpu().numpy())
        target_list.append(target.numpy())
        loss_list.append(loss_fn(pred,target.to(device)).detach().cpu().numpy())
    pred,target=np.concatenate(pred_list),np.concatenate(target_list)
    accuracy = np.sum(pred == target) / pred.shape[0]
    timetaken=time.time()-st
    if savePowerLogs: addCsvLog("Model Evaluation Finish",expariment=expariment)
    with open(resultFile, "a") as f:
        f.write(f"{experimentinfo},{dpinfo},{IP2HOST[worker.name.rsplit('_', 1)[0]]},{worker.name},{sl},Evaluation,{epoch},-1,{batchcount},{lr},{np.mean(loss_list)},{accuracy},{timetaken}\n")
    logger.info("Evaluation Accuracy for %s Epoch %s worker and client %s is %s",epoch,worker.name,clientid+1,np.sum(pred == target) * 100 / pred.shape[0])

def LoadKasaLogs(clientName,clientLog,expariment,SaveLogsLocally=True,DROPExtra=True):
    if SaveLogsLocally:
        LogFile = f"{KASALOGS}LOG_{expariment}.csv"
        clientLog.to_csv(LogFile, index=None)
        PowerFile = f"{KASALOGS}Power_{expariment}.csv"
    else:
        PowerFile = f"{KASALOGS}Power_{expariment}_TEMP.csv"
    clientLog["Time"] = clientLog["Time"].map(pd.to_datetime)
    shutil.copy(KASAEnergyLogs, PowerFile)
    power_log = pd.read_csv(PowerFile)
    power_log=power_log[power_log["Socket"]==KASAClientScoketID[clientName]]
    power_log = power_log.rename(columns={"Timestamp": "Time"})
    power_log["Watts"]=power_log["Power (W)"]
    power_log["Time"] = power_log["Time"].map(pd.to_datetime)
    if DROPExtra:
        expStart, expEnd = clientLog.loc[0, "Time"] - pandas.Timedelta(10, "s"), clientLog.loc[
            clientLog.index[-1], "Time"] + pandas.Timedelta(10, "s")
        power_log = power_log[(expStart < power_log["Time"]) & (expEnd > power_log["Time"])]
    if SaveLogsLocally:
        power_log.to_csv(PowerFile, index=None)
    else:
        os.remove(PowerFile)  # Removing Temporyr File
    power_log = power_log[["Time", "Watts"]]
    return clientLog,power_log


def SaveServerLogs(worker,logger,expariment,KASA=False):
    raise Exception("Check it Properly Before Use")
    logger.info("Saving ServerLog for experiment",expariment)
    if KASA:
        clientLog=worker.getExperimentLOGS(experiment=expariment)
        clientName = IP2HOST[worker.name.rsplit('_', 1)[0]]
        clientLog, power_log=LoadKasaLogs(clientName,clientLog,expariment,SaveLogsLocally=True,DROPExtra=True)
        results={"Power":power_log,"Log":clientLog}
    else:
        results=worker.getAndRestPowerConumptionAndLogs()
    beforesize=os.path.getsize(f"{csvFolder}Power_{expariment}_{worker.name}.csv") if os.path.exists(f"{csvFolder}Power_{expariment}_{worker.name}.csv") else 0
    with open(f"{csvFolder}Power_{expariment}_{worker.name}.csv","a") as f:
        f.writelines(results["Power"])
    with open(f"{csvFolder}Log_{expariment}_{worker.name}.csv","a") as f:
        worker.addWorkerCsvLog("Start Downloading Power Logs")
        f.writelines(results["Log"])
        worker.addWorkerCsvLog("Stop Downloading Power Logs")
    # assert os.path.getsize(f"{csvFolder}Power_{expariment}_{worker.name}.csv") > 0, f"The Power Consumption Stop for the client {worker.name}"
    assert beforesize<os.path.getsize(f"{csvFolder}Power_{expariment}_{worker.name}.csv") > 0, f"The Power Consumption Stop for the client {worker.name}"

    logger.info("Saving ServerLog Finish")



def UploadModelParametrs(model,worker,sl,logger):
    model.setStartEnd(0,sl)
    # model.setEnd(sl)
    logger.info("Sending parameter to %s", worker.name)
    worker.uploadModelParameter({k: v.cpu() for k, v in model.getTrainedParameter().items()})

def downloadModelParametrs(model,worker,logger):
    logger.info("Downloading model parameters from %s", worker.name)
    parameter = worker.getModelPatameter()
    model.load_state_dict(parameter,strict=False)
    logger.info("Local Model Updated with paramepter of  %s", worker.name)
    return model

def BuildOptimizerAndSchedular(args,model,train_config,optimizerdict):
    optimizer = _build_optimizer(train_config.optimizer, model, optimizer_args=train_config.optimizer_args)
    if optimizerdict["optimizer"] is not None:
        optimizer.load_state_dict(optimizerdict["optimizer"])
    if train_config.scheduler_args is not None:
        del train_config.scheduler_args["name"]
        if args.schedular == "CosineAnnealingLR":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **train_config.scheduler_args)
        elif args.schedular == "OneCycleLR":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, **train_config.scheduler_args)
        if scheduler is not None and optimizerdict["scheduler"] is not None:
            scheduler.load_state_dict(optimizerdict["scheduler"])
    else:
        scheduler=None
    return optimizer,scheduler

def dataModelSetup(args,workers: list,train_dataset_key:str,test_dataset_key:str,experimentname:str,
                    stepoch:int=0,modelFolder:str=modelFolder,logger=logger,multipleModels:bool=False):
    datasize,nclasses=workers[0].getSizeANdClasses(dataset_key=args.dataset+"_train")
    if datasize=="Error":
        logger.error(nclasses)
        exit()
    logger.info("Datasize %d nclasses %d",datasize,nclasses)
    model=getModels(args.modelName, nchannel=args.nchannels,nclasses=nclasses,pretrained=args.preTrain)
    modellist=[]
    trainbatchcounts,evalbatchcount,dpinfo = {},{},{}
    for ci,sl,worker in zip(range(1,50),args.sl,workers):
        dpinfo[worker.name]=worker.getDPInfo()
        train_batchcount =worker.getBatchCount(dataset_key=train_dataset_key,batch_size=args.batch_size)
        test_batchcount =worker.getBatchCount(dataset_key=test_dataset_key,batch_size=args.batch_size)
        trainbatchcounts[worker.name],evalbatchcount[worker.name]=train_batchcount,test_batchcount
        logger.info("Train Test loader for %s build Train have %s batches Test have %s batches",worker.name,train_batchcount,test_batchcount)
        model.setStartEnd(0,sl)
        # model.setEnd(sl)
        if args.optimizer=="SGD":
            optimizer_args=dict(lr=args.lr,momentum=0.9, weight_decay=args.l2)
        elif args.optimizer=="Adam":
            optimizer_args=dict(lr=args.lr, amsgrad=True, weight_decay=args.l2)

        if args.schedular=="CosineAnnealingLR":
            scheduler_args={"name":args.schedular, "T_max": args.epoches, "last_epoch": stepoch - 1}
        elif args.schedular=="OneCycleLR":
            scheduler_args={"name":args.schedular, "max_lr":args.lr,"epochs": args.epoches, "last_epoch": stepoch - 1,"steps_per_epoch":1}
        else:
            scheduler_args =None
        trainconfigdict=dict(device="cpu", batch_size=args.batch_size, optimizer=args.optimizer, optimizer_args=optimizer_args,scheduler_args=scheduler_args)
        if args.useJit:
            tracemodel = torch.jit.trace(model, torch.zeros([1, args.nchannels, datasize, datasize], dtype=torch.float))
            train_config = TrainConfig(model=tracemodel.cpu(),**trainconfigdict)
        else:
            train_config = TrainConfig(model=model.cpu(),**trainconfigdict)
        train_config.send(worker)
        logger.info("Train Config sent for %s with sl %s",worker.name,sl)
        if stepoch>0:
            if args.mode=="SSL":
                modelpath = f"{modelFolder}{args.dataset}_{args.modelName}_Server_{experimentname}"
            else:
                modelpath = f"{modelFolder}{args.dataset}_{args.modelName}_Client_{worker.name}_{ci}_{experimentname}"
            if os.path.exists(f"{modelpath}.pt"):
                model = getModels(args.modelName, nchannel=args.nchannels, nclasses=nclasses,pretrained=args.preTrain)
                logger.info("Loading model for %s from %s", worker.name,modelpath)
                worker.uploadModelParameter(torch.load(modelpath+".pt",map_location=worker.device))
                worker.uploadOptimizerParameter(torch.load(modelpath+"_opt.pt",map_location=worker.device))
                # model.load_state_dict(torch.load(modelpath+".pt"))
                logger.info("Model  parameter uploaded for client %s ",worker.name)
            else:
                raise Exception(f"Unable To Load model for server from {modelFolder}{args.dataset}_{args.modelName}_Client_{worker.name}_{ci}_{experimentname}")
                stepoch=0
        worker.buildOptimizer()
        logger.info("Optimizer Build for %s",worker.name)
        if args.powerSaver: worker.suspend()
        if multipleModels:
            modellist.append(model.to(worker.device))
            model = getModels(args.modelName, nchannel=args.nchannels, nclasses=nclasses, pretrained=args.preTrain)


    if multipleModels:
        modelInfo=[]
        for model in modellist:
            optimizer, scheduler = BuildOptimizerAndSchedular(args, model, copy.deepcopy(train_config))
            modelInfo.append({"Model": model, "Optimizer": optimizer, "schedular": scheduler})
        if stepoch > 0:
            raise Exception("Check this part")
            for ci, modelinfo, worker in zip(range(1, 50),modellist, workers):
                model, optimizer, scheduler = modelinfo["Model"], modelinfo["Optimizer"], modelinfo["schedular"]
                modelpath = f"{modelFolder}{args.dataset}_{args.modelName}_Server_{worker.name}_{ci}_{experimentname}"
                model.load_state_dict(torch.load(modelpath + ".pt"))
                optimizerdict = torch.load(modelpath + "_opt.pt", map_location=worker.device)
                optimizer.load_state_dict(optimizerdict["optimizer"])
                if scheduler is not None:
                    scheduler.load_state_dict(optimizerdict["scheduler"])

        os.makedirs("Models", exist_ok=True)
        torch.save(modelInfo[0],f"Models/ModelBase_{experimentname}.pt")
        return stepoch,modelInfo, datasize, nclasses, (trainbatchcounts, evalbatchcount),dpinfo
    else:
        model=model.to(worker.device)
        optmizerWeights={"optimizer":None,"scheduler":None}
        if stepoch > 0:
            modelpath = f"{modelFolder}{args.dataset}_{args.modelName}_Server_{experimentname}"
            if os.path.exists(f"{modelpath}.pt"):
                logger.info("Found Model for Epoch %s starting from their", stepoch + 1)
                logger.info("Loading Model from %s", modelpath)
                model.load_state_dict(torch.load(modelpath + ".pt", map_location=worker.device))
                optmizerWeights = torch.load(modelpath + "_opt.pt", map_location=worker.device)
            else:
                raise Exception(f"Unable To Load model for server from {modelpath}")
                stepoch = 0
        optimizer, scheduler=BuildOptimizerAndSchedular(args,model,train_config,optmizerWeights)
        os.makedirs("Models",exist_ok=True)
        torch.save({"Model":model,"Optimizer":optimizer,"schedular":scheduler},f"Models/ModelBase_{experimentname}.pt")
        return stepoch,model,optimizer,scheduler,datasize,nclasses,(trainbatchcounts,evalbatchcount),dpinfo


def MakeTempDataFrame(nminutes=10):
    if "parmpal" not in os.getcwd(): return
    tz = timezone('US/Eastern')
    base = datetime.datetime.now(tz).replace(tzinfo=None)
    date_list = [base + datetime.timedelta(seconds=x) for x in range(nminutes*60)]
    Powerw=np.random.random(size=[len(date_list)])*4
    DATA={"Timestamp":date_list,"Socket":KASAClientScoketID["local"],"Alias":"Plug 0","Power (W)":Powerw}
    pd.DataFrame(DATA).to_csv(KASAEnergyLogs,index=None)



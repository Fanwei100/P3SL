"""WebSocket server worker (data-owner side).

Implements :class:`WebsocketServerWorkerCustom`, the per-device process that
hosts the client-side portion of the split neural network, accepts a
connection from the training coordinator, and performs local forward / backward
passes on each mini-batch. Also implements activation (de)serialization,
optional zlib compression, optional Tucker / PARAFAC decomposition, and
P3SL activation-noise injection on the uploaded intermediate representations.
"""

import datetime,ssl,websockets,asyncio,logging,binascii,math
import json,base64,zlib,os
import time,threading
from pathlib import Path
import pandas as pd
from pytz import timezone
import torch,io
from torch.utils.data import SequentialSampler
from Worker import WorkerCommandMessage
import tensorly as tl
from tensorly.decomposition import tucker,parafac
from SplitPointProfiling import SplitPointProfileing,getPowerConsumption,getOptimimalSplitPoint
from P3SLPrivacyOptimization import ensure_privacy_table, select_split_point
tl.set_backend('pytorch')

logger=logging.getLogger(__name__)

class WebsocketServerWorkerCustom:
    def __init__(
        self,
        host: str,
        port: int,
        name:str,
        IID:str,
        alpha:float,
        privacy_alpha:float=0.5,
        log_msgs: bool = False,
        usecompression:bool=False,
        logger=logger,
        verbose: bool = False,
        loop=None,
        cert_path: str = None,
        key_path: str = None,
        useDp:bool=False,
        dpmethod:str="Laplace",
        dploc:float=0,
        dpscal:float=.1
    ):

        """This is a Custome extension to normal workers wherein
        all messages are passed over websockets. Note that because
        BaseWorker assumes a request/response paradigm, this worker
        enforces this paradigm by default.

        Args:
            hook (sy.TorchHook): a normal TorchHook object
            id (str or id): the unique id of the worker (string or int)
            log_msgs (bool): whether or not all messages should be
                saved locally for later inspection.
            verbose (bool): a verbose option - will print all messages
                sent/received to stdout
            host (str): the host on which the server should be run
            port (int): the port on which the server should be run
            data (dict): any initial tensors the server should be
                initialized with (such as datasets)
            loop: the asyncio event loop if you want to pass one in
                yourself
            cert_path: path to used secure certificate, only needed for secure connections
            key_path: path to secure key, only needed for secure connections
        """
        self.host=host
        self.port=port
        self.name=name
        self.IID=IID
        self.alpha=alpha
        self.privacy_alpha=privacy_alpha
        self.cert_path=cert_path
        self.key_pat=key_path
        self.datasets=dict()
        self.data_loader = dict()
        self.data_iterator = dict()
        self.nclasses=dict()
        self.datasize=dict()
        self.verbose=verbose
        self.log_msgs=log_msgs
        self.ResultFolder="CSVResults/"
        self.csvBase="CSV/"
        self.PowerConsumptions=self.csvBase+"PowerConsumptions.csv"
        self.PowerConsumptionsStats=self.csvBase+"PowerConsumptions_stats.csv"
        self.Logfile=self.csvBase+"Logs.csv"
        self.LogfileStats=self.csvBase+"Logs_Stats.csv"
        self.broadcast_queue = asyncio.Queue()
        if loop is None:
            loop = asyncio.new_event_loop()
        self.loop=loop
        self.optimizer=None
        self.usecompression = usecompression
        self.logger=logger
        self._message_router = {
            WorkerCommandMessage: self.execute_worker_command,
        }
        self.usedecomposition=False
        self.decompositionConfig=None
        self.optimizerparameters=None
        os.makedirs(self.ResultFolder,exist_ok=True)
        self.dpinfo={"usedp":useDp,"dpmethod":dpmethod,"dploc":dploc,"dpscale":dpscal}
        self.setDp()
        Path(self.csvBase).mkdir(exist_ok=True,parents=True)


    def setDp(self):
        if self.dpinfo["usedp"]:
            try:
                method = self.dpinfo["dpmethod"]
                sigma = float(self.dpinfo["dpscale"])
                # P3SL reports the personalized noise level as sigma with
                # Lap(0, sigma^2). PyTorch's Laplace distribution expects the
                # scale b, where Var = 2 b^2; therefore b = sigma / sqrt(2).
                if method == "Laplace":
                    scale = sigma / math.sqrt(2.0)
                    dist = torch.distributions.Laplace
                elif method == "Gaussian":
                    scale = sigma
                    dist = torch.distributions.Normal
                else:
                    raise ValueError(f"Unsupported activation-noise method: {method}")
                self.dpDist = dist(loc=float(self.dpinfo["dploc"]), scale=scale)
                self.logger.info(
                    "Using %s activation noise with loc %s, P3SL sigma %s, torch scale/std %s",
                    method, self.dpinfo["dploc"], sigma, scale,
                )
            except Exception as exc:
                self.logger.info("Not able to set activation-noise distribution, so noise will not be added: %s", exc)
                self.dpDist = None
                self.dpinfo["usedp"]=False
        else:
            self.dpDist=None
            # self.logger.info("No Noise will be added as usedp is False")


    def addCsvLog(self,message):
        tz = timezone('US/Eastern')
        with open(self.Logfile,"a") as f:
            f.write(f"{datetime.datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S.%f')},{message}\n")

    def execute_worker_command(self, message: tuple):
        """Executes commands received from other workers.

        Args:
            message: A tuple specifying the command and the args.

        Returns:
            A pointer to the result.
        """
        command_name = message.command_name
        args_, kwargs_, return_ids = message.message
        response = getattr(self, command_name)(*args_, **kwargs_)
        #  TODO [midokura-silvia]: send the tensor directly
        #  TODO this code is currently necessary for the async_fit method in websocket_client.py
        return response


    async def _consumer_handler(self, websocket: websockets.WebSocketCommonProtocol):
        """This handler listens for messages from WebsocketClientWorker
        objects.
        Args:
            websocket: the connection object to receive messages from and
                add them into the queue.

        """
        try:
            while True:
                msg = await websocket.recv()
                await self.broadcast_queue.put(msg)
        except websockets.exceptions.ConnectionClosed as e:
            self.logger.error("Exception in _consumer_handler %s",e)
            # await self._consumer_handler(websocket)

    async def _producer_handler(self, websocket: websockets.WebSocketCommonProtocol):
        """This handler listens to the queue and processes messages as they
        arrive.
        Args:
            websocket: the connection object we use to send responses
                back to the client.
        """
        message,response="",""
        try:
            while True:
                # get a message from the queue
                message = await self.broadcast_queue.get()
                # convert that string message to the binary it represent
                message = binascii.unhexlify(message[2:-1])
                # process the message
                response = self._recv_msg(message)
                # convert the binary to a string representation
                # (this is needed for the websocket library)
                response = str(binascii.hexlify(response))
                # send the response
                await websocket.send(response)
        except Exception as e:
            self.logger.warning("Exception _producer_handler %s",e)
            # self.logger.warning("resonse:- %s\nmessage:- %s ",response,message)
            raise e

    def torchSeralize(self, tensor):
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        if self.usecompression:
            return zlib.compress(buffer.getvalue())
        else:
            return buffer.getvalue()

    def torchdeseralize(self, serialized_data):
        try:
            if self.usecompression:
                serialized_data = zlib.decompress(serialized_data)
            return torch.load(io.BytesIO(serialized_data))
        except Exception as e:
            return torch.load(io.BytesIO(serialized_data))

    def seralize(self, data):
        return self.torchSeralize(data)

    def deseralize(self, data):
        return self.torchdeseralize(data)

    def recv_msg(self, bin_message: bin) -> bin:
        """Implements the logic to receive messages.
        The binary message is deserialized and routed to the appropriate
        function. And, the response serialized the returned back.
        Every message uses this method.
        Args:
            bin_message: A binary serialized message.

        Returns:
            A binary message response.
        """

        # Step -1: save message if log_msgs ==  True
        if self.log_msgs:
            self.msg_history.append(bin_message)
        # Step 0: deserialize message
        msg = self.deseralize(bin_message)
        if self.verbose:
            print(f"worker {self.name} received {msg.command_name}")

        # Step 1: route message to appropriate function
        response = self._message_router[type(msg)](msg)

        # Step 2: Serialize the message to simple python objects
        bin_response = self.seralize(response)

        return bin_response

    def _recv_msg(self, message: bin) -> bin:
        try:
            return self.recv_msg(message)
        except Exception as e:
            raise e

    async def _handler(self, websocket: websockets.WebSocketCommonProtocol, *unused_args):
        """Setup the consumer and producer response handlers with asyncio.

        Args:
            websocket: the websocket connection to the client

        """
        asyncio.set_event_loop(self.loop)
        consumer_task = asyncio.ensure_future(self._consumer_handler(websocket))
        producer_task = asyncio.ensure_future(self._producer_handler(websocket))

        done, pending = await asyncio.wait(
            [consumer_task, producer_task], return_when=asyncio.FIRST_COMPLETED
        )

        for task in pending:
            task.cancel()


    def start(self):
        """Start the server"""
        # Secure behavior: adds a secure layer applying cryptography and authentication
        if not (self.cert_path is None) and not (self.key_path is None):
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(self.cert_path, self.key_path)
            start_server = websockets.serve(
                self._handler,
                self.host,
                self.port,
                ssl=ssl_context,
                max_size=None,
                ping_timeout=None,
                close_timeout=None,
            )
        else:
            # Insecure
            start_server = websockets.serve(
                self._handler,
                self.host,
                self.port,
                max_size=None,
                ping_timeout=None,
                close_timeout=None,
            )

        asyncio.get_event_loop().run_until_complete(start_server)
        self.logger.info("Serving. Press CTRL-C to stop.")
        try:
            asyncio.get_event_loop().run_forever()
        except KeyboardInterrupt:
            logging.info("Websocket server stopped.")

    def Decomposition(self,X):
        if self.usedecomposition:
            self.logger.info("Client Decomposition")
            self.addCsvLog("Client Decomposition Start")
            if self.decompositionConfig["Method"]=="CP":
                X=parafac(X,rank=self.decompositionConfig["Rank"])
            elif self.decompositionConfig["Method"] == "Tucker":
                X=tucker(X, rank=self.decompositionConfig["Rank"])
            self.addCsvLog("Client Decomposition Finish")
        return X

    def Reconstruct(self,X):
        if self.usedecomposition:
            self.addCsvLog("Client Reconstruction Start")
            if self.decompositionConfig["Method"]=="CP":
                X=tl.cp_to_tensor(X)
            elif self.decompositionConfig["Method"] == "Tucker":
                X=tl.tucker_to_tensor(X)
            self.addCsvLog("Client Reconstruction Finish")
        return X


    def setConfig(self,configs):
        configs=json.loads(configs)
        self.usecompression=configs["usecompression"]
        self.usedecomposition=configs["usedecomposition"]
        msg="Using Compression" if self.usecompression else "Not Using Compression"
        if self.usedecomposition and "decompositionConfig" in configs:
            if "Rank" in configs["decompositionConfig"] and "Method" in configs["decompositionConfig"] and configs["decompositionConfig"]["Method"] in ["CP","Tucker"]:
                self.decompositionConfig = configs["decompositionConfig"]
                return msg+" AND Using Decompresion"
            else:
                self.usedecomposition=False
                return msg+" AND not Using Decompresion"


    def add_dataset(self, dataset, key: str,ncalsses:int,datasize:int):
        if key not in self.datasets:
            self.datasets[key] = dataset
            self.nclasses[key]=ncalsses
            self.datasize[key]=datasize
        else:
            raise ValueError(f"Key {key} already exists in Datasets")

    def getSizeANdClasses(self,dataset_key):
        self.addCsvLog("Get Data Size and Classes")
        if dataset_key not in self.datasize:
            return "Error",f"This ip is for the {list(self.datasets.keys())[0]} but you are requestion {dataset_key}"
        return self.datasize[dataset_key],self.nclasses[dataset_key]



    def addDPNoice(self,output):
        # This Function is used to add noise to intermdiate output.
        # we generate the same shape of intermedate out put and add that to intermediate output.
        if self.dpDist is None:
            return output
        self.logger.debug("Noise Is Added with %s",self.dpDist)
        return output + self.dpDist.sample(output.shape).to(device=output.device, dtype=output.dtype)

    def resetIterator(self,dataset_key):
        self.data_iterator[dataset_key] = iter(self.data_loader[dataset_key])

    def getMiddleOutput(self, dataset_key,train=True):
        try:
            self.addCsvLog("Loading Data Start")
            self.logger.debug("Receive request for midoutput %s", "Train" if train else "Test")
            if dataset_key not in self.data_iterator:
                raise ValueError(f"Dataset {dataset_key} unknown.")
            self.logger.debug("Model parameter %s",list(self.model.parameters())[1][20:26])
            try:
                data, target = next(self.data_iterator[dataset_key])
            except StopIteration:
                self.data_iterator[dataset_key]=iter(self.data_loader[dataset_key])
                data, target = next(self.data_iterator[dataset_key])
            self.logger.debug("get data from iterator")
            self.addCsvLog("Loading Data Finish")
            self.addCsvLog("Forward Propagation Start")
            if train:
                self.model.train()
                self.optimizer.zero_grad()
                self.logger.debug("Start processing model with shape %s",data.shape)
                self.output = self.model(data.to(self.device))
                self.logger.debug("Send output")
                output=self.addDPNoice(self.output.detach())
                # here we call the addDPNoise method just before send the intemediate output
                self.addCsvLog("Forward Propagation Finish")
                output=self.Decomposition(output)
                self.addCsvLog("Send MidOutput Start")
                return output,target
            else:
                with torch.no_grad():
                    self.logger.debug("Start processing model with shape %s", data.shape)
                    output = self.model(data.to(self.device))
                    self.logger.debug("Send output")
                    self.addCsvLog("Forward Propagation Finish")
                    output=self.Decomposition(output)
                    self.addCsvLog("Send MidOutput Start")
                    return output, target
        except Exception as e:
            self.logger.critical("Exception in midoutput %s",str(e))
            raise e

    def updateGrads(self, grads):
        self.addCsvLog("Send Gradient Finish")
        grads=self.Reconstruct(grads)
        self.addCsvLog("Update Gradient Start")
        self.output.backward(grads)
        self.optimizer.step()
        self.addCsvLog("Update Gradient Finish")

    def setDPInfo(self,dpinfo):
        self.addCsvLog("Adding Noise "+str(dpinfo))
        self.logger.info("Adding Noise %s",dpinfo)
        self.dpinfo.update(**dpinfo)
        self.setDp()


    def getDPInfo(self):
        return self.dpinfo

    def getBatchCount(self,dataset_key,batch_size):
        self.buildDataloader(dataset_key,batch_size)
        return len(self.data_loader[dataset_key])

    def buildDataloader(self,dataset_key,batch_size):
        self.addCsvLog("Build Dataloader Start")
        data_range = range(len(self.datasets[dataset_key]))
        sampler = SequentialSampler(data_range)
        self.data_loader[dataset_key] = torch.utils.data.DataLoader(
            self.datasets[dataset_key],
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
        )
        self.data_iterator[dataset_key]=iter(self.data_loader[dataset_key])
        self.addCsvLog(f"Build Dataloader Finish for {dataset_key}")


    def _build_optimizer(self, optimizer_name: str, model, optimizer_args: dict,scheduler_args:dict) -> torch.optim.Optimizer:
        """Build an optimizer if needed.

        Args:
            optimizer_name: A string indicating the optimizer name.
            optimizer_args: A dict containing the args used to initialize the optimizer.
        Returns:
            A Torch Optimizer.
        """
        try:
            if self.optimizer is None and optimizer_name in dir(torch.optim):
                optimizer = getattr(torch.optim, optimizer_name)
                optimizer_args.setdefault("params", model.parameters())
                self.optimizer = optimizer(**optimizer_args)
            elif self.optimizer is None :
                raise ValueError(f"Unknown optimizer: {optimizer_name}")
            if self.optimizerparameters is not None:
                try:
                    for k,v in self.optimizer.state_dict()["param_groups"][0].items():
                        if k not in self.optimizerparameters["optimizer"]["param_groups"][0]:
                            print("SSS Added", k, v)
                            self.optimizerparameters["optimizer"]["param_groups"][0][k]=v
                except:pass
                self.optimizer.load_state_dict(self.optimizerparameters["optimizer"])
                self.logger.info("Optimizer weights Uploaded")
                self.addCsvLog("Optimizer weights Updated")
            if scheduler_args is not None:
                schedulername=scheduler_args["name"]
                del scheduler_args["name"]
                self.scheduler=getattr(torch.optim.lr_scheduler, schedulername)(self.optimizer, **scheduler_args)
                # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, **scheduler_args)
                if self.optimizerparameters is not None:
                    self.scheduler.load_state_dict(self.optimizerparameters["scheduler"])
                    self.logger.info("scheduler weights Uploaded")
                    self.addCsvLog("scheduler weights Updated")
            else:
                self.scheduler=None
            self.logger.info("Optimizer and schedular build")
            return self.optimizer
        except Exception as e:
            self.logger.error("Exception in _build_optimizer %s",e)
            raise e


    def stepScheduler(self):
        self.addCsvLog("Add Schedular step")
        if self.scheduler is not None:
            self.scheduler.step()

    def buildOptimizer(self):
        self.addCsvLog("Build Optimizer Start")
        self._build_optimizer(self.train_config["optimizer"], self.model, optimizer_args=self.train_config["optimizer_args"],scheduler_args=self.train_config["scheduler_args"])
        self.addCsvLog("Build Optimizer Finish")


    def setTrainConfig(self, trainconfig):
        self.logger.debug("getModelParameter")
        try:
            loaded_raw_model = base64.b64decode(trainconfig["model"].encode('utf-8'))
            try:
                self.model=torch.jit.load(io.BytesIO(loaded_raw_model))
            except Exception as e:
                self.model = torch.load(io.BytesIO(loaded_raw_model))
            del trainconfig["model"]
            self.train_config=trainconfig
            self.device=torch.device(self.train_config["device"] if torch.cuda.is_available() else "cpu")
        except Exception as e:
            print("Exception",e)
            self.logger.error("Exception in setTrainConfig %s",e)
            raise e

    def getModelParameter(self):
        self.logger.debug("getModelParameter")
        return self.model.getTrainedParameter()


    def getOptimizerParameter(self):
        return {"optimizer":self.optimizer.state_dict(),"scheduler":self.scheduler.state_dict() if self.scheduler is not None else {}}

    def setModelEndLayer(self,end):
        self.logger.debug("setModelEndLayer %s",end)
        self.model.setStartEnd(0,end)

    def uploadModelParameter(self,parameter):
        self.logger.debug("uploadModelParameter")
        try:
            self.model.load_state_dict(parameter,strict=False)
        except Exception as e:
            self.logger.warning("Exception in uploadModelParameter %s",e)
            raise e

    def getIID(self):
        return self.IID

    def getIIDAndAplha(self):
        return self.IID,self.alpha

    def uploadOptimizerParameter(self,parameter):
        self.optimizerparameters = parameter

    def getAndRestPowerConumptionAndLogs(self):
        self.logger.info("getAndRestPowerConumptionAndLogs")
        with open(self.PowerConsumptions,"r") as f:
            powerData=f.readlines()
        with open(self.Logfile,"r") as f:
            logData=f.readlines()
        self.ResetPowerConsumptionAndLogs()
        return {"Power":powerData,"Log":logData}

    def ResetPowerConsumptionAndLogs(self):
        try:
            self.logger.info("ResetPowerConsumptionAndLogs")
            if os.path.exists(self.PowerConsumptions):
                with open(self.PowerConsumptionsStats,"a") as fw:
                    with open(self.PowerConsumptions) as fr:
                        fw.writelines(fr.readlines())
            if os.path.exists(self.Logfile):
                with open(self.LogfileStats,"a") as fw:
                    with open(self.Logfile) as fr:
                        fw.writelines(fr.readlines())
            open(self.PowerConsumptions,"w").close()
            open(self.Logfile,"w").close()
        except Exception as e:
            self.logger.info("Issue in ResetPowerConsumptionAndLogs %s ..",e)



    def _log_msgs(self, value):
        self.log_msgs = value

    def suspend(self):
        def createsuspendFile():
            time.sleep(2)
            self.logger.info("Suspending..... ", )
            self.addCsvLog("Suspending ")
            open("suspend_system","w").close()
        threading.Thread(target=createsuspendFile).start()

    def savePowerConsumptionAndLogs(self,expname):
        try:
            PowerConsumptionsStats=f"{self.ResultFolder}Power_{expname}.csv"
            LogfileStats=f"{self.ResultFolder}Log_{expname}.csv"
            self.logger.info("Saving PowerConsumption And Logs")
            if os.path.exists(self.PowerConsumptions):
                with open(PowerConsumptionsStats,"w") as fw:
                    with open(self.PowerConsumptions) as fr:
                        fw.writelines(fr.readlines())
            if os.path.exists(self.Logfile):
                with open(LogfileStats,"w") as fw:
                    with open(self.Logfile) as fr:
                        fw.writelines(fr.readlines())
            open(self.PowerConsumptions,"w").close()
            open(self.Logfile,"w").close()
        except Exception as e:
            self.logger.info("Issue in savePowerConsumptionAndLogs %s ..",e)

    def getMaximumPowerConsumed(self):
        try:
            with open(self.PowerConsumptions) as f:
                columnlen = len(f.readline().split(","))
            power_log = pd.read_csv(self.PowerConsumptions,names=["Time", "Watts", "Stime", "Ram"] + ["CPU" + str(i) for i in range(1, columnlen - 3)])
            return power_log["Watts"].max()
        except Exception as e:
            self.logger.info("Issue in getMaximumPowerConsumed %s ..", e)

    def splitPointProfileing(self,experimentlist,splitpoints):
        try:
            return SplitPointProfileing(explist=experimentlist,splitpoints=splitpoints)
        except Exception as e:
            self.logger.info("Issue in splitPointProfileing %s ..", e)

    def getPowerConsumption(self,experiment,splitpoint):
        try:
            return getPowerConsumption(experiment=experiment,splitpoint=splitpoint)
        except Exception as e:
            self.logger.info("Issue in splitPointProfileing %s ..", e)

    def getExperimentLOGS(self,experiment):
        try:
            LogfileStats = f"{self.ResultFolder}Log_{experiment}.csv"
            client_log = pd.read_csv(LogfileStats, names=["Time", "Message"])
            return client_log
        except Exception as e:
            self.logger.info("Issue in splitPointProfileing %s ..", e)


    def getPrivacyAlpha(self):
        return self.privacy_alpha

    def selectP3SLSplit(self, noise_assignment, privacy_leakage_table=None, smax=10, alpha=None,
                        max_power=None, energy_profile_path=None, energy_table=None):
        """Select the client split point using the P3SL lower-level objective.

        The privacy table and noise assignment are supplied by the server.
        The energy/power table is read locally by default, so the client does
        not need to disclose resource measurements to the server.
        """
        try:
            if privacy_leakage_table is None:
                privacy_leakage_table = ensure_privacy_table("Fsimmean.csv")
            if energy_table is None:
                candidate_paths = []
                if energy_profile_path is not None:
                    candidate_paths.append(energy_profile_path)
                candidate_paths.extend([
                    f"EnergyCsv/{self.name}.csv",
                    f"EnergyCsv/{self.name.rsplit('_', 1)[0]}.csv",
                    f"EnergyCsv/{self.name.split('_')[0]}.csv",
                ])
                for path in candidate_paths:
                    if path and os.path.exists(path):
                        energy_table = path
                        break
                if energy_table is None:
                    raise FileNotFoundError(
                        "No local energy table found. Provide energy_table or energy_profile_path "
                        "or create EnergyCsv/<client>.csv with Split Layer/Total/PeakPower columns."
                    )
            selected = select_split_point(
                privacy_table=privacy_leakage_table,
                noise_assignment=noise_assignment,
                energy_profile=energy_table,
                smax=smax,
                alpha=self.privacy_alpha if alpha is None else alpha,
                max_power=max_power,
            )
            self.logger.info("P3SL selected split=%s sigma=%s objective=%s", selected.split, selected.noise, selected.objective)
            self.addCsvLog(f"P3SL Select Split {selected.split} Noise {selected.noise} Objective {selected.objective}")
            # Return only the split/noise decision to the coordinator.  The
            # local objective, energy table, peak-power value, and privacy
            # sensitivity alpha_i remain on the client, as required by P3SL.
            return {"split": selected.split, "noise": selected.noise}
        except Exception as e:
            self.logger.info("Issue in selectP3SLSplit %s ..", e)
            raise e


    def getOptimimalSplitPoint(self,noicelevel,maxpoints,alpha=0.5):
        try:
            return getOptimimalSplitPoint(client=self.name,noicelevel=noicelevel,maxpoints=maxpoints,alpha=alpha)
        except Exception as e:
            self.logger.info("Issue in getOptimimalSplitPoint %s ..", e)
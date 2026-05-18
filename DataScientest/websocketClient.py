"""WebSocket client used by the coordinator to talk to one client worker.

Implements :class:`WebsocketClientWorkerCustom`, the thin per-worker proxy
that the coordinator instantiates to send commands (forward / backward,
parameter upload / download, evaluation requests) to a remote client worker
and to receive serialised tensors back over a WebSocket connection.
"""

import binascii,time
import json,torch
from typing import Union,List
from Worker import BaseWorker
import logging,datetime,websocket,ssl
from websocket._exceptions import WebSocketTimeoutException
from wakeonlan import send_magic_packet
import tensorly as tl
from tensorly.decomposition import tucker,parafac
tl.set_backend('pytorch')

class WebsocketClientWorkerCustom(BaseWorker):
    def __init__(
        self,
        host: str,
        port: int,
        macaddress:str,
        logger,
        deviceid:int=0,
        powerSaver:bool=False,
        usecompression:bool=False,
        usedecomposition:bool=False,
        decompositionmethod:str="Tucker",
        decompositionrank:int=10,
        reconnect:bool=False,
        secure: bool = False,
        name: Union[int, str] = None,
        is_client_worker: bool = False,
        log_msgs: bool = False,
        verbose: bool = False,
        data: List[Union[torch.Tensor]] = None,
    ):
        """A DataScientest which will forward all messages to a remote worker running a
        WebsocketServerWorker and receive all responses back from the server.
        """
        super().__init__(usecompression=False)
        self.device= torch.device("cuda:"+str(deviceid) if torch.cuda.is_available() and deviceid!=-1 else "cpu")
        self.host=host
        self.port=port
        self.macaddress=macaddress
        self.powerSaver=powerSaver
        self.name=name if name is not None else "_".join(host.split("."))+"_"+str(port)
        self.data=data
        self.is_client_worker=is_client_worker
        self.log_msgs=log_msgs
        self.verbose=verbose
        self.secure=secure
        self.reconnect=reconnect
        self.ws = None
        self.logger=logger
        self.connect()
        self.usedecomposition = usedecomposition
        if usedecomposition:
            assert decompositionmethod in ("CP","Tucker"),"Supported Decomposition methods are CP and Tucker"
        self.decompositionConfig={"Method":decompositionmethod,"Rank":decompositionrank} if self.usedecomposition else None
        self.InitalSetting(usecompression)
        self.skipEpoches=[]


    def InitalSetting(self,usecompression):
        try:
            self.logger.info("%s worker created ..",self.name)
            self.setConfig(usecompression,usedecomposition=self.usedecomposition,decompositionConfig=self.decompositionConfig)
        except Exception as e:
            self.logger.info("Exception in worker %s :- %s",self.name,e)
            self.connect()
            self.InitalSetting(usecompression)


    def setConfig(self,usecompression,usedecomposition=False,decompositionConfig=None):
        if usecompression:
            self.logger.info("Using Commpression while transfering for %s",self.name)
        else:
            self.logger.info("Using without Commpression while transfering for %s",self.name)
        if usedecomposition:
            self.logger.info("Using Decomposition %s while transfering for %s",decompositionConfig, self.name)
        else:
            self.logger.info("Using without Decomposition while transfering for %s", self.name)
        config=json.dumps(dict(usecompression=usecompression,usedecomposition=usedecomposition,decompositionConfig=decompositionConfig))
        response=self._send_msg_and_deserialize("setConfig",configs=config)
        self.logger.info("Config Sent and response is %s",response)
        self.usecompression=usecompression

    def Decomposition(self, X):
        if self.usedecomposition:
            self.addWorkerCsvLog(f"Local Decomposition Start")
            if self.decompositionConfig["Method"] == "CP":
                X=parafac(X, rank=self.decompositionConfig["Rank"])
            elif self.decompositionConfig["Method"] == "Tucker":
                X=tucker(X, rank=self.decompositionConfig["Rank"])
            self.addWorkerCsvLog(f"Local Decomposition Finish")
        return X

    def Reconstruct(self, X):
        if self.usedecomposition:
            self.addWorkerCsvLog(f"Local Reconstruction Start")
            if self.decompositionConfig["Method"] == "CP":
                X=tl.cp_to_tensor(X)
            elif self.decompositionConfig["Method"] == "Tucker":
                X=tl.tucker_to_tensor(X)
            self.addWorkerCsvLog(f"Local Reconstruction Finish")
        return X

    def _send_msg_and_deserialize(self, command_name: str,returnSentSize:bool=False,returnResponseSize:bool=False, *args, **kwargs):
        try:
            # self.logger.debug("_send_msg_and_deserialize")
            message = self.create_worker_command_message(command_name=command_name, *args, **kwargs)
            # self.logger.debug("Message Created")
            # Send the message and return the deserialized response.
            serialized_message = self.seralize(message)
            # self.logger.debug("Message Serialized to size of %s MB",len(serialized_message)/1e+6)
            response = self._send_msg(serialized_message)
            # self.logger.debug("Message sent Recived Response have size %s MB",len(response)/1e+6)
            # dresponse=self.deseralize(response)
            # if isinstance(dresponse,tuple) and isinstance(dresponse[0],torch.Tensor): print([(type(l),l.shape,l.dtype) for l in dresponse])
            if returnSentSize:
                return self.deseralize(response),len(serialized_message)/1e+6
            if returnResponseSize:
                return self.deseralize(response),len(response)/1e+6
            return self.deseralize(response)
        except WebSocketTimeoutException as e:
            self.logger.warning("Exception in _send_msg_and_deserialize %s",command_name,e)
            self._send_msg_and_deserialize(command_name, *args, ** kwargs)
        except Exception as e:
            self.logger.warning("Exception in _send_msg_and_deserialize %s %s",command_name,e)
            if self.reconnect:
                time.sleep(2)
                self.connect()
                self._send_msg_and_deserialize(command_name, *args, ** kwargs)
            else:
                raise e


    def sendTorchModel(self,model):
        serialized_model = self.torchSeralize(model)
        response = self._send_msg(json.dumps({"name":"model","value":serialized_model}))
        return self.deseralize(response)

    def buildOptimizer(self):
        return self._send_msg_and_deserialize("buildOptimizer")

    def stepScheduler(self):
        return self._send_msg_and_deserialize("stepScheduler")

    def getBatchCount(self,dataset_key,batch_size):
        return self._send_msg_and_deserialize("getBatchCount",dataset_key=dataset_key,batch_size=batch_size)

    def setDPInfo(self,dpinfo):
        return self._send_msg_and_deserialize("setDPInfo",dpinfo=dpinfo)

    def getDPInfo(self):
        return self._send_msg_and_deserialize("getDPInfo")

    def getSizeANdClasses(self,dataset_key):
        return self._send_msg_and_deserialize("getSizeANdClasses",dataset_key=dataset_key)

    def setModel(self,model):
        return self._send_msg_and_deserialize("setModel",model=model)


    def setTrainConfig(self,trainconfig):
        return self._send_msg_and_deserialize("setTrainConfig",trainconfig=trainconfig)

    def resetIterator(self,dataset_key):
        return self._send_msg_and_deserialize("resetIterator",dataset_key=dataset_key)


    def getMiddleOutput(self,dataset_key,train=True,returnResponseSize=False):
        return self._send_msg_and_deserialize("getMiddleOutput",returnResponseSize=returnResponseSize, dataset_key=dataset_key,train=train)

    def updateGrads(self,grads,returnSentSize=False):
        return self._send_msg_and_deserialize("updateGrads",returnSentSize=returnSentSize, grads=grads)

    def getModelPatameter(self):
        return self._send_msg_and_deserialize("getModelParameter")

    def getOptimizerPatameter(self):
        return self._send_msg_and_deserialize("getOptimizerParameter")

    def getIID(self):
        return self._send_msg_and_deserialize("getIID")

    def getIIDAndAplha(self):
        return self._send_msg_and_deserialize("getIIDAndAplha")

    def getPrivacyAlpha(self):
        return self._send_msg_and_deserialize("getPrivacyAlpha")

    def selectP3SLSplit(self, noise_assignment, privacy_leakage_table=None, smax=10, alpha=None,
                        max_power=None, energy_profile_path=None, energy_table=None):
        return self._send_msg_and_deserialize(
            "selectP3SLSplit",
            noise_assignment=noise_assignment,
            privacy_leakage_table=privacy_leakage_table,
            smax=smax,
            alpha=alpha,
            max_power=max_power,
            energy_profile_path=energy_profile_path,
            energy_table=energy_table,
        )

    def uploadModelParameter(self,parameter):
        self._send_msg_and_deserialize("uploadModelParameter",parameter=parameter)

    def setModelEndLayer(self,end):
        self._send_msg_and_deserialize("setModelEndLayer",end=end)

    def uploadOptimizerParameter(self,parameter):
        return self._send_msg_and_deserialize("uploadOptimizerParameter",parameter=parameter)

    def getAndRestPowerConumptionAndLogs(self):
        return self._send_msg_and_deserialize("getAndRestPowerConumptionAndLogs")

    def ResetPowerConsumptionAndLogs(self):
        self.logger.info("Resting PowerConsumptionLoads for %s..", self.name)
        self._send_msg_and_deserialize("ResetPowerConsumptionAndLogs")

    def savePowerConsumptionAndLogs(self,expname):
        self.logger.info("Saving PowerConsumptionLoads at client side for %s..", self.name)
        return self._send_msg_and_deserialize("savePowerConsumptionAndLogs",expname=expname)



    def getMaximumPowerConsumed(self):
        return self._send_msg_and_deserialize("getMaximumPowerConsumed")

    def splitPointProfileing(self,experimentlist,splitpoints):
        return self._send_msg_and_deserialize("splitPointProfileing",experimentlist=experimentlist,splitpoints=splitpoints)

    def getPowerConsumption(self,experiment,splitpoint):
        return self._send_msg_and_deserialize("getPowerConsumption",experiment=experiment,splitpoint=splitpoint)

    def getExperimentLOGS(self,experiment):
        return self._send_msg_and_deserialize("getExperimentLOGS",experiment=experiment)


    def addWorkerCsvLog(self,message):
        return self._send_msg_and_deserialize("addCsvLog",message=message)

    def suspend(self):
        if self.powerSaver:
            self.logger.info("Suspending the %s..", self.name)
            self._send_msg_and_deserialize("suspend")
            self.close()

    def wakeUp(self):
        if self.powerSaver:
            self.logger.info("Sending Wakeup Signal to %s ", self.name)
            send_magic_packet(self.macaddress)

    def getOptimimalSplitPoint(self,noicelevel, maxpoints, alpha=0.5):
        return self._send_msg_and_deserialize("getOptimimalSplitPoint",noicelevel=noicelevel,maxpoints=maxpoints,alpha=alpha)



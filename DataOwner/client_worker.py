"""Entry point for a P3SL client worker (data-owner side).

Run this script on every device that holds private training data. It launches
a WebSocket server worker that:

  * loads the device's local dataset partition,
  * holds the client-side portion of the split neural network, and
  * waits for the training coordinator (see ``DataScientest/training_coordinator.py``)
    to connect and drive the personalized sequential split-learning loop.

A companion subprocess (``CheckPowerConsumptionCode.py``) is spawned in the
background to keep the energy-consumption logger alive for the duration of
the run.

Typical invocation::

    python3 client_worker.py --name Client1 --host 0.0.0.0 \\
        --port 8777 --dataset cifar10 --iid 0
"""

import logging,argparse
import subprocess
from websocketServer import WebsocketServerWorkerCustom
from DataSet import getDataSet

FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(format=FORMAT)

subprocess.Popen(["python3", "CheckPowerConsumptionCode.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE,shell=False)

def start_websocket_server_worker(clientName,IID,alpha, host, port, verbose,datasetname="cifar", training=True,logger=logging.getLogger(__name__),dpinfo={}, privacy_alpha=0.5):
    """Helper function for spinning up a websocket server and setting up the local datasets."""
    datasize={"cifar10":32,"cifar100":32,"f_mnist":32,"flower102":256}[datasetname]
    server = WebsocketServerWorkerCustom(name=clientName,IID=IID,alpha=alpha,privacy_alpha=privacy_alpha,host=host, port=port, verbose=verbose,logger=logger,**dpinfo)
    trainset, testset = getDataSet(datasetname,clientName,IID=IID, datasize=(datasize, datasize), nclasses=None,alpha=alpha)
    ncalsses={"cifar10":10,"cifar100":100,"f_mnist":10,"flower102":102}[datasetname]
    server.add_dataset(trainset, key=datasetname+"_train",ncalsses=ncalsses,datasize=datasize)
    server.add_dataset(testset, key=datasetname+"_test",ncalsses=ncalsses,datasize=datasize)

    if training:
        logger.info("Train len(datasets["+datasetname+"_train]): %s", len(server.datasets[datasetname+"_train"]))
        logger.info("Test len(datasets["+datasetname+"_test]): %s", len(server.datasets[datasetname+"_test"]))

    logger.info("Server Start at %s:%s"%(host,port))
    server.start()
    return server

if __name__ == "__main__":
    logger = logging.getLogger("client_worker")
    parser = argparse.ArgumentParser(description="Run a P3SL data-owner worker for personalized sequential split learning.")
    parser.add_argument("--port","-p",default=7777,type=int,help="port number of the websocket server worker, e.g. --port 8777",)
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host for the connection")
    parser.add_argument("--dataset", type=str, default="cifar10",choices=["cifar10","cifar100","f_mnist","flower102"], help="DataSetTo Use")
    parser.add_argument("--name", type=str,default=None,choices=["Test"]+["Client"+str(i+1) for i in range(20)], help="name (id) of the websocket server worker, e.g. --id alice")
    parser.add_argument("--iid", type=int,default=0,choices=range(0,51), help="IID for Data 0 mean normal Data, 1 mean 2 Class Data, 2 means Random Unbalanced Data, 3 means normal full  ")
    parser.add_argument("--alpha", type=float,default=10, help="Dirichlet/data-partition alpha used by the dataset splitter")
    parser.add_argument("--privacy_alpha", type=float, default=0.5, help="P3SL privacy-sensitivity coefficient alpha_i in [0,1] used for local split-point selection")
    parser.add_argument("--verbose","-v",action="store_true",help="if set, websocket server worker will be started in verbose mode",)
    parser.add_argument("--useDp",action="store_true",help="legacy flag: add P3SL activation noise before sending intermediate representations")
    parser.add_argument("--dpmethod", type=str, default="Laplace",choices=["Laplace","Gaussian"], help="activation-noise distribution to use")
    parser.add_argument("--dploc", type=float, default=0, help="activation-noise location/mean")
    parser.add_argument("--dpscale", type=float, default=1, help="P3SL noise level sigma; Laplace scale is set so Var(noise)=sigma^2")
    args = parser.parse_args()
    dpinfo={"useDp":args.useDp,"dpmethod":args.dpmethod,"dploc":args.dploc,"dpscal":args.dpscale}

    if args.verbose:
        logger.setLevel(level=logging.DEBUG)
    else:
        logger.setLevel(level=logging.INFO)

    # Hook and start server
    server = start_websocket_server_worker(
        clientName=args.name,
        IID=args.iid,
        alpha=args.alpha,
        host=args.host,
        port=args.port,
        verbose=args.verbose,
        logger=logger,
        dpinfo=dpinfo,
        datasetname=args.dataset,
        privacy_alpha=args.privacy_alpha
    )
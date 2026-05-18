"""Entry point for the P3SL training coordinator (data-scientist side).

Run this script on the central machine that orchestrates P3SL personalized
sequential split learning. It connects over WebSockets to one or more client
workers (see ``DataOwner/client_worker.py``), holds the server-side portion
of the split neural network, and drives the training loop:

  * clients are trained sequentially at their personalized split points;
  * each client injects its own Laplacian activation noise before uploading
    intermediate representations;
  * the coordinator periodically aggregates W_1:smax on the server and does
    not redistribute the aggregated client-side weights in P3SL mode.

Optional legacy compression/decomposition and hardware energy logging remain
available for experiments.

Typical invocation::

    python3 training_coordinator.py --dataset cifar10 --modelName VGG16bn \\
        --hosts 192.168.1.10 192.168.1.11 --ports 8777 8777 \\
        --sl 4 4 --epoches 200 --batch_size 128
"""

import logging,argparse,sys,random
from Training import TrainingMain
from Util import HOST2IP,IP2HOST

FORMAT = "%(asctime)s %(message)s"
logging.basicConfig(format=FORMAT)


logger = logging.getLogger(__name__)

def define_and_get_arguments(args=sys.argv[1:]):
    parser = argparse.ArgumentParser(description="Training coordinator: orchestrates P3SL personalized sequential split learning across client workers.")
    parser.add_argument("--epoches", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--sub_epoches", type=int, default=1, help="Number subepoches in 1 epoch")
    parser.add_argument("--modelName", type=str, default="VGG16bn", help="Model To Use to use")
    parser.add_argument("--preTrain",  action="store_true", help="If want to use PreTrain Model")
    parser.add_argument("--dataset", type=str, default="cifar10",choices=["cifar10","cifar100","f_mnist","flower102"], help="dataset to use")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size of the training")
    parser.add_argument("--aggtestdatacount", type=int, default=-1, help="Test Data size for aggeval")
    parser.add_argument("--test_batch_size", type=int, default=128, help="batch size used for the test data")
    parser.add_argument("--id", type=str, default="alice", help="Host to connect")
    parser.add_argument("--sl", nargs="+", type=int,choices=[1,2,3,4,5,6,7,8,9,10], default=None, help="Split Point")
    parser.add_argument("--sllist", nargs="+", type=str, default=None, help="Split Point List")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--l2", type=float, default=5e-4, help="learning rate")
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["Adam", "SGD"], help="optimizer")
    parser.add_argument("--schedular", type=str, default=None,choices=["OneCycleLR", "CosineAnnealingLR"], help="optimizer")
    parser.add_argument("--hosts", type=str, nargs="+", default=None, help="Host to connect")
    parser.add_argument("--macaddress", type=str, nargs="+", default=None, help="Mac Addres to WakeUp")
    parser.add_argument("--ports", type=int, nargs="+", default=[8777], help="Host to connect")
    parser.add_argument("--noreconnect",  action="store_true", help="If you don't want to reconnect when disconnected")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    parser.add_argument("--cudadevice", type=int,nargs="+", default=[0], help="Device to use")
    parser.add_argument("--seed", type=int, default=142, help="seed used for randomization")
    parser.add_argument("--evalaute_model_per_epoch", type=int, default=1, help="Evaluate After number of rounds")
    parser.add_argument("--save_model", action="store_true", help="if set, model will be saved")
    parser.add_argument("--save_model_per_epoch", type=int, default=1, help="Save Model After Epoches")
    parser.add_argument("--aggrigate_evaluation",action="store_true",help="Do you want to do Aggrigated Evaluation")
    parser.add_argument("--aggrigate_eval_per_epoch", type=int, default=1, help="Do Aggrigated Evaluation After")
    parser.add_argument("--aggregationmethod", type=str, default="AggEqual",choices=["AggEqual","AggClientOnly"], help="Method For aggrigation")
    parser.add_argument("--uploadaggrigate", action="store_true", help="Legacy/baseline option to upload aggregated weights; ignored in P3SL mode because the paper keeps aggregated W_1:smax on the server")
    parser.add_argument("--uploadaggrigate_per_epoch", type=int, default=-1, help="Do Aggrigated Evaluation After")
    parser.add_argument("--AllAggrigation", type=int, default=None, help="Set 1 value for save_model_per_epoch,aggrigate_eval_per_epoch and uploadaggrigate_per_epoch")
    parser.add_argument("--useExpariment", type=str, default=None, help="give the name of experiment csv file to continue")
    parser.add_argument("--skipEpochs",  action="store_true", help="If want to Skip Sine Epoches")
    parser.add_argument("--Compression", action="store_true", help="if set, It will use the compression before data trasnfer")
    parser.add_argument("--useJit",  action="store_true", help="If want to use jit. use it if not using jetson")
    parser.add_argument("--savePowerLogs",  action="store_true", help="If want to use savePowerLogs for jetson it will save with experimentname on client side")
    parser.add_argument("--getPowerLogs",  action="store_true", help="If want to Download local powerlogs for jetson")
    parser.add_argument("--PowerLogger", type=str, default="KASA",choices=["KASA","JTOPSTAT"], help="Which Power Logger you wants to use")
    parser.add_argument("--powerSaver",  action="store_true", help="If want to enable power saver with suspend and wakeup for jetson nano")
    parser.add_argument("--usedecomposition",  action="store_true", help="If want to use decomposition for midoutput and gradient")
    parser.add_argument("--decompositionmethod",  type=str, default="Tucker", help="Method to use for Decomposition")
    parser.add_argument("--decompositionrank",  type=int, default=2, help="Rank to use for Decomposition")
    parser.add_argument("--preTrainEpoch", type=int,  default=100, help="Number of pretraineEpoches")
    parser.add_argument("--preTrain_batch_size", type=int,  default=64, help="Batch Size for the pretraine")
    parser.add_argument("--BaseRounds", type=int,  default=1, help="For How many time you want to run experiment")
    parser.add_argument("--dpNoise",  type=float,nargs="+",  default=None, help="Set P3SL activation-noise sigma for each client (legacy name)")
    parser.add_argument("--dpNoiseList",  type=str,nargs="+",  default=None, help="Set per-client activation-noise sigma values for each sllist entry")
    parser.add_argument("--randomNoise", action="store_true", help="Set random activation-noise sigma values for clients")
    parser.add_argument("--p3sl_auto_config", action="store_true", help="Run the P3SL PL/T_sigma broadcast and client-side split/noise selection before training")
    parser.add_argument("--p3sl_privacy_table", type=str, default="Fsimmean.csv", help="CSV privacy leakage table PL(s, sigma); default is DataScientest/Fsimmean.csv")
    parser.add_argument("--p3sl_fsim_threshold", type=float, default=0.36, help="FSIM threshold T_FSIM used to build T_sigma[s]")
    parser.add_argument("--p3sl_normalize_threshold", action="store_true", help="Interpret --p3sl_fsim_threshold on the normalized PL table instead of raw FSIM")
    parser.add_argument("--p3sl_smax", type=int, default=10, choices=range(1,11), help="Maximum allowable split point smax")
    parser.add_argument("--p3sl_privacy_alpha", type=float, nargs="+", default=None, help="Experimental override for per-client privacy-sensitivity coefficients alpha_i; omit this for paper-aligned privacy so each client uses its local --privacy_alpha")
    parser.add_argument("--p3sl_pmax", type=float, nargs="+", default=None, help="Optional per-client peak-power thresholds Pmax passed to local split selection")
    parser.add_argument("--p3sl_energy_profile", type=str, nargs="+", default=None, help="Optional per-client local energy-profile CSV paths on each client")
    parser.add_argument("--p3sl_reference_accuracy", type=float, default=None, help="Optional no-noise reference accuracy A_ref for A_min=(1-beta)A_ref")
    parser.add_argument("--p3sl_min_accuracy", type=float, default=None, help="Optional direct minimum accuracy threshold A_min for noise reassignment")
    parser.add_argument("--p3sl_accuracy_discount", type=float, default=0.05, help="Accuracy discount beta used when --p3sl_reference_accuracy is provided")
    parser.add_argument("--mode", type=str,  default="P3SL",choices=["P3SL","SSL","Sequential1","Sequential2","ParallellN"], help="Different modes for traininng and aggrigations"
                                                                                                    "Normal means Normal Aggrigated at every N epoch Also Call P3SL and NormalAgg"
                                                                                                    "Sequntial1 means Upload weights every time and download the weights at nth epcoch which will be used by next no Aggrigation Upload"
                                                                                                   "Sequntial2 means Upload and download the weights at nth epcoch which will be used by next worker  no Aggrigation Upload"
                                                                                                   "SSL means Upload and download the weights at Every epcoch which will be used by next worker no Aggrigation Upload. it's like same model used all the time"
                                                                                                    "ParallellN Train n models parallel and after every epoch do Aggriegation State Of Art My Code")
    parser.add_argument("--uploadperepoch", type=int,default=1, help="for how much epoch want to upload parampeters linked to Sequential1 mode")
    parser.add_argument("--verbose","-v",action="store_true",
        help="if set, websocket Data Scientist workers will be started in verbose mode",)
    args = parser.parse_args(args=args)

    args.nchannels={"cifar10":3,"cifar100":3,"f_mnist":1,"flower102":3}[args.dataset]
    print(args)
    if args.hosts is not None:
        args.hosts=[HOST2IP[c] if c in HOST2IP else c for c in args.hosts]
    if args.getPowerLogs: args.savePowerLogs=True
    # if args.getPowerLogs: args.powerSaver=True
    if args.powerSaver:
        args.savePowerLogs=args.getPowerLogs=True
        assert (args.hosts is None and args.macaddress is None) or args.macaddress is not None and len(args.hosts)==len(args.macaddress), "You should Provide mac and hosts both or let it run for default"
        if args.hosts is None and args.macaddress is None:
            args.hosts=["192.168.4.105","192.168.4.147","192.168.4.124","192.168.4.142",]
            args.macaddress = ["48:b0:2d:c1:68:63","48:b0:2d:c1:66:bb","48:b0:2d:c1:69:d6","48:b0:2d:ec:24:04"]
            # args.hosts = ["192.168.4.124", "192.168.4.142", ]
            # args.macaddress = ["48:b0:2d:c1:69:d6", "48:b0:2d:ec:24:04"]
    if args.hosts is None:
        args.hosts=["0.0.0.0"]
    assert args.p3sl_auto_config or args.sl is not None or args.sllist is not None," Please provide sl/sllist, or use --p3sl_auto_config"
    if len(args.ports) == 1 and len(args.hosts) > 1:
        args.ports=[args.ports[0] for i in args.hosts]
    if len(args.hosts)==1 and len(args.ports)>1:
        args.hosts=[args.hosts[0] for i in args.ports]
    if args.dpNoiseList is not None:
        if args.sllist is None:
            raise Exception("dpNoiseList only supported with sllist")
        if "_" not in args.dpNoiseList[0]:
            raise Exception("dpNoiseList elements must have _")
        for noise in args.dpNoiseList:
            assert len(noise.split("_"))==len(args.hosts),f" Noise Should same as host {args.hosts}"
            for n in noise.split("_"):
                float(n)  # validate numeric activation-noise sigma
    if args.sllist is not None and len(args.ports)>1:
        args.sllist=[sll if "_" in sll else "_".join([sll for i in args.ports]) for sll in args.sllist]
    if args.sl is not None and len(args.sl)==1 and len(args.ports)>1:
        args.sl=[args.sl[0] for i in args.ports]
    if args.AllAggrigation is not None:
        args.save_model_per_epoch=args.aggrigate_eval_per_epoch=args.uploadaggrigate_per_epoch=args.AllAggrigation
        args.save_model=args.aggrigate_evaluation=args.uploadaggrigate=True
    assert len(args.hosts)==len(args.ports),"Number of Host and port should be same "
    if args.randomNoise: args.dpNoise=[int(random.random()*250)/100 for i in range(len(args.hosts))]
    if args.dpNoise is not None:
        if len(args.dpNoise)==1:
            args.dpNoise=[args.dpNoise[0] for _ in args.hosts] # if only 1 given repeat that
        assert len(args.hosts) == len(args.dpNoise), "Dp Noice Should be same as clients"
    if not args.p3sl_auto_config:
        assert args.sl is not None and len(args.sl)==len(args.ports) or args.sllist is not None and len(args.sllist[0].split("_"))==len(args.ports),"Number of Host and Split Layer Should be same "
    if args.macaddress is None:
        args.macaddress=[None for _ in args.hosts]
    if args.skipEpochs:
        args.skipEpochsList={1:[[10,15],[25,35],[70,80]],2:[[10,15],[25,35],[70,80]],3:[[8,13],[80,90]],4:[[1,10]]}
        args.skipEpochsList={server:sum([list(range(*r)) for r in ranges],[]) for server,ranges in args.skipEpochsList.items()}
    else:
        args.skipEpochsList={}
    if not args.savePowerLogs:
        args.savePowerLogs=None
    for h in args.hosts:
        assert h.replace('.','_') in IP2HOST, f"Please Update Ip Address in Util.py {h} Is not in Hosts "
    logger.info("Running for Clients %s ",",".join([IP2HOST[h.replace('.', '_')] for h in args.hosts]))
    logger.info("Args %s",args)
    return args



def main():
    logger = logging.getLogger("training_coordinator")
    logger.setLevel(level=logging.INFO)

    args = define_and_get_arguments()
    TrainingMain(args,logger)

if __name__ == "__main__":
    main()


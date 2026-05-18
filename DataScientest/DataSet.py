"""Dataset loading and per-client partitioning (data-scientist side).

Coordinator-side mirror of :mod:`DataOwner.DataSet`. Used when the
coordinator itself needs to materialise a dataset partition (for example,
when running aggregated evaluation on a held-out test split).
"""

import random
import os,numpy as np

from torchvision import transforms, datasets
from fedlab.utils.dataset.partition import CIFAR10Partitioner
from Dirichlet_Partition import get_direct_partion_indexes
np.random.seed(41)

def FilterDataWithLables(trainset,lables):
    if hasattr(trainset,"targets"):
        indexies=[i for i,l in enumerate(trainset.targets) if l in lables ]
        trainset.data=trainset.data[indexies]
        trainset.targets=trainset.targets[indexies]
        idx = np.random.permutation(len( trainset.data))
        trainset.data, trainset.targets = trainset.data[idx], trainset.targets[idx]
    else:
        labeltoindex={l:i for i,l in enumerate(lables)}
        indexies = [i for i, l in enumerate(trainset._labels) if l in lables]
        trainset._image_files = np.array([trainset._image_files[i] for i in indexies])
        trainset._labels = np.array([labeltoindex[trainset._labels[i]] for i in indexies])
        idx = np.random.permutation(len( trainset._image_files))
        trainset._image_files, trainset._labels = np.array(trainset._image_files)[idx], np.array(trainset._labels)[idx]
    return trainset

def get_Train_Test_index_for_Direct(datasetname,alpha,testdatasize=1600,num_clients=7):
    # this is for iid=8
    dataTrainIndexies=get_direct_partion_indexes(datasetname,num_clients,alpha)
    dataTestIndexies={"Server":(0,testdatasize),"ServerFull":(0,testdatasize),"Test":(0,200)}
    return dataTrainIndexies,dataTestIndexies

def getdatapercentage(numberofclients=10,traindataperclient=800,testdatasize=1600):
    dataTrainIndexiesIID0={"Server":(0, traindataperclient * numberofclients),"ServerFull":(0,50000),"Test":(0,200)}
    for n in range(numberofclients):dataTrainIndexiesIID0[f"Client{n + 1}"]=(traindataperclient * n, traindataperclient * (n + 1))
    dataTestIndexies={"Server":(0,testdatasize),"ServerFull":(0,10000),"Test":(0,200)}
    for n in range(numberofclients):dataTestIndexies[f"Client{n+1}"]=(0,testdatasize)
    return dataTrainIndexiesIID0,dataTestIndexies


def getDatasetForServer(dataset, datasetname,IID=0, client=None, type="Train",testsize=-1, shuffle=True,alpha=10.0,databasepath="data/"):
    #IID 0 Equal Class for all Client from dataTrainIndexies (Equal Data Partition) with 800 Point Each
    #IID 1 2 Class per Clients from cifar10_NONIID_2Class.npz
    #IID 2 Random data from cifar10_NONIID_Random.npz (Unequal Data partition)
    #IID 3 Random data from Equal 5000 to each client
    #IID 4 Random data from Equal 3200 to each client
    #IID 5 4 Data Partions
    #IID 7 7 Data Partions
    #IID 20 20 Data Partions
    IIDDefinations={0:"Equal Class for all Client from dataTrainIndexies (Equal Data Partition) with  Equal 800 to each client",
                    1:"2 Class per Clients from cifar10_NONIID_2Class.npz",
                    2:"Random data from cifar10_NONIID_Random.npz (Unequal Data partition)",
                    3:"Equal Class for all Client from dataTrainIndexies (Equal Data Partition) with  Equal 5000 to each client",
                    4: "Equal Class for all Client from dataTrainIndexies (Equal Data Partition) with  Equal 3200 to each client",
                    5: "Equal Data for 4 Clients",
                    6: "1 Batch Train And test 128 samples each",
                    7: "Equal Data for 7 Clients",
                    8: f"Direchlet Partion for 7 Clients with {alpha} alpha",
                    12: "Direchlet Different Data for 10 Clients",
                    20: "Equal Different Data for 20 Clients",
                    50: "Each Client 100 Data points for debugging"}

    if client is None or client=="ServerFull": return dataset
    print("Using Dataset :-",IIDDefinations[IID])
    fullData4partions={"cifar10":12500,"f_mnist":15000,"cifar100":12500,"flower102":1537}
    fullData7partions={"cifar10":7142,"f_mnist":8571,"cifar100":7142,"flower102":878}
    fullData20partions={"cifar10":2500,"f_mnist":3000,"cifar100":2500,"flower102":307}
    fullDataTestSize={"cifar10":10000,"f_mnist":10000,"cifar100":10000,"flower102":1020}
    if not os.path.exists(f"{databasepath}{datasetname}.npz"):databasepath="../DataOwner/"+databasepath
    if IID==0:
        dataTrainIndexiesIID, dataTestIndexies=getdatapercentage(traindataperclient=800,testdatasize=1600)
    elif IID==3:
        dataTrainIndexiesIID, dataTestIndexies = getdatapercentage(traindataperclient=5000, testdatasize=10000)
    elif IID==4:
        dataTrainIndexiesIID, dataTestIndexies = getdatapercentage(traindataperclient=3200,testdatasize=1600)
    elif IID==5:
        dataTrainIndexiesIID, dataTestIndexies = getdatapercentage(traindataperclient=fullData4partions[datasetname],testdatasize=10000)
    elif IID==6:
        dataTrainIndexiesIID, dataTestIndexies = getdatapercentage(traindataperclient=128,testdatasize=128)
    elif IID==7:
        dataTrainIndexiesIID, dataTestIndexies = getdatapercentage(traindataperclient=fullData7partions[datasetname],testdatasize=fullDataTestSize[datasetname])
    elif IID == 8:
        dataTrainIndexiesIID, dataTestIndexies = get_Train_Test_index_for_Direct(datasetname,alpha,testdatasize=fullDataTestSize[datasetname],num_clients=7)
    elif IID==20:
        dataTrainIndexiesIID, dataTestIndexies = getdatapercentage(numberofclients=20,traindataperclient=fullData20partions[datasetname],testdatasize=fullDataTestSize[datasetname])
    Indexlist=(0,3,4,5,6,7,20)
    if type=="Train":
        if IID in Indexlist:
            start,end=dataTrainIndexiesIID[client]
            if shuffle:
                indexixes=np.load(f"{databasepath}{datasetname}.npz")["trainidx"][start:end]
            else:
                indexixes=np.arange(start,end)
            print("Using Shuffled Index for Train with ",len(indexixes),"Instances from ",start,"to",end)
        elif IID==1:
            print(f"{databasepath}{datasetname}_NONIID_2Class.npz")
            assert os.path.exists(f"{databasepath}{datasetname}_NONIID_2Class.npz"), f"{databasepath}{datasetname}_NONIID_2Class not Exists"
            indexixes=np.load(f"{databasepath}{datasetname}_NONIID_2Class.npz")[client]
            print("Using 2 Class Index For ",client)
        elif IID==2:
            assert os.path.exists(f"{databasepath}{datasetname}_NONIID_Random.npz"), f"{datasetname}_NONIID_Random not Exists"
            indexixes=np.load(f"{databasepath}{datasetname}_NONIID_Random.npz")[client]
            print("Using Random Data Index For ",client)
        elif IID == 8:
            clientindex = int(client.lstrip("Clinet"))
            indexixes = np.array(dataTrainIndexiesIID[clientindex-1])
            print(f"Using Train Data from Direct Partition {len(indexixes)} with seed 42")
        elif IID == 10:
            indexixes = np.arange(0, 5000)
        elif IID == 11:
            clientindex = int(client.lstrip("Clinet"))
            indexixes = np.arange(5000 * (clientindex - 1), 5000 * clientindex)
            print(f"Using Train Data from {5000 * (clientindex - 1)} to {5000 * clientindex}")
        elif IID == 12:
            clientindex = int(client.lstrip("Clinet"))
            hetero_dir_part = CIFAR10Partitioner(dataset.targets, num_clients=10, balance=None,
                                                 partition="dirichlet", dir_alpha=alpha, seed=42, verbose=False)
            indexixes = hetero_dir_part[clientindex - 1]
            print(f"Using Train Data from CIFAR10Partitioner {len(indexixes)} with seed 42")
        elif IID == 50:
            clientindex = int(client.lstrip("Clinet"))
            indexixes = np.arange(100 * (clientindex - 1), 100 * clientindex)
        # elif IID==3200:
    else:
        if testsize != -1:
            start, end = 0, min(testsize,len(dataset))
        elif IID in (8,10,11,12):
            start, end=0,len(dataset)
        elif IID ==50:
            start, end = 0, 100
        elif IID  in Indexlist:
            start,end = dataTestIndexies[client]
        else:
            start, end = 0, len(dataset)
        if shuffle:
            indexixes=np.load(f"{databasepath}{datasetname}.npz")["testidx"][start:end]
        else:
            indexixes=np.arange(start,end)
        print("Using Shuffled Index for Test with ",len(indexixes),"Instances from ",start,"to",end)
    if hasattr(dataset,"data"):
        random.shuffle(indexixes)
        dataset.data, dataset.targets = np.array(dataset.data)[indexixes], np.array(dataset.targets)[indexixes]
    elif hasattr(dataset,"_image_files"):
        random.shuffle(indexixes)

        dataset._image_files, dataset._labels = np.array(dataset._image_files)[indexixes], np.array(dataset._labels)[indexixes]
    else:
        raise Exception("Attribute not supported")
    return dataset

datapath="data/"

datasetslist=["mnist","f_mnist","cifar10","cifar100","food101","flower","flower102"]
datasizes={"mnist":(28,28),"f_mnist":(28,28),"cifar10":(32,32),"cifar100":(32,32),"food101":(256,256),"flower":(256,256),"flower102":(256,256)}


def getDataSet(datasetname, clientName=None,IID=0, datasize=None,nclasses=None,testsize=-1, shuffle=True,alpha=10):
    print(f"Loading Dataset {datasetname} for Client {clientName} with IID {IID} ...........")
    # load datasets and initialize DataScientest, server, and clone models
    if datasize is None: datasize=datasizes[datasetname]
    assert datasetname in datasetslist, " Only " + str(datasetslist) + " supported"
    # Data transforms (normalization & data augmentation)

    stats = {"cifar10": ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), "f_mnist": ((0.5,), (0.5)),
             "cifar100": ((0.4914, 0.4822, 0.4465), (0.2471, 0.2436, 0.2617)),
             "flower102":((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))}[datasetname]
            # "cifar10": ((0.4914, 0.4822, 0.4465), (0.2471, 0.2436, 0.2617))
             # "flower102":((0.42469782, 0.363852 ,  0.27730465),(0.30292657, 0.24979997, 0.26888368))}[datasetname]
             # "flower102": ((0.44066918, 0.37353307, 0.27921927), (0.30989096, 0.25385082, 0.27395234))
    mtrainTransform = transforms.Compose([transforms.Resize(datasize),
                            transforms.RandomCrop(datasize[0], padding=4, padding_mode='reflect'),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomRotation(degrees=(0, 10)),
                             # tt.RandomPerspective(distortion_scale=0.14),
                             # tt.RandomResizedCrop(256, scale=(0.5,0.9), ratio=(1, 1)),
                             # tt.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                             transforms.ToTensor(),
                             transforms.Normalize(*stats, inplace=True)])
    mytesttransforms = transforms.Compose([transforms.Resize(datasize),transforms.ToTensor(), transforms.Normalize(*stats)])
    if datasetname == 'mnist':
        trainset = datasets.MNIST(datapath+'mnist', download=True, train=True, transform=mtrainTransform)
        testset = datasets.MNIST(datapath+'mnist', download=True, train=False, transform=mytesttransforms)
        # dataattrib,labelattrib="data","targets"
    elif datasetname == 'f_mnist':
        trainset = datasets.FashionMNIST(datapath+'f_mnist', download=True, train=True, transform=mtrainTransform)
        testset = datasets.FashionMNIST(datapath+'f_mnist', download=True, train=False, transform=mytesttransforms)
        # dataattrib,labelattrib="data","targets"
    elif datasetname == 'cifar10':
        trainset = datasets.CIFAR10(datapath+'cifar10', download=True, train=True, transform=mtrainTransform)
        testset = datasets.CIFAR10(datapath+'cifar10', download=True, train=False, transform=mytesttransforms)
        # dataattrib,labelattrib="data","targets"
    elif datasetname == 'cifar100':
        trainset = datasets.CIFAR100(datapath + 'cifar100', download=True, train=True, transform=mtrainTransform)
        testset = datasets.CIFAR100(datapath + 'cifar100', download=True, train=False, transform=mytesttransforms)
        # dataattrib,labelattrib="data","targets"
    elif datasetname == 'food101':
        trainset = datasets.Food101(datapath + 'food101', download=True, split="train", transform=mtrainTransform)
        testset = datasets.Food101(datapath + 'food101', download=True, split="test", transform=mytesttransforms)
        # dataattrib,labelattrib="_image_files","_labels"
    elif datasetname == 'flower102':
        trainset = datasets.Flowers102(datapath+'flower102', download=True, split="test", transform=mtrainTransform)
        testset  = datasets.Flowers102(datapath+'flower102', download=True, split="train", transform=mytesttransforms)
        # dataattrib,labelattrib="_image_files","_labels"
    elif datasetname == 'flower':
        trainset = datasets.Flowers102(datapath+'flower102', download=True, split="test", transform=mtrainTransform)
        testset  = datasets.Flowers102(datapath+'flower102', download=True, split="train", transform=mytesttransforms)
        # dataattrib,labelattrib="_image_files","_labels"
    if nclasses:
        classes=list(map(lambda x: x[0], list(sorted(list(zip(*np.unique(trainset._labels, return_counts=True))), key=lambda x: x[1], reverse=True))))
        trainset, testset = FilterDataWithLables(trainset, classes[:nclasses]), FilterDataWithLables(testset, classes[:nclasses])
    return getDatasetForServer(trainset, datasetname,IID, clientName,shuffle=shuffle,alpha=alpha),getDatasetForServer(testset, datasetname,IID, clientName, "Test",testsize=testsize,shuffle=False,alpha=alpha)




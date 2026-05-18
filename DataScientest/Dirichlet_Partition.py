"""Dirichlet-based non-IID dataset partitioner (data-scientist side).

Mirror of :mod:`DataOwner.Dirichlet_Partition`. Used by the coordinator when
materialising aggregated evaluation splits or when tagging experiments with
the same partition seed as the client workers.
"""

import numpy as np
from collections import defaultdict, Counter
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from torchvision import  datasets
import numpy as np

import torchvision
from datasets import load_dataset
import hashlib
import numpy as np

datapath="data/"
def get_hf_to_tv_Indexes_for_cifar10(datapath=datapath):
    # Torchvision CIFAR10
    tv_cifar = torchvision.datasets.CIFAR10(root=datapath+"cifar10", train=True, download=True)
    # HF CIFAR10
    hf_cifar = load_dataset("cifar10", split="train")

    # Build lookup: hash -> index
    def hash_img(img):
        return hashlib.md5(np.array(img).tobytes()).hexdigest()

    tv_map = {hash_img(tv_cifar[i][0]): i for i in range(len(tv_cifar))}
    hf_to_tv = [tv_map[hash_img(x["img"])] for x in hf_cifar]
    return hf_to_tv
hf_to_tv=get_hf_to_tv_Indexes_for_cifar10()

def dirichlet_partition(labels, num_clients, alpha, seed=42):
    np.random.seed(seed)
    labels = np.array(labels)
    num_classes = len(np.unique(labels))
    idx_by_class = {c: np.where(labels == c)[0] for c in range(num_classes)}

    client_indices = defaultdict(list)

    for c, idxs in idx_by_class.items():
        np.random.shuffle(idxs)
        # Sample proportions for each client from Dirichlet
        proportions = np.random.dirichlet(alpha=[alpha]*num_clients)
        # Scale proportions by class size
        proportions = (proportions * len(idxs)).astype(int)

        # Adjust to match class size
        while proportions.sum() < len(idxs):
            proportions[np.argmax(proportions)] += 1
        while proportions.sum() > len(idxs):
            proportions[np.argmax(proportions)] -= 1

        start = 0
        for client_id, count in enumerate(proportions):
            client_indices[client_id].extend(idxs[start:start+count])
            start += count

    return client_indices

def get_direct_partion(dataset_name,num_clients,alpha):
    dir_partitioner = DirichletPartitioner(
        num_partitions=num_clients,
        partition_by="label",
        alpha=alpha,
        min_partition_size=10,       # you can adjust
        self_balancing=True,
        shuffle=True,
        seed=42
    )
    # Load federated dataset
    fds = FederatedDataset(
        dataset=dataset_name,
        partitioners={"train": dir_partitioner}
    )
    # partition=fds.load_partition(client_index, "train")
    # print("SSSS",len(partition._indices),len(partition._indices[0]),type(partition._indices),client_index)
    # print(partition["label"][:20])
    # return list(partition._indices[0])
    return {ci:fds.load_partition(ci, "train")._indices[0] for ci in range(num_clients)}


def get_direct_partion_indexes(dataset,num_clients,alpha):
    if dataset=="cifar10":
        indexes=get_direct_partion("cifar10",  num_clients, alpha)
        return {ind:[hf_to_tv[i] for i in index] for ind,index in indexes.items()}
    elif dataset=="f_mnist":
        indexes=get_direct_partion("zalando-datasets/fashion_mnist",num_clients,alpha)
        return indexes
    elif dataset=="flower102":
        dataset = datasets.Flowers102(datapath + 'flower102', download=True, split="test")
        return dirichlet_partition(dataset._labels, num_clients, alpha)
    else:
        print(f"{dataset} not supported for ")


def countClasses(dataset,datasetname,num_clients, alpha,num_classes=10):
    print("datasetname",datasetname)
    LabelDictList=[]
    for ci in range(num_clients):
        indexies=get_direct_partion_indexes(datasetname, num_clients, alpha)
        # print(indexies)
        # print(indexies[ci])
        # print(indexies.keys())
        labels=[dataset[ind][1] for ind in indexies[ci]]
        # print(labels[:20])
        print(ci,":",len(list(set(labels))),Counter(labels))
        LabelDictList.append(Counter(labels))
    print("Full Training data",{cl: sum([s[cl] for s in LabelDictList]) for cl in range(num_classes)})
    print("Full Training data",sum([sum([s[cl] for s in LabelDictList]) for cl in range(num_classes)]))


def checkIndex(alpha=0.1,num_clients=7):
    # dataset = datasets.Flowers102(datapath + 'flower102', download=True, split="test")
    # countClasses(dataset, "flower102", num_clients, alpha,num_classes=102)
    # trainset = datasets.FashionMNIST(datapath + 'f_mnist', download=True, train=True)
    # countClasses(trainset, "f_mnist", num_clients, alpha,num_classes=10)
    trainset = datasets.CIFAR10(datapath + 'cifar10', download=True, train=True)
    countClasses(trainset, "cifar10", num_clients, alpha,num_classes=10)

if __name__ == '__main__':
    checkIndex(alpha=0.1)

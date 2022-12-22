from config import config
import random
from data_hyper import data
from collections import defaultdict
import copy
import os, torch, numpy as np
import time
from model import networks
import torch, os, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from model import utils
import itertools
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import logging
import statistics
import pickle


def initialize_settings(seed,gpu):
    torch.manual_seed(seed)
    np.random.seed(seed)
    logging.propagate = False
    logging.getLogger().setLevel(logging.ERROR)
    # gpu, seed
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ['PYTHONHASHSEED'] = str(seed)

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(Z, Y):
    """
    arguments:
    Z: predictions
    Y: ground truth labels

    returns:
    accuracy
    """

    predictions = Z.max(1)[1].type_as(Y)
    correct = predictions.eq(Y).double()
    correct = correct.sum()

    accuracy = correct / len(Y)
    return accuracy


def test_regression_model(model,E,X,Y,input_weight,idx_test):
    model.eval()
    output = model(E,X,input_weight)
    loss_test = F.binary_cross_entropy(output[idx_test], Y[idx_test])
    return loss_test.item()


def train_regression_model(model,optimiser,E,X,Y,input_weight,idx_train,epoch):
    t = time.time()
    model.train()
    optimiser.zero_grad()
    output = model(E,X,input_weight)
    loss_train = F.binary_cross_entropy(output[idx_train], Y[idx_train])
    loss_train.backward()
    optimiser.step()
    print('Epoch: {:04d}'.format(epoch+1),'loss_train: {:.4f}'.format(loss_train.item()),
    'time: {:.4f}s'.format(time.time() - t))

def test_classification_model(model,E,X,Y,input_weight,idx_test):
    model.eval()
    output = model(E,X,input_weight)
    loss_test = F.nll_loss(output[idx_test], Y[idx_test])
    acc_test = accuracy(output[idx_test], Y[idx_test])
    print(acc_test.item(), end="\t", flush=True)
    print('\n')
    return loss_test.item(),acc_test.item()


def train_classification_model(model,optimiser,E,F,X,Y,input_weight,idx_train,epoch):
    t = time.time()
    model.train()
    optimiser.zero_grad()
    output = model(E,X,input_weight)
    loss_train = F.nll_loss(output[idx_train], Y[idx_train])
    acc_train = accuracy(output[idx_train], Y[idx_train])
    loss_train.backward()
    optimiser.step()
    if(epoch%10==0):
        print('Epoch: {:04d}'.format(epoch+1),'loss_train: {:.4f}'.format(loss_train.item()),'acc_train: {:.4f}'.format(acc_train.item()),
       'time: {:.4f}s'.format(time.time() - t))

def get_multimedia_data(dataset):
    E  = {}

    index = 0
    edges_nodes = np.transpose(dataset['YAll'])
    for row in edges_nodes:
        row = row.tolist()
        non_zero_relations = np.nonzero(row)[0].tolist()
        E[index] = non_zero_relations
        index = index + 1

    # E =  dataset['hypergraph']
    X, Y = dataset['XAll'], dataset['LAll']

    for k, v in E.items():
       E[k] = list(v)
    #    if k not in E[k]:
    #       E[k].append(k)
    return E, X, Y

def process_multimedia_dataset(dataset,cuda):
    E, X, Y = get_multimedia_data(dataset)
    return prepare_graph_data(E,X,Y,True)

def train_hypermsg_multimedia_model(E,X,Y,input_weight,idx_train,idx_test,d,c,depth,epochs,dropout,rate,decay,cuda):
    hypermsg = networks.HyperMSG_multimedia(E, X, d, c, dropout, cuda)

    optimiser = optim.Adam(list(hypermsg.parameters()), lr=args.rate, weight_decay=args.decay)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.95)

    scheduler = optim.lr_scheduler.StepLR(optimiser, 200, gamma=0.2, last_epoch=-1)

    if cuda:
        hypermsg.cuda()

    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train_regression_model(hypermsg,optimiser,E,X,Y,input_weight,idx_train,epoch)
        if(epoch%20==0):
            print(epoch)
            print("------------ TEST ------------------")
            loss_test = test_regression_model(hypermsg,E,X,Y,input_weight,idx_test)
            print(loss_test)
    fin_loss = test_regression_model(hypermsg,E,X,Y,input_weight,idx_test)
    print("\n")
    return hypermsg, fin_loss


def get_data(dataset):
    E =  dataset['hypergraph']
    X, Y = dataset['features'], dataset['labels']
   
    for k, v in E.items():
        E[k] = list(v)
        if k not in E[k]:
            E[k].append(k)
    return E, X, Y


def process_dataset(dataset,cuda):
    E, X, Y = get_data(dataset)
    return prepare_graph_data(E,X,Y)

def prepare_graph_data(E,X,Y, multimedia = False):
    global_neighborhood = defaultdict(list)
    edge_count = np.zeros(X.shape[0])
    global_neighborhood_count = np.zeros(X.shape[0])
    unique_nodes = []
    for edge, nodes in E.items():
        for node in nodes:
            unique_nodes.append(node)
            neighbor_nodes = ([i for i in nodes if i != node])
            edge_count[node] = edge_count[node] + 1
            global_neighborhood[node].extend(neighbor_nodes)

    for k,v in global_neighborhood.items():
        global_neighborhood_count[k] = len(set(v))
    unique_nodes = list(set(unique_nodes))
    input_weight = np.concatenate((np.expand_dims(global_neighborhood_count, axis = 1), np.expand_dims(edge_count, axis = 1)), axis = 1)
    
    X = normalize(X)
    d, c = X.shape[1], Y.shape[1]
    X = torch.FloatTensor(np.array(X))
    Y = np.array(Y)
    if multimedia:
        Y = torch.FloatTensor(np.array(Y))
    else:
        Y = torch.LongTensor(np.where(Y)[1])
    input_weight = torch.FloatTensor(input_weight)
    if cuda:
        X, Y = X.cuda(), Y.cuda()
        input_weight = input_weight.cuda()
        for key, value in E.items():
            E[key] = torch.Tensor(list(E[key])).cuda()
    return E, X, Y, d, c, input_weight


def train_hypermsg_model(E,X,Y,input_weight,idx_train,idx_test,d,c,depth,epochs,dropout,rate,decay,cuda):
    hypermsg = networks.HyperMSG(E, X, d, depth, c, dropout, cuda)
    optimiser = optim.Adam(list(hypermsg.parameters()), lr=rate, weight_decay=decay)
    scheduler = optim.lr_scheduler.StepLR(optimiser, 200, gamma=0.2, last_epoch=-1)
 
    # cuda
    if cuda:
        hypermsg.cuda()

    # Train model
    t_total = time.time()
    for epoch in range(epochs):
        train_classification_model(hypermsg,optimiser,E,F,X,Y,input_weight,idx_train,epoch)
        if(epoch%50==0):
            print("EPOCH: ", epoch)
            loss_test,acc_test = test_classification_model(hypermsg,E,X,Y,input_weight,idx_test)
    return hypermsg, acc_test

def get_split(split,input,dset, cuda):
    dataset, train, test = data.load(split,input,dset)
    idx_train = torch.LongTensor(train)
    idx_test = torch.LongTensor(test)
    if cuda:
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()
    return dataset, idx_train, idx_test

            
def get_multimedia_split(split,input,dset, cuda, doshuffle):
    dataset, train, test = data.load_multimedia(input,dset,doshuffle)
    idx_train = torch.LongTensor(train)
    idx_test = torch.LongTensor(test)
    if cuda:
        idx_train = idx_train.cuda()
        idx_test = idx_test.cuda()
    return dataset, idx_train, idx_test
    
def eval_multimedia_splits(input,dset,splits,depth,epochs,dropout,rate,decay,cuda,doshuffle):
    test_result_each_split = []
    for num in range(splits):
        print("SPLIT: ", num )
        print('\n')
	
        dataset, idx_train, idx_test = get_multimedia_split(num+1,input,dset,cuda,doshuffle)
        E, X, Y, d, c, input_weight = process_multimedia_dataset(dataset, cuda)
        hypermsg,acc_test = train_hypermsg_multimedia_model(E,X,Y,input_weight,idx_train,idx_test,d,c,depth,epochs,dropout,rate,decay,cuda)

        fin_loss = test_regression_model(hypermsg,E,X,Y,input_weight,idx_test)
        test_result_each_split.append(fin_loss)
    return test_result_each_split


def eval_splits(input,dset,splits,depth,epochs,dropout,rate,decay,cuda):
    test_result_each_split = []
    for num in range(splits):
        print("SPLIT: ", num )
        print('\n')

        dataset, idx_train, idx_test = get_split(num+1,input,dset,cuda)
        E, X, Y, d, c, input_weight = process_dataset(dataset, cuda)

        hypermsg,acc_test = train_hypermsg_model(E,X,Y,input_weight,idx_train,idx_test,d,c,depth,epochs,dropout,rate,decay,cuda)

        fin_loss,fin_acc = test_classification_model(hypermsg,E,X,Y,input_weight,idx_test)
        test_result_each_split.append(fin_acc)
    return test_result_each_split


def store_result(res, filename):
    file = open(filename,"w")
    file.write(str(args))
    file.write('\n')
    file.write("RESULTS FOR SPLITS:\n")
    for r in res:
        file.write(str(r)+'\n')
    file.write("\nSTATISTICS:\n")
    file.write("mean: " + str(statistics.mean(res)))
    file.write(" sdev: " + str(statistics.stdev(res)))
    file.close()



args = config.parse()
initialize_settings(args.seed,args.gpu)
nsplits = args.split
cuda = args.cuda and torch.cuda.is_available()
doshuffle = args.shuffle
dataset = args.dataset
res = []
if dataset == "MIRFLICKR":
    res = eval_multimedia_splits(args.data,args.dataset,nsplits,args.depth,args.epochs,args.dropout,args.rate,args.decay,cuda,doshuffle)
if dataset in ["cora","citeseer","pubmed"]:
    res = eval_splits(args.data,args.dataset,nsplits,args.depth,args.epochs,args.dropout,args.rate,args.decay,cuda)
if res != []:
    print("mean: ", statistics.mean(res))
    print("sdev: ", statistics.stdev(res))
    store_result(res,"hyperout.txt")

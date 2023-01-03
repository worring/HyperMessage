'''
data: coauthorship/cocitation
dataset: cora/dblp/acm for coauthorship and cora/citeseer/pubmed for cocitation
MIRFLICKR for multimedia
'''
inputdir = "./data_hyper"
data = "cocitation"
dataset = "cora"
split = 1
addself = False


'''
gpu: gpu number to use
cuda: use cuda
seed: an integer
'''
gpu = 0
seed = 42



'''
model related parameters
depth: number of hidden layers
dropout: dropout probability for  hidden layer
epochs: number of training epochs
'''
depth = 2
dropout = 0.5
epochs =150



'''
parameters for optimisation
rate: learning rate
decay: weight decay
'''
rate = 0.01
decay = 0.0005

power = 1
shuffle = False
out = "hyperout"

import configargparse, os,sys,inspect
from configargparse import YAMLConfigFileParser



def parse():
	"""
	adds and parses arguments / hyperparameters
	"""
	default = os.path.join(current(), data + ".yml")
	p = configargparse.ArgParser(config_file_parser_class = YAMLConfigFileParser, default_config_files=[default])
	p.add('-c', '--my-config', is_config_file=True, help='config file path')
	p.add('--data', type=str, default=data, help='data name (coauthorship/cocitation)')
	p.add('--inputdir', type=str, default=inputdir, help='basedir where data is stored')
	p.add('--dataset', type=str, default=dataset, help='dataset name (e.g.: cora/dblp/acm for coauthorship, cora/citeseer/pubmed for cocitation)')
	p.add('--addself', dest='addself', action='store_true', help='add node to neighbor list')
	p.add('--split', type=int, default=split, help='train-test split used for the dataset')
	p.add('--generatesplit', dest='generatesplit', action='store_true', help='create a split rather than using a stored one')
	p.add('--depth', type=int, default=depth, help='number of hidden layers')
	p.add('--dropout', type=float, default=dropout, help='dropout for hidden layer')
	p.add('--rate', type=float, default=rate, help='learning rate')
	p.add('--decay', type=float, default=decay, help='weight decay')
	p.add('--epochs', type=int, default=epochs, help='number of epochs to train')
	p.add('--gpu', type=int, default=gpu, help='gpu number to use')
	p.add('--power', type=float, default=power, help='power mean')
	p.add('--cuda', dest='cuda', action='store_true', help='cuda for gpu')
	p.add('--seed', type=int, default=seed, help='seed for randomness')
	p.add('--shuffle', dest='shuffle', action='store_true',help='shuffle the data for splits')
	p.add('--out', type=str, default=out, help='file to store statistics of experiment')
	p.add('-f') # for jupyter default
	return p.parse_args()



def current():
	"""
	returns the current directory path
	"""
	current = os.path.abspath(inspect.getfile(inspect.currentframe()))
	head, tail = os.path.split(current)
	return head

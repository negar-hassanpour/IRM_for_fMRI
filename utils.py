import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd

import scipy.io

data_file = 'DS007-stopmanual_stopvocal-WB-std4mm-DYL-TempAlign.mat'

class MLP(nn.Module):
	def __init__(self, num_feats, hidden_dim, num_class):
		super(MLP, self).__init__()
		lin1 = nn.Linear(num_feats, hidden_dim)
		lin2 = nn.Linear(hidden_dim, hidden_dim)
		lin3 = nn.Linear(hidden_dim, num_class)
		for lin in [lin1, lin2, lin3]:
			nn.init.xavier_uniform_(lin.weight)
			nn.init.zeros_(lin.bias)
		self._main = nn.Sequential(lin1, nn.ReLU(True), lin2, nn.ReLU(True), lin3)
	
	def forward(self, inpt):
		out = self._main(inpt)
		return out

def mean_accuracy(logits, y):
	accuracy = (logits.argmax(dim=1) == y).sum().float() / float( y.size(0) )
	return accuracy

def pretty_print(*values):
	col_width = 13
	def format_val(v):
		if not isinstance(v, str):
			v = np.array2string(v, precision=5, floatmode='fixed')
		return v.ljust(col_width)
	str_values = [format_val(v) for v in values]
	print("	 ".join(str_values))

def load_data():
	mat = scipy.io.loadmat('data/'+data_file)

	data = {}
	for i, item in enumerate(np.squeeze(mat['subject'])):
		if item not in data:
			data[item] = {}
		if 'data' not in data[item]:
			data[item]['data'] = []
			data[item]['label'] = []
		data[item]['data'].append(mat['data'][i, :])
		data[item]['label'].append(mat['label'][0][i]-1)

	for i, item in enumerate(data.keys()):
		data[item]['data'] = torch.tensor(data[item]['data']).cuda()
		data[item]['label'] = torch.tensor(data[item]['label'], dtype=torch.long).cuda()
	num_feats = data[item]['data'].size()[1]
	num_class = len(np.unique(mat['label']))

	return data, num_feats, num_class

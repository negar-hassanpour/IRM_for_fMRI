import sys
import argparse
import numpy as np
import torch
from torchvision import datasets
from torch import nn, optim, autograd

from utils import *

parser = argparse.ArgumentParser(description='ERM for fMRI')
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--l2_regularizer_weight', type=float,default=0.001)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_restarts', type=int, default=10)
parser.add_argument('--steps', type=int, default=501)
flags = parser.parse_args()

print('Flags:')
for k,v in sorted(vars(flags).items()):
	print("\t{}: {}".format(k, v))

data, num_feats, num_class = load_data()

final_train_accs = []
final_test_accs = []
for restart in range(flags.n_restarts):
	print("Restart", restart)

	# Choose train/test splits and make envs
	np.random.seed(1234567890+restart)
	perm_subjects = np.random.permutation(list(data.keys()))
	
	envs = [{'data': data[perm_subjects[0]]['data'], 
			'label': data[perm_subjects[0]]['label']}]
	for subject in perm_subjects[1:-1]:
		envs[0]['data'] = torch.vstack((envs[0]['data'], data[subject]['data']))
		envs[0]['label'] = torch.hstack((envs[0]['label'], data[subject]['label']))
	envs.append(data[perm_subjects[-1]])  # test env is the last one

	mlp = MLP(num_feats, flags.hidden_dim, num_class).cuda()
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(mlp.parameters(), lr=flags.lr)

	pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

	for step in range(flags.steps):
		train_nll = 0
		train_acc = 0
		train_penalty = 0
		for index, env in enumerate(envs):
			logits = mlp(env['data'])
			env['nll'] = criterion(logits, env['label'])
			env['acc'] = mean_accuracy(logits, env['label'])
			if index < len(envs)-1:	# keep the last env for test
				train_nll += env['nll']
				train_acc += env['acc']
		train_nll /= (len(envs)-1)
		train_acc /= (len(envs)-1)

		weight_norm = torch.tensor(0.).cuda()
		for w in mlp.parameters():
			weight_norm += w.norm().pow(2)

		loss = train_nll.clone()
		loss += flags.l2_regularizer_weight * weight_norm

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		
		test_acc = envs[-1]['acc']
		if step % 100 == 0:
			pretty_print(
				np.int32(step),
				train_nll.detach().cpu().numpy(),
				train_acc.detach().cpu().numpy(),
				test_acc.detach().cpu().numpy()
			)

	final_train_accs.append(train_acc.detach().cpu().numpy())
	final_test_accs.append(test_acc.detach().cpu().numpy())
	print('Final train acc (mean/std across restarts so far):')
	print(np.mean(final_train_accs), np.std(final_train_accs))
	print('Final test acc (mean/std across restarts so far):')
	print(np.mean(final_test_accs), np.std(final_test_accs))
	sys.stdout.flush()

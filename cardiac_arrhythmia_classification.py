#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:38:17 2020

@author: aligazanayi
"""


# from google.colab import drive
# drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# #preprocessing Installations
# !pip install wfdb
# !pip install pywavelets
# !pip install syft

from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(cm,
                          target_names= ['N', 'L', 'R', 'A', 'V'],
                          title='Confusion Matrix',
                          cmap=None,
                          normalize=False):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(4, 3), dpi=200)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title + ' Confusion matrix')
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}     misclass={:0.4f}'.format(accuracy, misclass))
    fig1=plt.gcf()
    plt.show()
    fig1.savefig(title+'_cm', bbox_inches='tight')

import numpy as np
import sys, os
import wfdb
import pywt
from collections import Counter
import pandas as pd
import seaborn as sns
from yellowbrick.features import Rank1D
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from yellowbrick.target import ClassBalance

#seeding
from numpy.random import seed
seed(12)
# from tensorflow import set_random_seed
# set_random_seed(1234)

data_names = ['100', '101', '102', '103', '104', '105', '106', '107', 
              '108', '109', '111', '112', '113', '114', '115', '116', 
              '117', '118', '119', '121', '122', '123', '124', '200', 
              '201', '202', '203', '205', '207', '208', '209', '210', 
              '212', '213', '214', '215', '217', '219', '220', '221', 
              '222', '223', '228', '230', '231', '232', '233', '234']

wid = 100
labels =  ["N", 'L', 'R', 'A', 'V']
X_total = []
Y_total = []
for d in data_names:
  r=wfdb.rdrecord('mitdb/1.0.0/'+d)
  ann=wfdb.rdann('mitdb/1.0.0/'+d, 'atr', return_label_elements=['label_store', 'symbol'])
  sig = np.array(r.p_signal[:,0])
  sig_len = len(sig)
  sym = ann.symbol
  pos = ann.sample
  beat_len = len(sym)
  for i in range(1,beat_len-1):
    if sym[i] in labels: 
      if (pos[i]-pos[i-1])>200 and (pos[i+1]-pos[i])>200:
        a = sig[pos[i]-150:pos[i]+150]
        a, cD3, cD2, cD1 = pywt.wavedec(a, 'db6', level=3)
        X_total.append(a)
        Y_total.append(labels.index(sym[i]))

X_total = np.array(X_total)
Y_total = np.array(Y_total)
X = X_total
Y = Y_total
columns = ['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26','C27','C28','C29','C30','C31','C32','C33','C34','C35', 'C36','C37','C38','C39','C40','C41','C42','C43','C44','C45','C46','C47']

# Hide grid lines
plt.rcParams["axes.grid"] = False
def Data_visualize(Y, title):
  # Instantiate the visualizer
  visualizer = ClassBalance(labels= labels,  size=(1080, 720), dpi=200)
  visualizer.fit(Y)     
  fig1 = plt.gcf()
  visualizer.poof()              # Finalize and render the figure
  fig1.savefig(title+' ClassBalance.png', bbox_inches='tight')

Data_visualize(Y_total, 'Prior')

figure(num=None, figsize=(4, 3), dpi=200, facecolor='w', edgecolor='k')
plt.plot(X[0])
plt.gcf().savefig('pulse.png', bbox_inches='tight')
plt.show()

#Random Under Sampler
rus = RandomUnderSampler(random_state=100)
X, Y = rus.fit_resample(X, Y)

Data_visualize(Y,'Undersampled')

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)
X_train_df, X_valid_df, Y_train_df, Y_valid_df = pd.DataFrame(X_train.squeeze(), index= range(len(X_train)), columns = columns), pd.DataFrame(X_valid, index= range(len(X_valid)), columns = columns),pd.DataFrame(Y_train, index= range(len(Y_train)), columns = ['C48']), pd.DataFrame(Y_valid, index= range(len(Y_valid)), columns = ['C48'])

from yellowbrick.features import Rank2D
visualizer = Rank1D(algorithm='shapiro',  size=(720, 480), orient='v', dpi=300)
visualizer.fit_transform(X,Y)     # Transform the data
fig1 = plt.gcf()
visualizer.poof()              # Finalize and render the figure
fig1.savefig('1D_Rank.png', bbox_inches='tight',dpi=200)

visualizer = Rank2D(algorithm="pearson",  size=(1080, 720), dpi=300)
visualizer.fit_transform(X_train)
fig1 = plt.gcf()
visualizer.poof()
fig1.savefig("pearson47.png", bbox_inches='tight',dpi=200)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(0)

class Arguments():
    def __init__(self):
        self.batch_size = 95
        self.test_batch_size = 95
        self.epochs = 200
        self.lr = 0.001 # learning rate
        self.log_interval = 95

args = Arguments()

import syft as sy  
hook = sy.TorchHook(torch) 
client = sy.VirtualWorker(hook, id="client") 
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
crypto_provider = sy.VirtualWorker(hook, id="crypto_provider")

import torch
import numpy as np
from torch.utils import data
from sklearn.model_selection import train_test_split


X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=285*5, stratify=Y, shuffle=True)
np.unique(y_train, return_counts=True)
np.unique(y_val, return_counts=True)
print(y_val.shape)
print(Counter(y_val))
print(Counter(y_train))

train_dataset = data.TensorDataset(torch.Tensor(X_train).float(), torch.Tensor(y_train).long())
train_loader = data.DataLoader(train_dataset,batch_size=args.batch_size, shuffle=True)
print(torch.Tensor(X_train).float())

test_dataset = data.TensorDataset(torch.Tensor(X_val).float(), torch.Tensor(y_val).long())
test_loader = data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True)

boby = sy.VirtualWorker(hook, id="boby")
anne = sy.VirtualWorker(hook, id="anne")

train_labels = torch.Tensor(y_train).long()
train_inputs = torch.Tensor(X_train).float()
test_labels = torch.Tensor(y_val).long()
test_inputs = torch.Tensor(X_val).float()

# threshold indexes for dataset split (one half for Bob, other half for Anne)
train_idx = int(len(train_labels)/2)
test_idx = int(len(test_labels)/2)


# Sending toy datasets to virtual workers
bob_train_dataset = sy.BaseDataset(train_inputs[:12*95], train_labels[:12*95]).send(boby)
anne_train_dataset = sy.BaseDataset(train_inputs[12*95:], train_labels[12*95:]).send(anne)
bob_test_dataset = sy.BaseDataset(test_inputs[:8*95], test_labels[:8*95]).send(boby)
anne_test_dataset = sy.BaseDataset(test_inputs[8*95:], test_labels[8*95:]).send(anne)

# Creating federated datasets, an extension of Pytorch TensorDataset class
federated_train_dataset = sy.FederatedDataset([bob_train_dataset, anne_train_dataset])
print(federated_train_dataset)
federated_test_dataset = sy.FederatedDataset([bob_test_dataset, anne_test_dataset])

# Creating federated dataloaders, an extension of Pytorch DataLoader class
federated_train_loader = sy.FederatedDataLoader(federated_train_dataset, 
                                                shuffle=True, batch_size=args.batch_size)
federated_test_loader = sy.FederatedDataLoader(federated_test_dataset, 
                                               shuffle=False, batch_size=args.batch_size)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(47, 235)
        self.fc11= nn.Dropout(p=0.1, inplace=False)
        self.fc2 = nn.Linear(235, 141)
        self.fc22= nn.Dropout(p=0.1, inplace=False)
        self.fc3 = nn.Linear(141, 5)
    def forward(self, x):
        x = x.view(-1, 47)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc11(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc22(x)
        x = self.fc3(x)
        return x
def train(args, model, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (inputs, labels) in enumerate(federated_train_loader):
        optimizer.zero_grad()
        worker = inputs.location
        model.send(worker)
        output = model(inputs)
        output = F.log_softmax(output, dim=1)
        loss = F.nll_loss(output, labels)
        loss.backward()
        model.get()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('\nTrain Epoch: {} \t loss: {}'.format(
                epoch, loss.get()))

model= Net()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)    #optim.SGD(model.parameters(), lr=args.lr)
for epoch in range(1, 100):
    train(args, model, federated_train_loader, optimizer, epoch)

model.eval()
n_correct_priv = 0
n_total = 0

with torch.no_grad():
  for data, target in federated_test_loader:
      worker = data.location
      model.send(worker)
      output = model(data)
      pred = output.squeeze().argmax(dim=1).get()
      del output
      target_view = target.get().view_as(pred)
      n_correct_priv += pred.eq(target_view).sum()
      del pred
      n_total += args.test_batch_size
      n_correct = n_correct_priv.copy()

      print('Test set: Accuracy: {}/{} ({:.3f}%)'.format(n_correct, n_total, 100. * n_correct / n_total))
      
      model.get()
#from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt


#for epoch in range (epoch):
#    out = model(federated_train_loader,).to(worker)
#    _, pred = out.max(1)
#    n_total += target_view.size(0)
#    n_correct += (pred == target_view).sum().item()
#    print(federated_train_loader)
#    print(pred)
#    loss = F.nll_loss(out,target_view)
#    counter +=1
#    print('loss train', "Epoch N", counter,loss.data[0])
#    model.zero_grad()
#    opt.step()
#print('Accuracy of the network on train dataset: {} %'.format(100 * n_correct / n_total))

#conf_matrix = metrics.confusion_matrix(100. * n_correct / n_total, test_labels)

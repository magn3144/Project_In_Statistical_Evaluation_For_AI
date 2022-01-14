from rpy2.robjects import r, pandas2ri
import math
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn import model_selection
import torch.optim as optim
from matplotlib import pyplot as plt
import torch.nn.init as init
import os
from math import ceil
import random
import datetime

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed_everything(2)

curr_path  = os.getcwd()
path = curr_path + "/armdata.RData"
print(path)

robj = r.load('armdata.RData')

#%% get data

# eksperiment person repitition time dimension
pos_data = np.array(r['armdata'], dtype="float32")


# there's 12*3 missing datapoints
# it's all the first 1, 2, or 4 datapoints i a series in 6 of the trails
# nan_data = np.isnan(pos_data)
# nan_indexes = np.where(nan_data)

# pd.DataFrame(nan_indexes).to_csv('nan_indexes.csv')

# the nan values are replaced by copying the first non nan value in the
# dataseries
pos_data[4, 8, 0, 0, 0:3] = pos_data[4, 8, 0, 1, 0:3]
pos_data[6, 8, 1, 0, 0:3] = pos_data[6, 8, 1, 1, 0:3]
pos_data[9, 8, 0, 0:2, 0:3] = pos_data[9, 8, 0, 2, 0:3]
pos_data[10, 8, 0, 0:2, 0:3] = pos_data[10, 8, 0, 2, 0:3]
pos_data[12, 8, 0, 0:4, 0:3] = pos_data[12, 8, 0, 4, 0:3]
pos_data[13, 8, 1, 0:2, 0:3] = pos_data[13, 8, 1, 2, 0:3]

## get velecity and acceleration data
#vel_data = np.diff(pos_data, n=1, axis=3)
#acc_data = np.diff(pos_data, n=2, axis=3)

# calculate mean trajectories
#mean_pos = np.mean(pos_data, axis=(1,2))
#mean_vel = np.mean(vel_data, axis=(1,2))
#mean_acc = np.mean(acc_data, axis=(1,2))

# MES of test statistics
#pos_test_stat = np.mean(np.array([(e1 - e2)**2 for e1, e2 in zip(pos_data, mean_pos)])**2, axis=3)
#vel_test_stat = np.mean(np.array([(e1 - e2)**2 for e1, e2 in zip(vel_data, mean_vel)])**2, axis=3)
#acc_test_stat = np.mean(np.array([(e1 - e2)**2 for e1, e2 in zip(acc_data, mean_acc)])**2, axis=3)

# We quickly validate that the data is still good. Here we take the  3rd experiment, 7th person, 5th repetition,
# 5th data point


pos_data = pos_data.reshape((16,100,300))
pos_data = pos_data.reshape((100*16,300))

new_col = [[n] for n in range(16) for i in range(100)]

pos_data = np.append(pos_data, new_col, axis=1)
pos_data = pos_data.astype('float32')

# Just flatten the last dimension and use an incrementor that increments every 160 iterations to add the experiment value to the end of the 300 list.
def preprocess(data):
    """
    min-max normalize data according to each x, y and z value
    """
    x_max = np.amax(data[:,0:100])
    y_max = np.amax(data[:,100:200])
    z_max = np.amax(data[:,200:300])

    x_min = np.amin(data[:,0:100])
    y_min = np.amin(data[:,100:200])
    z_min = np.amin(data[:,200:300])

    data[:,0:100] = (data[:,0:100] - x_min)/(x_max - x_min)
    data[:,100:200] = (data[:,100:200] - y_min)/(y_max - y_min)
    data[:,200:300] = (data[:,200:300] - z_min)/(z_max - z_min)

    return [[x_max, y_max, z_max], [x_min, y_min, z_min]]

preprocess(pos_data)

def k_fold_split(data, folds=2, val=False, shuffle=False):

    kf = model_selection.KFold(n_splits=folds, shuffle = shuffle)
    kf.get_n_splits(data)

    return kf


class NN(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, batch_size, folds):

        super(NN, self).__init__()

        self.batch_size = batch_size
        self.folds = folds

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.Softmax(dim=1)
        )
        # ONLY USE if you use double inputs for neural network
        # Actually don't use, since CrossEntropy isn't defined for double type
        #self.layers.double()

        self.optimizer = torch.optim.Adam(NN.parameters(self))
        self.criterion = torch.nn.CrossEntropyLoss()

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(self.device)
        self.layers = self.layers.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)



    def reset_weights(self, layers, fancy_init=False):
        """
        Layers should be of type nn.sequential
        """
        for layer in layers:
            if isinstance(layer, nn.Linear):
                if fancy_init:
                    init.kaiming_normal_(layer.weight, mode='fan_in',
                                         nonlinearity='relu')
                    init.kaiming_normal_(layer.bias, mode='fan_in',
                                         nonlinearity='relu')
                else:
                    layer.reset_parameters()



    def train_epoch(self, train_dat: torch.tensor ,running_loss=False, v=False) -> float:

        self.train()
        trainloader = torch.utils.data.DataLoader(train_dat,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)
        loss = 0
        for i, batch in enumerate(trainloader):
            self.optimizer.zero_grad()

            # The below doesn't work, because OBVIOUSLY it detaches the gradient
            #out = torch.argmax(self.forward(batch[:,0:-1]), dim=1)
            out = (self.forward(batch[:,:-1]))

            target = batch[:, -1]

            # Ok, so this is where it gets funky
            # Cross entropy needs its out value to be floats, but its target values to be Long, (doubles in py)
            # This is because python ints are seen as C++ Long.
            #out = out.type(torch.FloatTensor)
            target = target.type(torch.LongTensor).to(self.device)

            #print(out)
            #print(target)

            batch_loss = self.criterion(out, target)
            batch_loss.backward()
            self.optimizer.step()

            r = i + 1
            loss += batch_loss

        # If running loss, return the sum of loss across the entire epoch
        if running_loss:
            if v:
                print('Running loss is', loss.item())

            return loss.item()

        if v:
            print("Loss is ", loss.item()/r)

        # Otherwise, return the mean loss for the entire epoch
        return loss.item()/r



    def one_layer_train(self, data: np.ndarray, epochs, kf, v=False):
        gen_accs = []

        for i, (train_idxs, test_idxs) in enumerate(kf):

            _train = data[train_idxs]
            _test = data[test_idxs]

            for epoch in range(epochs):
                print("Current Loss: ", self.train_epoch(torch.from_numpy(_train).to(self.device)))

            gen_accs.append(self.test_network(torch.from_numpy(_test).to(self.device), acco=True))
            print("Current test acc: ", gen_accs[i])
            self.reset_weights(self.layers, fancy_init=False)

        gen_errors = [1-i for i in gen_accs]
        gen_error = sum(gen_errors)/len(gen_errors)

        print("Final generalization error: " , gen_error)

        return gen_errors



    def two_layer_train(self, data: np.ndarray, epochs,params, kf, v=False,pf=False):
        """
        Yes, this function right here, officer
        It sucks ass
        """
        gen_accs = []
        gen_errors = []

        print(pf)
        print("Now entering training loop")
        print(kf)

        [print("Herp") for i in kf]
        print(kf)

        for p, (train_idxs,test_idxs) in enumerate(kf):
            print("I here")
            print(len(train_idxs))

            print("R here")
            print(len(test_idxs))
            _train = data[train_idxs]
            _test = data[test_idxs]

            secondary_kf = k_fold_split(_train, folds=len(params))
            secondary_it = secondary_kf.split(_train)

            print("We have done stuff")
        print("Creating k_f")
        k_f = k_fold_split(data, folds=2)
        it = k_f.split(data)

        for t,(derp, herp) in enumerate(it):
            print("HEr de derp")

        for i, (train_idxs, test_idxs) in enumerate(kf):
            if pf or not pf:
                print(i)
            _train = data[train_idxs]
            _test = data[test_idxs]

            secondary_kf = k_fold_split(_train, folds=len(params))
            secondary_it = secondary_kf.split(_train)

            best_fold_acc = 0
            best_param = 0

            for r, (par_idxs, val_idxs) in enumerate(secondary_it):
                if pf:
                    print(r)

                # CHANGE PARAMS HERE
                ########################################
                # CHANGE PARAMS HERE

                #if v:
                #    PRINT PARAM INFO HERE
                #print("Now testing PARAM {param}")

                _par = _train[par_idxs]
                _val = _train[val_idxs]

                model_par_loss = []

                for epoch in range(epochs):
                    model_par_loss.append(self.train_epoch(torch.from_numpy(_par).to(self.device), v=False))

                val_acc = self.test_network(torch.from_numpy(_val).to(self.device), acco=True)
                # Check if current model lr is the best out of the current fold of training and test data
                if val_acc > best_fold_acc:
                    best_fold_acc = val_acc
                    # Assign new best parameter here
                    #best_lr = learning_rates[r]

                # Reset model weights in-between different parameters
                self.reset_weights(self.layers, fancy_init=False)

            for epoch in range(epochs):
                self.train_epoch(torch.from_numpy(_train).to(self.device),v=False)

            gen_accs.append(self.test_network(torch.from_numpy(_test).to(self.device), acco=True))
            print("El test is here: ", self.test_network(torch.from_numpy(_test).to(self.device), acco=True))

        gen_errors = [1-i for i in gen_accs]
        print("Gen error", gen_errors,"Gen Accuracy", gen_accs)
        gen_error = sum(gen_errors)/len(gen_errors)
        print("Gen error has been completed once, ",gen_error)
        return gen_error



    def train_multiple_models(self, data, models,epochs, params, v=False):
        model_gen_errors = []

        for m, model in enumerate(models):
            # Define k_folder every team, because otherwise it doesn't work appareantly
            k_f = k_fold_split(data, folds=2)
            k_f_it = k_f.split(data)

            print(f"Now testing model {m+1}")
            self.layers = model.to(self.device)

            if m == 1:
                model_gen_errors.append(self.two_layer_train(data, epochs=epochs, params=params, kf=k_f_it,pf=True))

            model_gen_errors.append(self.two_layer_train(data, epochs=epochs, params=params, kf=k_f_it))

        for i, error in enumerate(model_gen_errors):
            print(f"Model {i}: {error}")


    def test_network(self, test_dat, acco=True, v=False):
        """
        For testing model prediction, returns model test loss and list of actual model differences
        """

        self.eval()
        predicted = torch.argmax(self.forward(test_dat[:,:-1]), dim=1)
        predicted = predicted.cpu().detach().numpy()

        true = test_dat[:, -1].cpu().detach().numpy()
        true = true.astype('int64')

        corrects = true[true == predicted]
        acc = len(corrects)/len(true)

        #print("Corrects here, ", corrects)
        #print("Len of corrects, ", len(corrects))
        #print("Len of true, ", len(true))
        #print("TRUE HERE, ",true)
        #print("PREDICTED HERE ",predicted)
        if v:
            print("Currect accuracy is, ", acc)

        if acco:
            return acc

        return 1-acc



    def un_norm(self, data, un_norms):
        data[:,0:100] = data[:,0:100] * (un_norms[0][0] - un_norms[1][0]) + un_norms[1][0]
        data[:,100:200] = data[:,100:200] * (un_norms[0][1] - un_norms[1][1]) + un_norms[1][1]
        data[:,200:300] = data[:,200:300] * (un_norms[0][2] - un_norms[1][2]) + un_norms[1][2]

        return data

begin_time = datetime.datetime.now()

in_dim = 300
out_dim = 16

models = [
    nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
            nn.Softmax(dim=1)
        ),
    nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, out_dim),
            nn.Softmax(dim=1)
        ),
        nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(128, out_dim),
            nn.Softmax(dim=1)
        ),
        nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Dropout(0.25),
                nn.Linear(128, out_dim),
                nn.Softmax(dim=1)
            ),
        nn.Sequential(
                nn.Linear(in_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.BatchNorm1d(64),
                nn.Linear(64,32),
                nn.ReLU(),
                nn.Linear(32, out_dim),
                nn.Softmax(dim=1)
            ),
        nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LazyBatchNorm1d(),
            nn.Dropout(0.25),
            nn.Linear(32,out_dim),
            nn.Softmax(dim=1)
        )
]

np.random.shuffle(pos_data)


un_normalize = lambda val: val * (maxs[-1] - mins[-1]) + mins[-1]

network = NN(in_dim=300,out_dim=16,batch_size=8,folds=2)

k_f = k_fold_split(pos_data, folds=2)
k_f_it = k_f.split(pos_data)

params = ["derp", "derp"]

print(pos_data[0].dtype)
#network.train_epoch(torch.from_numpy(pos_data))
network.train_multiple_models(data=pos_data, epochs=100,params=params,models=models)

#test_gen, diff_gen = net.two_layer_k_fold(data,learning_rates, epochs=10, kf=k_f_it, v=True)

print(datetime.datetime.now() - begin_time)

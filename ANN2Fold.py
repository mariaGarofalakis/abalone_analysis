import pandas as pd
import numpy as np
import seaborn as sns
from scipy.linalg import svd
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
import matplotlib.pyplot as plt

# exercise 8.1.1

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
from scipy.io import loadmat
import torch
import sklearn.linear_model as lm
from sklearn import model_selection
from toolbox_02450 import rlr_validate, train_neural_net
import scipy.stats as st




# reading csv files into a pandas dataframe
data =  pd.read_csv('C:/Users/maria/Desktop/mathimata/machine_learning/02450Toolbox_Python/Data/abalone.data', sep=",", header=None)
#seting the names of the columns of our dataframe
data.columns =['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']


data_200 = data.copy()

#our dataset was normalized to 200 so we multiply by 200 in order to get the original mesurments which makes more sense
data_200[['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']] = data_200[['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']]*200

#we make a dataframe with only ratio data in order to normalize them-> substruct mean and devide by std
data_only_ratios = data_200.drop(['Sex', 'Rings'], axis=1)
normalized_df=(data_only_ratios-data_only_ratios.mean())/data_only_ratios.std()


data_one_hot=pd.get_dummies(data_200)

final = pd.concat([normalized_df, data_one_hot[['Rings','Sex_F', 'Sex_I', 'Sex_M']]], axis=1)



zero_outlier_index = np.where(data_200['Height'].to_numpy()==0)[0]
final=final.drop(zero_outlier_index,axis=0)


X =final.drop(['Rings'], axis=1).to_numpy()
y = final[['Rings']].to_numpy()
attributeNames = final.drop(['Rings'], axis=1).columns.to_list()
N, M = X.shape

# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)



hidden_units = np.arange(5)


########    Generalization error for ANN
Error_train_ANN = np.empty((K,1))
Error_test_ANN = np.empty((K,1))

opt_units_array = np.zeros(K)




k=0
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index].squeeze()
    X_test = X[test_index]
    y_test = y[test_index].squeeze()
    
    

    X_train[:, :8] = (X_train[:, :8] - mu[k, :] ) / sigma[k, :] 
    X_test[:, :8] = (X_test[:, :8] - mu[k, :] ) / sigma[k, :] 
    
    
    
    #################################################################################################################################
    ######################################### iner loop for  NN  ####################################################################
    
    
    CV2 = model_selection.KFold(10, shuffle=True)

    
 

    test_error = np.empty((K,hidden_units.shape[0]))
    
    for (i, (train_index_ANN, test_index_ANN)) in enumerate(CV2.split(X_train,y_train)): 
        
        
        # Extract training and test set for current CV fold, convert to tensors
        X_train_ANN = torch.Tensor(X_train[train_index_ANN,1:])
        y_train_ANN = torch.Tensor(y_train[train_index_ANN])
        X_test_ANN = torch.Tensor(X_train[test_index_ANN,1:])
        y_test_ANN = torch.Tensor(y_train[test_index_ANN])
        
        for n_hidden_units in hidden_units:
            
            loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear((M), n_hidden_units), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
        
        # Train the net on training data
            net, train_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train_ANN,
                                                               y=y_train_ANN,
                                                               n_replicates=1,
                                                               max_iter=10000)
            
            
            
            print('\n\tBest loss: {}\n'.format(train_loss))
            
            # Determine estimated class labels for test set
            y_test_est = net(X_test_ANN)
            
            
            se = (y_test_est.squeeze().float()-y_test_ANN.float())**2 # squared error
            mse = (sum(se).type(torch.float)/len(y_test_ANN)).data.numpy()
            
            test_error[i,n_hidden_units] =  mse
            
    opt_val_err = np.min(np.mean(test_error,axis=0))
    opt_hiden_unit = hidden_units[np.argmin(np.mean(test_error,axis=0))]
    
    model = lambda: torch.nn.Sequential(
                                torch.nn.Linear((M), opt_hiden_unit), #M features to n_hidden_units
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(opt_hiden_unit, 1), # n_hidden_units to 1 output neuron
                                # no final tranfer function, i.e. "linear output"
                                )
        
        # Train the net on training data
    net, train_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train_ANN,
                                                               y=y_train_ANN,
                                                               n_replicates=1,
                                                               max_iter=10000)
            
            
            
    print('\n\tBest loss: {}\n'.format(train_loss))
            
            # Determine estimated class labels for test set
    y_test_est = net(torch.from_numpy(X_test[:,1:]).float())
    
    
    se = (y_test_est.float()-torch.from_numpy(y_test).unsqueeze(-1).float())**2 # squared error
            
    Error_test_ANN[k] = (sum(se).type(torch.float)/len(torch.from_numpy(y_test))).data.numpy()

    opt_units_array[k] = opt_hiden_unit
    


    
 
##################################  save error results##################################################################################

all_errors = np.array([Error_test_ANN.squeeze(),opt_units_array])

error_Table = pd.DataFrame(np.transpose(all_errors.squeeze()),columns=["error_ANN","units"])

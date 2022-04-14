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
data =  pd.read_csv('C:/Users/nikos/Documents/ML/02450Toolbox_Python/Data/abalone.data', sep=",", header=None)
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
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 2
CV = model_selection.KFold(K, shuffle=True)


# Values of lambda
lambdas = np.power(10.,range(0,2))
hidden_units = np.arange(2)


########    Generalization error with linear Regrassion
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))

########    Generalization error with Ridge Regression
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))

########    Generalization error with Baseline model
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))


########    Generalization error for ANN
Error_train_ANN = np.empty((K,1))
Error_test_ANN = np.empty((K,1))


opt_lamda_array = np.zeros(K)
opt_units_array = np.zeros(K)


w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K)) 

######### 3 tables for the statistical comparison #######

lossANN = np.empty((N,1))
lossLinear = np.empty((N,1))
lossBase = np.empty((N,1))

min_error_ANN_tmp = 1000000000





k=0
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index].squeeze()
    X_test = X[test_index]
    y_test = y[test_index].squeeze()
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]
    
    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
    Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]
    y_est_no_reg = X_test @ w_noreg[:,k]
    
    
    #################################################################################################################################
    ######################################### iner loop for  NN  ####################################################################
    
    
    CV2 = model_selection.KFold(2, shuffle=True)

    
 

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
                                torch.nn.Linear((M-1), n_hidden_units), #M features to n_hidden_units
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
                                torch.nn.Linear((M-1), opt_hiden_unit), #M features to n_hidden_units
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
    
    opt_lamda_array[k] = opt_lambda
    opt_units_array[k] = opt_hiden_unit
    
    y_est_base = y_train.mean()
    y_base = y_est_base * np.ones([len(y_test),1])
    
    if Error_test_ANN[k] < min_error_ANN_tmp:
        min_error_ANN_tmp = Error_test_ANN[k]
        lossANN = (y_test - y_test_est.detach().numpy().squeeze()) ** 2
        lossLinear = (y_test - y_est_no_reg ) ** 2
        lossBase = (y_test - y_base.squeeze()) ** 2
    
    
    
    
########################################################################################################################################


    

    ####################################plot results
    # Display the results for the last cross-validation fold
    if k == K-1:
        figure(k, figsize=(12,8))
        subplot(1,2,1)
        semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    
    # To inspect the used indices, use these print statements
    #print('Cross validation fold {0}/{1}:'.format(k+1,K))
    #print('Train indices: {0}'.format(train_index))
    #print('Test indices: {0}\n'.format(test_index))

    k+=1

show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

# w_rlr[:,k] 

min_ind = np.argmin(Error_test_rlr)

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,min_ind],2)))

##################################  save error results##################################################################################

all_errors = np.array([Error_test.squeeze(),Error_test_rlr.squeeze(),Error_test_nofeatures.squeeze(),Error_test_ANN.squeeze(),opt_lamda_array,opt_units_array])

error_Table = pd.DataFrame(np.transpose(all_errors.squeeze()),columns=["linear_Regression","Ridge_Regression","Baseline_model","error_ANN","lamda","units"])


z1 = lossANN - lossLinear
z2 = lossANN - lossBase
z3 = lossLinear - lossBase



alpha = 0.05


#####################################   statistical testing t-statistic ##############################

conf_int1 = st.t.interval(1-alpha, len(z1)-1, loc=np.mean(z1), scale=st.sem(z1))  # Confidence interval
p1 = st.t.cdf( -np.abs( np.mean(z1) )/st.sem(z1), df=len(z1)-1)  # p-value
##############################################################
conf_int1 = st.t.interval(1-alpha, len(z2)-1, loc=np.mean(z2), scale=st.sem(z2))  # Confidence interval
p2 = st.t.cdf( -np.abs( np.mean(z2) )/st.sem(z2), df=len(z2)-1)  # p-value
#####################################################################
conf_int1 = st.t.interval(1-alpha, len(z3)-1, loc=np.mean(z3), scale=st.sem(z3))  # Confidence interval
p3 = st.t.cdf( -np.abs( np.mean(z3) )/st.sem(z3), df=len(z3)-1)  # p-value

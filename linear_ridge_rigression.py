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
M = M+1

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)


# Values of lambda
lambdas = np.power(10.,range(-9,9))



########    Generalization error with linear Regrassion
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))

########    Generalization error with Ridge Regression
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))

########    Generalization error with Baseline model
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))


# initialize vector for optimal values for lamda
opt_lamda_array = np.zeros(K)



w_rlr = np.empty((M,K))
w_noreg = np.empty((M,K)) 



k=0
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index].squeeze()
    X_test = X[test_index]
    y_test = y[test_index].squeeze()
    internal_cross_validation = 10    
    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    
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
    
    
    
    opt_lamda_array[k] = opt_lambda

    
    
    
    
    
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
        
        
        subplot(1,2,2)
        title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        loglog(lambdas,train_err_vs_lambda.T,'b.-',lambdas,test_err_vs_lambda.T,'r.-')
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
    

    k+=1

show()



all_errors = np.array([Error_test.squeeze(),Error_test_rlr.squeeze(),Error_test_nofeatures.squeeze(),opt_lamda_array])

error_Table = pd.DataFrame(np.transpose(all_errors.squeeze()),columns=["linear_Regression","Ridge_Regression","Baseline_model","lamda"])



# Display results
print('Linear regression without feature selection:')
print('- Training error Linear: {0}'.format(Error_train.mean()))
print('- Test error Linear:     {0}'.format(Error_test.mean()))

print('Regularized linear regression:')
print('- Training error Ridge: {0}'.format(Error_train_rlr.mean()))
print('- Test error Ridge:     {0}'.format(Error_test_rlr.mean()))



min_ind = np.argmin(Error_test_rlr)

print('Weights of the fold with the smallest generalized error:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,min_ind],2)))


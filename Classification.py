import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg import svd
from matplotlib.pyplot import figure, plot, title, xlabel, ylabel, show, legend
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import sklearn.linear_model as lm



# reading csv files
data =  pd.read_csv('../Data/abalone.data', sep=",", header=None)
data.columns =['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']







data_200 = data.copy()
data_200[['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']] = data_200[['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']]*200

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
y = np.zeros(X.shape[0])
attributeNames = final.drop(['Rings'], axis=1).columns.to_list()

idx_young = np.where((final['Rings'].to_numpy())<6)[0]
idx_middle = np.where(((final['Rings'].to_numpy())>6)&((final['Rings'].to_numpy())<15))[0]
idx_old = np.where((final['Rings'].to_numpy())>15)[0]


y[idx_young] = 0
y[idx_middle] = 1
y[idx_old] = 2


N, M = X.shape

classNames = ['young','middle','old']


## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10
CV = model_selection.KFold(K, shuffle=True)


# Values of lambda
lambdas = np.power(10.,range(0,2))
L = 2
mu = np.empty((K, M-3))
sigma = np.empty((K, M-3))

########    Generalization error with Logistic Regrassion
Error_logistic_test = np.empty((K,1))

########    Generalization error with KNN
Error_KNN_test = np.empty((K,1))

########    Generalization error with Baseline model
Error_test_baseline = np.empty((K,1))


opt_lamda_array = np.zeros(K)
opt_K_array = np.zeros(K)

k=0
for train_index, test_index in CV.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index].squeeze()
    X_test = X[test_index]
    y_test = y[test_index].squeeze()
    
    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, :7], 0)
    sigma[k, :] = np.std(X_train[:, :7], 0)
    
    X_train[:, :7] = (X_train[:, :7] - mu[k, :] ) / sigma[k, :] 
    X_test[:, :7] = (X_test[:, :7] - mu[k, :] ) / sigma[k, :] 
    
   
    
    
    #################################################################################################################################
    ######################################### iner loop   ####################################################################
    
    
    CV2 = model_selection.KFold(10, shuffle=True)

    
 

    test_error_k = np.empty((K,L))
    test_error_lamda = np.empty((K,lambdas.shape[0]))
    
    for (i, (train_index_inner, test_index_inner)) in enumerate(CV2.split(X_train,y_train)): 
        
        
        # Extract training and test set for current CV fold, convert to tensors
        X_train_inner = X_train[train_index_inner,:]
        y_train_inner = y_train[train_index_inner]
        X_test_inner = X_train[test_index_inner,:]
        y_test_inner = y_train[test_index_inner]
        
        
        ###### KNN classifier #####
    
        for l in range(1,L+1):
            knclassifier = KNeighborsClassifier(n_neighbors=l);
            knclassifier.fit(X_train_inner, y_train_inner);
            y_est = knclassifier.predict(X_test_inner);
            test_error_k[i,l-1] = np.sum(y_est!=y_test_inner)
            
        
        ####### Logistic Regression ####
        
        for k1 in range(0, lambdas.shape[0]):
            multiReg = lm.LogisticRegression(penalty='l2', C=1/lambdas[k1], max_iter=10000,multi_class='multinomial')
            
            multiReg.fit(X_train_inner, y_train_inner)

            y_est = multiReg.predict(X_test_inner).T
            
            test_error_lamda[i,k1] = np.sum(y_est!=y_test_inner)

        i+=1
    
    
    
    ######  Find Optimal K neighbors #####
    opt_val_err_KNN = np.min(np.mean(test_error_k,axis=0))
    opt_k = np.argmin(np.mean(test_error_k,axis=0))+1
    
    ######  Find Optimal Lamdas #####
    opt_val_err_log = np.min(np.mean(test_error_lamda,axis=0))
    opt_lamda = lambdas[np.argmin(np.mean(test_error_lamda,axis=0))]
    
    
    ####### KNN for outer loop with optimal values ######
    knclassifier = KNeighborsClassifier(n_neighbors=opt_k);
    knclassifier.fit(X_train, y_train);
    y_est = knclassifier.predict(X_test);
    Error_KNN_test[k] = np.sum(y_est!=y_test)/len(y_test)
    
    
    ####### Logistic Reg. for outer loop with optimal values ######
    multiReg = lm.LogisticRegression(penalty='l2', C=1/opt_lamda, max_iter=10000,multi_class='multinomial')           
    multiReg.fit(X_train, y_train)
    y_est = multiReg.predict(X_test).T       
    Error_logistic_test[k] = np.sum(y_est!=y_test)/len(y_test)
    
    
    ##### Baseline for outer loop#####
    class_num=np.zeros(3)
    class_num[0] = np.count_nonzero(y_train==0)
    class_num[1] = np.count_nonzero(y_train==1)
    class_num[2] = np.count_nonzero(y_train==2)
    
    
    y_est = np.argmax(class_num)*np.ones([len(y_test),1])
    Error_test_baseline[k] = np.sum(y_est.squeeze()!=y_test)/len(y_test)
    
    opt_lamda_array[k] = opt_lamda
    opt_K_array[k] = opt_k
    
    



    k+=1


all_errors = np.array([Error_KNN_test.squeeze(),Error_logistic_test.squeeze(),Error_test_baseline.squeeze(),opt_lamda_array,opt_K_array])

error_Table = pd.DataFrame(np.transpose(all_errors.squeeze()),columns=["KNN","Logistic_Regression","Baseline_model","lamda","K"])
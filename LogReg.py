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

# Maximum number of neighbors

lambdas = np.power(10.,range(-9,9))
K = 10
CV = model_selection.KFold(K, shuffle=True)
errors = np.zeros((K,lambdas.shape[0]))
i=0
for train_index, test_index in CV.split(X, y):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,10))    
    
    # extract training and test set for current CV fold
    X_train = X[train_index,1:]
    y_train = y[train_index]
    X_test = X[test_index,1:]
    y_test = y[test_index]

    for k in range(0, lambdas.shape[0]):
            multiReg = lm.LogisticRegression(penalty='l2', C=1/lambdas[k], max_iter=10000,multi_class='multinomial')
            
            multiReg.fit(X_train, y_train)

            y_est = multiReg.predict(X_test).T
            
            errors[i,k] = np.sum(y_est!=y_test)

    i+=1
    
# Plot the classification error rate

mean_errors = np.mean(errors,0)

figure()
plot(mean_errors)
xlabel('lamda range')
ylabel('Classification error rate (%)')
show()

print('Ran Exercise 6.3.2')


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg import svd
# reading csv files
data =  pd.read_csv('../Data/abalone.data', sep=",", header=None)
data.columns =['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']

data['Height']

data_200 = data.copy()
data_200[['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']] = data_200[['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']]*200

##########  drop outlier #############################
a=data_200['Height'].to_numpy()
np.where(a>100)
data_200['Whole_weight'][1417]
########### auto  einai to provlhmatiko
data_200['Whole_weight'][2051]
data_200 = data_200.drop(data_200.index[2051])

data_only_ratios = data.drop(['Sex', 'Rings'], axis=1)
normalized_df=(data_only_ratios-data_only_ratios.mean())/data_only_ratios.std()

X = np.array(normalized_df)



# PCA by computing SVD of Y
U,S,Vh = svd(X,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

attributeNames = normalized_df.columns
V=Vh.T
N,M = X.shape

# We saw in 2.1.3 that the first 3 components explaiend more than 90
# percent of the variance. Let's look at their coefficients:
pcs = [0,1,2]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r','g','b']
bw = .2
r = np.arange(1,M+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames)
plt.xlabel('Attributes')
plt.xticks(rotation=90)
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('NanoNose: PCA Component Coefficients')
plt.show()




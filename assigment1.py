import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

# one hot encoded dataframe
one_hot_data = pd.get_dummies(data_200)


Y  = np.array(one_hot_data.iloc[:,8])
X = np.array(one_hot_data.drop(['Rings'], axis=1))

data_only_ratios = data.drop(['Sex', 'Rings'], axis=1)
normalized_df=(data_only_ratios-data_only_ratios.mean())/data_only_ratios.std()


summary_statistics = one_hot_data.describe()
data_corr = one_hot_data.corr()

heatmap = sns.heatmap(data_corr, vmin=-1, vmax=1, annot=True)

#histograms = data.hist(figsize=(16,8))
#histograms_200 = data_200.hist(figsize=(16,8))
#histograms_only_rations = normalized_df.hist(figsize=(16,8))

# box plot

#ax = data_200.plot.box()
#ax.tick_params(labelrotation=90)

#histogram , scatter plot kai box plot
plt.figure(figsize=(20, 15))

#colors = sns.color_palette()

#rows = 1
#cols = 2

#plt.subplot(rows, cols, i)
#_ = sns.distplot(abalone['Whole weight'], color=colors[i % cols])
fig_size, axs_size = plt.subplots(3, 3, figsize=(20, 15))
sns.histplot(one_hot_data['Length'],ax=axs_size[0,0], kde=False, bins=np.arange(one_hot_data['Length'].min(), one_hot_data['Length'].max(), 10))
sns.histplot(one_hot_data['Diameter'],ax=axs_size[0,1], kde=False, bins=np.arange(one_hot_data['Diameter'].min(), one_hot_data['Diameter'].max(), 10))
sns.histplot(one_hot_data['Height'],ax=axs_size[0,2], kde=False, bins=np.arange(one_hot_data['Height'].min(), one_hot_data['Height'].max(), 5))


sns.scatterplot(data=one_hot_data, x="Length", y="Rings",ax=axs_size[1,0])
sns.scatterplot(data=one_hot_data, x="Diameter", y="Rings",ax=axs_size[1,1])
sns.scatterplot(data=one_hot_data, x="Height", y="Rings",ax=axs_size[1,2])

sns.boxplot(one_hot_data['Length'],ax=axs_size[2,0])
sns.boxplot(one_hot_data['Diameter'],ax=axs_size[2,1])
sns.boxplot(one_hot_data['Height'],ax=axs_size[2,2])


fig_size, axs_size = plt.subplots(3, 4, figsize=(20, 15))

sns.histplot(one_hot_data['Whole_weight'],ax=axs_size[0,0], kde=False, bins=np.arange(one_hot_data['Whole_weight'].min(), one_hot_data['Whole_weight'].max(), 35))
sns.histplot(one_hot_data['Shucked_weight'],ax=axs_size[0,1], kde=False, bins=np.arange(one_hot_data['Shucked_weight'].min(), one_hot_data['Shucked_weight'].max(), 20))
sns.histplot(one_hot_data['Viscera_weight'],ax=axs_size[0,2], kde=False, bins=np.arange(one_hot_data['Viscera_weight'].min(), one_hot_data['Viscera_weight'].max(), 10))
sns.histplot(one_hot_data['Shell_weight'],ax=axs_size[0,3], kde=False, bins=np.arange(one_hot_data['Shell_weight'].min(), one_hot_data['Shell_weight'].max(), 10))



sns.scatterplot(data=data_200, x="Whole_weight", y="Rings",ax=axs_size[1,0])
sns.scatterplot(data=data_200, x="Shucked_weight", y="Rings",ax=axs_size[1,1])
sns.scatterplot(data=data_200, x="Viscera_weight", y="Rings",ax=axs_size[1,2])
sns.scatterplot(data=data_200, x="Shell_weight", y="Rings",ax=axs_size[1,3])

sns.boxplot(data_200['Whole_weight'],ax=axs_size[2,0])
sns.boxplot(data_200['Shucked_weight'],ax=axs_size[2,1])
sns.boxplot(data_200['Viscera_weight'],ax=axs_size[2,2])
sns.boxplot(data_200['Shell_weight'],ax=axs_size[2,3])


#### all scater plots 

sns.pairplot(data_200)
plt.show()







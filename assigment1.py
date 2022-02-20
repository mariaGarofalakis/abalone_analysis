import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# reading csv files
data =  pd.read_csv('../Data/abalone.data', sep=",", header=None)
data.columns =['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight', 'Rings']

data_200 = data.copy()
data_200[['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']] = data_200[['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']]*200

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
sns.distplot(one_hot_data['Length'],ax=axs_size[0,0], kde=False, bins=np.arange(one_hot_data['Length'].min(), one_hot_data['Length'].max(), 10))
sns.distplot(one_hot_data['Diameter'],ax=axs_size[0,1], kde=False, bins=np.arange(one_hot_data['Diameter'].min(), one_hot_data['Diameter'].max(), 10))
sns.distplot(one_hot_data['Height'],ax=axs_size[0,2], kde=False, bins=np.arange(one_hot_data['Height'].min(), one_hot_data['Height'].max(), 5))


sns.scatterplot(data=one_hot_data, x="Length", y="Rings",ax=axs_size[1,0])
sns.scatterplot(data=one_hot_data, x="Diameter", y="Rings",ax=axs_size[1,1])
sns.scatterplot(data=one_hot_data, x="Height", y="Rings",ax=axs_size[1,2])

sns.boxplot(one_hot_data['Length'],ax=axs_size[2,0])
sns.boxplot(one_hot_data['Diameter'],ax=axs_size[2,1])
sns.boxplot(one_hot_data['Height'],ax=axs_size[2,2])








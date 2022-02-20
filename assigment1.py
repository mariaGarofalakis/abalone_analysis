import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# reading csv files
data =  pd.read_csv('../Data/abalone.data', sep=",", header=None)
data.columns =['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

Y  = np.array(data.iloc[:,8])
X = data.drop(8, axis=0)


summary_statistics = data.describe()
data_corr = data.corr()

heatmap = sns.heatmap(data_corr, vmin=-1, vmax=1, annot=True)



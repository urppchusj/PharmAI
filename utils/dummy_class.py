#%%[markdown]
# # Baseline
#
# Calculate baseline accuracies by comparing each target to 
# top1, top10 and top30 lists

#%%[markdown]
# ## Imports

#%%
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

#%%[markdown]
# ## Load the data

#%%
with open('mimic/preprocessed_data_mimic/targets_list.pkl', mode='rb') as file:
	data = pickle.load(file)
data = [drug for encounter in data.values() for drug in encounter ]

#%%[markdown]
# ## Get the list of top 1, 10 and 30 drugs

#%%
s = pd.Series(data)
counts = s.value_counts()
tops = counts.index.tolist()

top1 = tops[0]
top10 = tops[0:10]
top30 = tops[0:30]
intop1, intop10, intop30 = 0,0,0

#%%[markdown]
# ## Calculate the number of guesses by using top 1, 10 and 30 lists

#%%
for target in data:
	if target in top1:
		intop1 += 1 
	if target in top10:
		intop10 += 1 
	if target in top30:
		intop30 += 1 
denum = len(data)

#%%[markdown]
# ## Results

#%%
print('Dataset contains {} samples '.format(len(data)))
print('Baseline train-test top 1 : {:.2f}%'.format(100*intop1/denum))
print('Baseline train-test top 10 : {:.2f}%'.format(100*intop10/denum))
print('Baseline train-test top 30 : {:.2f}%'.format(100*intop30/denum))
f = sns.countplot(s, order=s.value_counts().index[:50])
f.set(xticklabels='', xlabel='classes')
plt.savefig('targets_distribution.png')

#%%

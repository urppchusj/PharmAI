#%%[markdown]
# # Evaluate a trained model on a test set

#%%[markdown]
# ## Imports

#%%
import os
import pathlib
import pickle
from datetime import datetime
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import (classification_report,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.preprocessing import label_binarize

from components import (TransformedGenerator, check_ipynb, data,
                        neural_network, transformation_pipelines)

#%%[markdown]
# ## Parameters

# %%
# Save path and parameter loading

# LOAD_MODEL_FROM specifies where the trained model will be
# loaded. Must be a subdirectory of "experiments". Will save there.
# LOAD_TEST_DATA_FROM specifies where to find the test set data.
# Must be a subdirectory of "preprocessed_data".
LOAD_MODEL_FROM = 'retrospective/32 - trained full model'
LOAD_TEST_DATA_FROM = 'valid'

save_path = os.path.join('experiments', LOAD_MODEL_FROM)
with open(os.path.join(save_path, 'hp.pkl'), mode='rb') as file:
    parameters_dict = pickle.load(file)
    param = SimpleNamespace(**parameters_dict)
    print('Parameters of trained model successfully loaded.')

# %%
# Check if running inside Jupyter notebook or not (will be used later for Keras progress bars)

in_ipynb = check_ipynb().is_inipynb()

#%%[markdown]
# ## Data
#
# Load the preprocessed data of the test set

#%%
# Load the data

d = data(LOAD_TEST_DATA_FROM, param.MODE)

if os.path.isfile(os.path.join(save_path, 'sampled_encs.pkl')):
    print('USE CAUTION ! Loaded model was trained with RESTRICTED DATA !')

d.load_data(save_path=save_path, get_profiles=False)

#%%[markdown]
# Make the data lists

#%%
_, targets, pre_seqs, post_seqs, active_meds, active_classes, depas, _, _, _, _, _, _, definitions = d.make_lists(get_valid=False, shuffle_train_set=False)

#%%[markdown]
# ## Word2vec embeddings
#
# Load the previously fitted word2vec pipeline

#%%
_, w2v = joblib.load(os.path.join(save_path, 'w2v.joblib'))

#%%[markdown]
# ## Profile state encoder (PSE)
#
# Load the previously fitted profile state encoder

#%%
_, _, pse, pse_pp, pse_a = joblib.load(os.path.join(save_path, 'pse.joblib'))

#%%[markdown]
# ## Label encoder
#
# Load the previously fitted label encoder

#%%
_, _, le = joblib.load(os.path.join(save_path, 'le.joblib'))

#%%[markdown]
# Filter out previously unseen labels and count how many
# are discarded

#%%
print('Filtering unseen targets...')
pre_discard_n_targets = len(targets)
filtered_targets = [target for target in targets if target in le.classes_]
filtered_pre_seqs = [seq for seq, target in zip(pre_seqs, targets) if target in le.classes_]
filtered_post_seqs = [seq for seq, target in zip(post_seqs, targets) if target in le.classes_]
filtered_active_meds = [active_med for active_med, target in zip(active_meds, targets) if target in le.classes_]
filtered_active_classes = [active_class for active_class, target in zip(active_classes, targets) if target in le.classes_]
filtered_depas = [depa for depa, target in zip(depas, targets) if target in le.classes_]
post_discard_n_targets = len(filtered_targets)

print('Predicting on {} samples, {:.2f} % of original samples, {} samples discarded because of unseen labels.'.format(post_discard_n_targets, 100*post_discard_n_targets/pre_discard_n_targets, pre_discard_n_targets-post_discard_n_targets))

#%%[markdown]
# ## Neural network

#%%[markdown]
# ### Sequence generators

# Build the generators, prepare the variables for fitting
w2v_step = w2v.named_steps['w2v']
eval_generator = TransformedGenerator(param.MODE, w2v_step, param.USE_LSI, pse, le, filtered_targets, filtered_pre_seqs, filtered_post_seqs, filtered_active_meds, filtered_active_classes, filtered_depas, param.W2V_EMBEDDING_DIM, param.SEQUENCE_LENGTH, param.BATCH_SIZE, shuffle=False)

#%%[markdown]
# ### Instantiate the model

#%%
n = neural_network(param.MODE)
model = tf.keras.models.load_model(os.path.join(save_path, 'model.h5'), custom_objects={'sparse_top10_accuracy':n.sparse_top10_accuracy, 'sparse_top30_accuracy':n.sparse_top30_accuracy})

#%%[markdown]
# ### Evaluate the model
#
# Use in_ipynb to check if running in Jupyter or not to print
# progress bars if in terminal and log only at epoch level if
# in Jupyter. This is a bug of Jupyter or Keras where progress
# bars will flood stdout slowing down and eventually crashing 
# the notebook.

#%%
if in_ipynb:
	verbose=2
else:
	verbose=1

#%%
results = model.evaluate_generator(eval_generator, verbose=verbose)
predictions = model.predict_generator(eval_generator, verbose=verbose)

#%%[markdown]
# ### Compute and print evaluation metrics

#%%
print('Evaluation on test set results')
print('Predictions for {} classes'.format(len(le.classes_)))
print('{} classes reprensented in targets'.format(len(set(targets))))
for metric, result in zip(model.metrics_names, results):
	print('Metric: {}   Score: {:.5f}'.format(metric,result))

#%%
prediction_labels = le.inverse_transform([np.argmax(prediction) for prediction in predictions])
pl_series = pd.Series(prediction_labels)
f = sns.countplot(pl_series, order=pl_series.value_counts().index[:50])
f.set(xticklabels='', xlabel='classes')
if in_ipynb:
	plt.show()
else:
	plt.savefig(os.path.join(save_path, 'predictions_distribution.png'))
plt.gcf().clear()
nb_predicted_classes = len(pl_series.value_counts().index)
print('Number of classes predicted on evaluation set: {}'.format(nb_predicted_classes))

#%%
cr = classification_report(filtered_targets, prediction_labels, output_dict=True)
cr_df = pd.DataFrame(cr).transpose()
cr_df.to_csv(os.path.join(save_path,  'eval_classification_report.csv'))
p_we, r_we, _, _ = precision_recall_fscore_support(filtered_targets, prediction_labels, average='weighted')
print('Weighted average precision score for present labels: {:.3f}'.format(p_we))
print('Weighted average recall score for present labels: {:.3f}'.format(r_we))

#%%
department_data = pd.read_csv('data/depas.csv', sep=';')
department_data.set_index('Numéro', inplace=True)
department_cat_dict = department_data['Catégorie'].to_dict()
department_cat_dict['null'] = 'null'
categorized_departments = [department_cat_dict[department[0] if len(department)>0 else 'null'] for department in filtered_depas]
unique_categorized_departments = list(set(categorized_departments))

#%%
# Compute evaluation metrics by department as a proxy for patient
# category. Make a dataframe with this.
results_by_depa_dict = {
	'Department':[],
	'Top 1 accuracy':[],
	'Top 10 accuracy':[],
	'Top 30 accuracy':[]
}
for department in unique_categorized_departments:
	if department == 'null':
		continue
	print('\n\nResults for category: {}'.format(department))
	indices = [i for i, value in enumerate(categorized_departments) if value == department]
	selected_pre_seqs = [filtered_pre_seqs[i] for i in indices]
	selected_post_seqs = [filtered_post_seqs[i] for i in indices]
	selected_active_meds = [filtered_active_meds[i] for i in indices]
	selected_active_classes = [filtered_active_classes[i] for i in indices]
	selected_depa = [filtered_depas[i] for i in indices]
	selected_targets = [filtered_targets[i] for i in indices]
	eval_generator = TransformedGenerator(param.MODE, w2v_step, param.USE_LSI, pse, le, selected_targets, selected_pre_seqs, selected_post_seqs, selected_active_meds, selected_active_classes, selected_depa, param.W2V_EMBEDDING_DIM, param.SEQUENCE_LENGTH, param.BATCH_SIZE)
	results = model.evaluate_generator(eval_generator, verbose=1)
	predictions = model.predict_generator(eval_generator, verbose=1)
	print('Evaluation results')
	print('Predicting on {} samples, {:.2f} % of {} samples.'.format(len(selected_targets), (100*len(selected_targets))/len(filtered_targets), len(filtered_targets)))
	print('Predictions for {} classes'.format(len(le.classes_)))
	print('{} classes reprensented in targets'.format(len(set(selected_targets))))
	for metric, result in zip(model.metrics_names, results):
		print('Metric: {}   Score: {:.5f}'.format(metric,result))
	results_by_depa_dict['Department'].append(department)
	results_by_depa_dict['Top 1 accuracy'].append(results[1])
	results_by_depa_dict['Top 10 accuracy'].append(results[2])
	results_by_depa_dict['Top 30 accuracy'].append(results[3])
	prediction_labels = le.inverse_transform([np.argmax(prediction) for prediction in predictions])
	p_we, r_we, _, _ = precision_recall_fscore_support(selected_targets, prediction_labels, average='weighted')
	print('Weighted average precision score for present labels: {:.3f}'.format(p_we))
	print('Weighted average recall score for present labels: {:.3f}'.format(r_we))


#%%
# Visualize the prediction results by department
results_by_depa_df = pd.DataFrame(results_by_depa_dict)
results_by_depa_df.set_index('Department', inplace=True)
results_by_depa_df.sort_values(by='Top 1 accuracy', inplace=True)
results_by_depa_df = results_by_depa_df.stack().reset_index()
results_by_depa_df.rename(columns={'level_1':'Metric', 0:'Result'}, inplace=True)
sns.set(style='darkgrid')
plt.xticks(rotation=60, ha='right')
sns.barplot(x='Department', y='Result', hue='Metric', data=results_by_depa_df)
if in_ipynb:
	plt.show()
else:
	plt.savefig(os.path.join(save_path, 'accuracy_by_depa.png'))

#%%

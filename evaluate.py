#%%[markdown]
# # Evaluate a trained model on a test set

#%%[markdown]
# ## Imports

#%%
import os
import pathlib
import pickle
import random
import warnings
from datetime import datetime
from itertools import chain
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


warnings.filterwarnings('ignore',category=UserWarning)

#%%[markdown]
# ## Parameters

# %%
# Save path and parameter loading

# LOAD_MODEL_FROM specifies where the trained model will be
# loaded. Must be a subdirectory of "experiments". Will save there.
# LOAD_TEST_DATA_FROM specifies where to find the test set data.
# Must be a subdirectory of "preprocessed_data".
LOAD_MODEL_FROM = 'retrospective-autoenc/20191008-0014'
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
if param.MODE == 'retrospective-autoenc':
	# No filtering
	filtered_targets = active_meds
	filtered_pre_seqs = pre_seqs
	filtered_post_seqs = post_seqs
	filtered_active_meds = active_meds
	filtered_active_classes = active_classes
	filtered_depas = depas
	post_discard_n_targets = len(filtered_targets)
else:
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
if param.MODE == 'retrospective-autoenc':
	custom_objects_dict = {'autoencoder_accuracy':n.autoencoder_accuracy}
else:
	custom_objects_dict = {'sparse_top10_accuracy': n.sparse_top10_accuracy, 'sparse_top30_accuracy': n.sparse_top30_accuracy}
model = tf.keras.models.load_model(os.path.join(save_path, 'model.h5'), custom_objects=custom_objects_dict)

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
print('{} classes reprensented in targets'.format(len(set(chain.from_iterable(filtered_targets)))))
for metric, result in zip(model.metrics_names, results):
	print('Metric: {}   Score: {:.5f}'.format(metric,result))

#%%
if param.MODE == 'retrospective-autoenc':
	prediction_label_matrix = (predictions >= 0.5) * 1
	prediction_labels = le.inverse_transform(prediction_label_matrix)
	pl_series = pd.Series(chain.from_iterable(prediction_labels))
	y_pred = prediction_label_matrix
	y_true = le.transform(filtered_targets)
else:
	prediction_labels = le.inverse_transform([np.argmax(prediction) for prediction in predictions])
	pl_series = pd.Series(prediction_labels)
	y_pred = prediction_labels
	y_true = filtered_targets
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

cr = classification_report(y_true, y_pred, output_dict=True)
cr_df = pd.DataFrame(cr).transpose()
cr_df.to_csv(os.path.join(save_path,  'eval_classification_report.csv'))
p_we, r_we, _, _ = precision_recall_fscore_support(y_true, y_pred, average='micro')
print('Micro average precision score for present labels: {:.3f}'.format(p_we))
print('Micro average recall score for present labels: {:.3f}'.format(r_we))

#%%
department_data = pd.read_csv(os.path.join(os.getcwd(), 'data', 'depas.csv'), sep=';')
department_data.set_index('orig_depa', inplace=True)
department_cat_dict = department_data['cat_depa'].to_dict()
# Account for errors in data
department_cat_dict['null'] = 'null'
department_cat_dict['nan'] = 'null'
categorized_departments = [department_cat_dict[department[0] if len(department)>0 else 'null'] for department in filtered_depas]
unique_categorized_departments = list(set(categorized_departments))

#%%
# Compute evaluation metrics by department as a proxy for patient
# category. Make a dataframe with this.
if param.MODE == 'retrospective-autoenc':
	results_by_depa_dict = {
		'Department':[],
		'Autoencoder accuracy':[]
	}
else:
	results_by_depa_dict = {
		'Department':[],
		'Dummy top 1':[],
		'Top 1 accuracy':[],
		'Dummy top 10':[],
		'Top 10 accuracy':[],
		'Dummy top 30':[],
		'Top 30 accuracy':[]
	}
for department in unique_categorized_departments:
	if department == 'null':
		continue
	print('\n\nResults for category: {}'.format(department))
	indices = [i for i, value in enumerate(categorized_departments) if value == department]
	selected_pre_seqs = [filtered_pre_seqs[i] for i in indices]
	if param.MODE == 'retrospective':
		selected_post_seqs = [filtered_post_seqs[i] for i in indices]
	elif param.MODE == 'prospective' or param.MODE == 'retrospective-autoenc':
		selected_post_seqs = []
	selected_active_meds = [filtered_active_meds[i] for i in indices]
	if len(filtered_active_classes) > 0:
		selected_active_classes = [filtered_active_classes[i] for i in indices]
	else:
		selected_active_classes = []
	selected_depa = [filtered_depas[i] for i in indices]
	selected_targets = [filtered_targets[i] for i in indices]
	if len(selected_targets) < 10:
		continue
	eval_generator = TransformedGenerator(param.MODE, w2v_step, param.USE_LSI, pse, le, selected_targets, selected_pre_seqs, selected_post_seqs, selected_active_meds, selected_active_classes, selected_depa, param.W2V_EMBEDDING_DIM, param.SEQUENCE_LENGTH, param.BATCH_SIZE)
	results = model.evaluate_generator(eval_generator, verbose=1)
	predictions = model.predict_generator(eval_generator, verbose=1)
	if param.MODE == 'retrospective-autoenc':
		for metric, result in zip(model.metrics_names, results):
			print('Metric: {}   Score: {:.5f}'.format(metric,result))
		results_by_depa_dict['Department'].append(department)
		results_by_depa_dict['Autoencoder accuracy'].append(results[1])
		prediction_label_matrix = (predictions >= 0.5) * 1
		prediction_labels = le.inverse_transform(prediction_label_matrix)
		p_we, r_we, _, _ = precision_recall_fscore_support(le.transform(selected_targets), prediction_label_matrix, average='micro')
		print('Micro average precision score for present labels: {:.3f}'.format(p_we))
		print('Micro average recall score for present labels: {:.3f}'.format(r_we))
		print('\n\n\n')
		print('Sampled predictions:\n')
		sampled_indices = random.sample(range(len(selected_targets)), 10)
		sampled_true = [[definitions[med] for med in profile] for profile in [selected_targets[i] for i in sampled_indices]]
		sampled_pred = [[definitions[med] for med in profile] for profile in [prediction_labels[i] for i in sampled_indices]]
		for true, pred in zip(sampled_true, sampled_pred):
			true.sort()
			pred.sort()
			print('\nTRUE PROFILE: \n{} \nPREDICTED PROFILE:{}\n'.format(true, pred))
		print('\n\n\n')
	else:
		# Compute the baseline predictions by frequency-based prediction
		s = pd.Series(selected_targets)
		counts = s.value_counts()
		tops = counts.index.tolist()
		top1 = tops[0]
		top10 = tops[0:10]
		top30 = tops[0:30]
		intop1, intop10, intop30 = 0,0,0
		for target in selected_targets:
			if target in top1:
				intop1 += 1 
			if target in top10:
				intop10 += 1 
			if target in top30:
				intop30 += 1 
			denum = len(selected_targets)
		top1_baseline = intop1/denum
		top10_baseline = intop10/denum
		top30_baseline = intop30/denum
		print('Baseline accuracy by dummy classifier')
		print('Baseline train-test top 1 : {:.2f}%'.format(100*intop1/denum))
		print('Baseline train-test top 10 : {:.2f}%'.format(100*intop10/denum))
		print('Baseline train-test top 30 : {:.2f}%'.format(100*intop30/denum))
		print('\n\nEvaluation results')
		print('Predicting on {} samples, {:.2f} % of {} samples.'.format(len(selected_targets), (100*len(selected_targets))/len(filtered_targets), len(filtered_targets)))
		print('Predictions for {} classes'.format(len(le.classes_)))
		print('{} classes reprensented in targets'.format(len(set(selected_targets))))
		for metric, result in zip(model.metrics_names, results):
			print('Metric: {}   Score: {:.5f}'.format(metric,result))
		results_by_depa_dict['Department'].append(department)
		results_by_depa_dict['Dummy top 1'].append(top1_baseline)
		results_by_depa_dict['Top 1 accuracy'].append(results[1])
		results_by_depa_dict['Dummy top 10'].append(top10_baseline)
		results_by_depa_dict['Top 10 accuracy'].append(results[2])
		results_by_depa_dict['Dummy top 30'].append(top30_baseline)
		results_by_depa_dict['Top 30 accuracy'].append(results[3])
		prediction_labels = le.inverse_transform([np.argmax(prediction) for prediction in predictions])
		p_we, r_we, _, _ = precision_recall_fscore_support(selected_targets, prediction_labels, average='weighted')
		print('Weighted average precision score for present labels: {:.3f}'.format(p_we))
		print('Weighted average recall score for present labels: {:.3f}'.format(r_we))


#%%
# Visualize the prediction results by department
results_by_depa_df = pd.DataFrame(results_by_depa_dict)
results_by_depa_df.set_index('Department', inplace=True)
if param.MODE == 'retrospective-autoenc':
	sort_string = 'Autoencoder accuracy'
else:
	sort_string = 'Top 1 accuracy'
results_by_depa_df.sort_values(by=sort_string, inplace=True)
results_by_depa_df = results_by_depa_df.stack().reset_index()
results_by_depa_df.rename(columns={'level_1':'Metric', 0:'Result'}, inplace=True)
sns.set(style='darkgrid')
a4_dims = (11.7, 8.27)
fig, ax = plt.subplots(figsize=a4_dims)
plt.xticks(rotation=60, ha='right')
sns.barplot(ax=ax, x='Department', y='Result', hue='Metric', data=results_by_depa_df)
if in_ipynb:
	plt.show()
else:
	plt.savefig(os.path.join(save_path, 'accuracy_by_depa.png'))

#%%

import argparse as ap
import os
import pathlib

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def makegraphs(datapath):
	# load the data files and concatenate them into a single pandas dataframe
	files_data = []
	for file in os.listdir(datapath):
		if file.endswith('.csv'):
			file_df = pd.read_csv(os.path.join(datapath, file))
			file_df['Parameters'] = os.path.splitext(file)[0]
			files_data.append(file_df)
	all_data = pd.concat(files_data)

	# make the graphs

	all_data_filtered = all_data[['Parameters', 'model_3_autoencoder_accuracy', 'model_3_aupr', 'val_model_3_autoencoder_accuracy', 'val_model_3_aupr']].copy()
	all_data_filtered.rename(inplace=True, index=str, columns={'model_3_autoencoder_accuracy':'Train accuracy', 'model_3_aupr':'Train AUPR', 'val_model_3_autoencoder_accuracy':'Val accuracy', 'val_model_3_aupr':'Val AUPR'})
	all_data_filtered.set_index('Parameters', inplace=True)
	all_data_graph_df = all_data_filtered.stack().reset_index()
	all_data_graph_df.rename(inplace=True, index=str, columns={'level_0':'Parameters', 'level_1':'Metric', 0:'Result'})
	#all_data_graph_df['Latent dim - Dropout rate'] = all_data_graph_df['Latent dim - Dropout rate'].astype('int8')
	sns.set(font_scale=1.5, style="whitegrid")
	f = sns.catplot(x="Parameters", y="Result", hue="Metric", hue_order=['Train accuracy', 'Val accuracy', 'Train AUPR', 'Val AUPR'], kind="point", data=all_data_graph_df)
	f.set_xticklabels(rotation=20,  horizontalalignment='right')
	#f.set_xlabels('')
	plt.gcf().subplots_adjust(bottom=0.2)
	plt.savefig(os.path.join(datapath, 'summary', 'experiments_results_acc.png'))
	results = pd.concat([all_data_graph_df.groupby(['Parameters', 'Metric']).mean(),all_data_graph_df.groupby(['Parameters', 'Metric']).std()], axis=1)
	results.to_csv(os.path.join(datapath, 'summary', 'experiment_results_acc.csv'))

	all_data_filtered = all_data[['Parameters', 'model_3_autoencoder_false_neg_rate', 'val_model_3_autoencoder_false_neg_rate']].copy()
	all_data_filtered.rename(inplace=True, index=str, columns={'model_3_autoencoder_false_neg_rate':'Train false negative rate', 'val_model_3_autoencoder_false_neg_rate':'Val false negative rate'})
	all_data_filtered.set_index('Parameters', inplace=True)
	all_data_graph_df = all_data_filtered.stack().reset_index()
	all_data_graph_df.rename(inplace=True, index=str, columns={'level_0':'Parameters', 'level_1':'Metric', 0:'Result'})
	#all_data_graph_df['Latent dim - Dropout rate'] = all_data_graph_df['Latent dim - Dropout rate'].astype('int8')
	sns.set(font_scale=1.5, style="whitegrid")
	f = sns.catplot(x="Parameters", y="Result", hue="Metric", hue_order=['Train false negative rate', 'Val false negative rate'], kind="point", data=all_data_graph_df)
	f.set_xticklabels(rotation=20,  horizontalalignment='right')
	#f.set_xlabels('')
	plt.gcf().subplots_adjust(bottom=0.2)
	plt.savefig(os.path.join(datapath, 'summary', 'experiments_results_atypical.png'))
	results = pd.concat([all_data_graph_df.groupby(['Parameters', 'Metric']).mean(),all_data_graph_df.groupby(['Parameters', 'Metric']).std()], axis=1)
	results.to_csv(os.path.join(datapath, 'summary', 'experiment_results_atypical.csv'))

	all_data_filtered_loss = all_data[['Parameters', 'model_3_loss', 'val_model_3_loss', 'model_2_loss', 'val_model_2_loss', 'model_loss', 'val_model_loss']].copy()
	all_data_filtered_loss.rename(inplace=True, index=str, columns={'model_3_loss': 'Contextual loss', 'val_model_3_loss':'Val contextual loss', 'model_2_loss': 'Encoder loss', 'val_model_2_loss':'Val encoder loss', 'model_loss': 'Adversarial loss', 'val_model_loss':'Val adversarial loss'})
	all_data_filtered_loss.set_index('Parameters', inplace=True)
	all_data_loss_graph_df = all_data_filtered_loss.stack().reset_index()
	all_data_loss_graph_df.rename(inplace=True, index=str, columns={'level_0':'Parameters', 'level_1':'Metric', 0:'Loss'})
	#all_data_loss_graph_df['Latent dim - Dropout rate'] = all_data_loss_graph_df['Latent dim - Dropout rate'].astype('int8')
	sns.set(font_scale=1.5, style="whitegrid")
	f = sns.catplot(x="Parameters", y="Loss", hue="Metric", kind="point", data=all_data_loss_graph_df)
	f.set_xticklabels(rotation=20,  horizontalalignment='right')
	#f.set_xlabels('')
	plt.gcf().subplots_adjust(bottom=0.2)
	plt.savefig(os.path.join(datapath, 'summary', 'experiments_results_loss.png'))
	results = pd.concat([all_data_graph_df.groupby(['Parameters', 'Metric']).mean(),all_data_graph_df.groupby(['Parameters', 'Metric']).std()], axis=1)
	results.to_csv(os.path.join(datapath, 'summary', 'experiment_results_loss.csv'))

#############
## EXECUTE ##
#############

if __name__ == '__main__':

	parser = ap.ArgumentParser(description='Make the summary graphs for model evaluation', formatter_class=ap.RawTextHelpFormatter)
	parser.add_argument('--datadir', metavar='Type_String', type=str, nargs="?", default='', help='The directory where csv files are located. No default.')

	args = parser.parse_args()
	datadir = args.datadir

	datapath = os.path.join(os.getcwd(), datadir)
	pathlib.Path(os.path.join(datapath, 'summary')).mkdir(parents=True, exist_ok=True)
	makegraphs(datapath)

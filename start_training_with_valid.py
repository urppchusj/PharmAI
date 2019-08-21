#%% [markdown]
# # Model.py
#
# Start training the model with validation.

#%% [markdown]
# ## Setup
#
# Setup the environment
#%%
# Imports

import os
import pathlib
import pickle
from datetime import datetime
from multiprocessing import cpu_count

import gensim.utils as gsu
import joblib
import pandas as pd
import tensorflow as tf
from gensim.matutils import Sparse2Corpus
from gensim.models import KeyedVectors
from gensim.sklearn_api import LsiTransformer, W2VTransformer
from joblib import dump
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder

from components import (TransformedGenerator, check_ipynb, data,
                        neural_network, pse_helper_functions, visualization)

#%%[markdown]
# ## Global variables

#%%
# Save path

SAVE_STAMP = datetime.now().strftime('%Y%m%d-%H%M')
SAVE_PATH = os.path.join(os.getcwd(), 'experiments', SAVE_STAMP)
pathlib.Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

#%%
# Data variables

RESTRICT_DATA = False
RESTRICT_SAMPLE_SIZE = 1000
DATA_DIR = '5yr'

#%%
# Word2vec parameters

W2V_ALPHA = 0.013
W2V_ITER = 32
W2V_EMBEDDING_DIM = 128
W2V_HS = 0
W2V_SG =  0
W2V_MIN_COUNT = 5
W2V_WORKERS = cpu_count()

#%%
# PSE parameters

USE_LSI = False
PSE_SHAPE = 200

#%%
# Neural network parameters

N_LSTM = 0
N_PSE_DENSE = 0
N_DENSE = 2

LSTM_SIZE = 512
DENSE_PSE_SIZE = 256
CONCAT_LSTM_SIZE = 512
CONCAT_TOTAL_SIZE = 512
DENSE_SIZE = 128
DROPOUT = 0.3
L2_REG = 0
SEQUENCE_LENGTH = 30
BATCH_SIZE = 256

#%%
# Check if running inside Jupyter notebook or not (will be used later for Keras progress bars)

in_ipynb = check_ipynb().is_inipynb()

#%% [markdown]
# ## Data
#
# Prepare the data

#%%
# Load the data

d = data(DATA_DIR)
d.load_data(restrict_data=RESTRICT_DATA, restrict_sample_size=RESTRICT_SAMPLE_SIZE)
if RESTRICT_DATA:
	with open(os.path.join(SAVE_PATH, 'sampled_encs.pkl'), mode='wb') as file:
		pickle.dump(d.enc, file)

#%%
# Split encounters into a train and test set

d.split()

#%%
# Make the data lists

profiles_train, targets_train, pre_seq_train, post_seq_train, active_profiles_train, active_classes_train, depa_train, targets_test, pre_seq_test, post_seq_test, active_profiles_test, active_classes_test, depa_test, definitions = d.make_lists()

#%% [markdown]
# ## Word2vec embeddings
#
# Create a word2vec pipeline and train word2vec embeddings in that pipeline on the training set profiles

#%%
# Define and train the pipeline

w2v = Pipeline([
	('w2v', W2VTransformer(alpha=W2V_ALPHA, iter=W2V_ITER, size=W2V_EMBEDDING_DIM, hs=W2V_HS, sg=W2V_SG, min_count=W2V_MIN_COUNT, workers=W2V_WORKERS)),
	])

print('Fitting word2vec embeddings...')
w2v.fit(profiles_train)
# Normalize the embeddings
w2v.named_steps['w2v'].gensim_model.init_sims(replace=True)
# save the fitted word2vec pipe
joblib.dump((W2V_EMBEDDING_DIM, SEQUENCE_LENGTH, w2v), os.path.join(SAVE_PATH, 'w2v.joblib'))

#%%
# OPTIONAL: Export w2v embeddings

print('Exporting word2vec embeddings...')
w2v.named_steps['w2v'].gensim_model.wv.save_word2vec_format(os.path.join(SAVE_PATH, 'w2v.model'))
model = KeyedVectors.load_word2vec_format(os.path.join(SAVE_PATH,'w2v.model'), binary=False)
outfiletsv = os.path.join(SAVE_PATH, 'w2v_embeddings.tsv')
outfiletsvmeta = os.path.join(SAVE_PATH, 'w2v_metadata.tsv')

with open(outfiletsv, 'w+') as file_vector:
	with open(outfiletsvmeta, 'w+') as file_metadata:
		for word in model.index2word:
			file_metadata.write(gsu.to_utf8(word).decode('utf-8') + gsu.to_utf8('\n').decode('utf-8'))
			vector_row = '\t'.join(str(x) for x in model[word])
			file_vector.write(vector_row + '\n')

print("2D tensor file saved to %s", outfiletsv)
print("Tensor metadata file saved to %s", outfiletsvmeta)

with open(outfiletsvmeta, mode='r', encoding='utf-8', errors='strict') as metadata_file:
	metadata = metadata_file.read()
converted_string = ''
for element in metadata.splitlines():
	string = element.strip()
	converted_string +=	definitions[string] + '\n'
with open(os.path.join(SAVE_PATH, 'w2v_defined_metadata.tsv'), mode='w', encoding='utf-8', errors='strict') as converted_metadata:
	converted_metadata.write(converted_string)

#%% [markdown]
# ## Profile state encoder (PSE)
#
# Encode the profile state, either as a multi-hot vector (binary count vectorizer) or using Latent Semantic Indexing

#%%
# PSE helper functions

phf = pse_helper_functions()
pse_pp = phf.pse_pp
pse_a = phf.pse_a

#%%
# Prepare the data for the pipeline and fit the PSE encoder to the training set

print('Preparing data for PSE...')
pse_data = [[ap, ac, de] for ap, ac, de in zip(active_profiles_train, active_classes_train, depa_train)]
n_pse_columns = len(pse_data[0])

pse_transformers = []
for i in range(n_pse_columns):
	pse_transformers.append(('pse{}'.format(i), CountVectorizer(lowercase=False, preprocessor=pse_pp, analyzer=pse_a), i))
pse_pipeline_transformers = [
	('columntrans', ColumnTransformer(transformers=pse_transformers))
	]
if USE_LSI == True:
	pse_pipeline_transformers.extend([
		('tfidf', TfidfTransformer()),
		('sparse2corpus', FunctionTransformer(func=Sparse2Corpus, accept_sparse=True, validate=False, kw_args={'documents_columns':False})),
		('tsvd', LsiTransformer(PSE_SHAPE))
	])
pse = Pipeline(pse_pipeline_transformers)

print('Fitting PSE...')
pse.fit(pse_data)
# save the fitted profile state encoder
joblib.dump((USE_LSI, PSE_SHAPE, pse), os.path.join(SAVE_PATH, 'pse.joblib'))

#%% [markdown]
# ## Label encoder
#
# Encode the targets

#%%
# Fit the encoder to the train set targets

le = LabelEncoder()
le.fit(targets_train)
# save the fitted label encoder
joblib.dump(le, os.path.join(SAVE_PATH, 'le.joblib'))

#%% [markdown]
# ## Neural network
#
# Train a neural network to predict each drug present in a pharmacological profile from the sequence of drug orders that came before or after it and the profile state excluding that drug.

#%%
# Get the variables necessary to train the model

w2v_step = w2v.named_steps['w2v']
if USE_LSI == False:
	PSE_SHAPE = sum([len(transformer[1].vocabulary_) for transformer in pse.named_steps['columntrans'].transformers_])
else:
	PSE_SHAPE = PSE_SHAPE
output_n_classes = len(le.classes_)

#%%
# Build the generators
train_generator = TransformedGenerator(w2v_step, USE_LSI, pse, le, targets_train, pre_seq_train, post_seq_train, active_profiles_train, active_classes_train, depa_train, W2V_EMBEDDING_DIM, SEQUENCE_LENGTH, BATCH_SIZE)

test_generator = TransformedGenerator(w2v_step, USE_LSI, pse, le, targets_test, pre_seq_test, post_seq_test, active_profiles_test, active_classes_test, depa_test, W2V_EMBEDDING_DIM, SEQUENCE_LENGTH, BATCH_SIZE, shuffle=False)

#%%

# Define the callbacks
n = neural_network()
callbacks = n.callbacks(SAVE_PATH)

#%%
# Build the model
model = n.define_model(LSTM_SIZE, N_LSTM, DENSE_PSE_SIZE, CONCAT_LSTM_SIZE, CONCAT_TOTAL_SIZE, DENSE_SIZE, DROPOUT, L2_REG, SEQUENCE_LENGTH, W2V_EMBEDDING_DIM, PSE_SHAPE, N_PSE_DENSE, N_DENSE, output_n_classes)
tf.keras.utils.plot_model(model, to_file=os.path.join(SAVE_PATH, 'model.png'))

#%%
# Train the model

# Check if running in Jupyter or not to print progress bars if in terminal and log only at epoch level if in Jupyter.
if in_ipynb:
	verbose=2
else:
	verbose=1

model.fit_generator(train_generator,
	epochs=1000,
	callbacks=callbacks,
	validation_data=test_generator,
	verbose=verbose)

model.save(os.path.join(SAVE_PATH, 'model.h5'))

#%%
# Plot the loss and accuracy during training

v = visualization()

history_df = pd.read_csv(os.path.join(SAVE_PATH, 'training_history.csv'))

v.plot_accuracy_history(history_df, SAVE_PATH)
v.plot_loss_history(history_df, SAVE_PATH)

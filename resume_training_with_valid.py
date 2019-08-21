#%% [markdown]
# # Model.py
#
# Resume training the model after interruption.

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

SAVE_DIR = '20190729-2025'
SAVE_PATH = os.path.join('experiments', SAVE_DIR)

#%%
# Data variables

DATA_DIR = '5yr'

#%%
# Neural network parameters

BATCH_SIZE = 256

#%%
# Check if running inside Jupyter notebook or not (will be used later for Keras progress bars)

in_ipynb = check_ipynb().is_inipynb()

#%%
# Define simple names for keras components

load_model = tf.keras.models.load_model

#%% [markdown]
# ## Data
#
# Load the data to resume the training

#%%
# Load the data

d = data(DATA_DIR)

if os.path.isfile(os.path.join(SAVE_PATH, 'sampled_encs.pkl')):
	enc_file = os.path.join(SAVE_PATH, 'sampled_encs.pkl')
	print('Loaded partially completed experiment was done with RESTRICTED DATA !')
else:
	enc_file = False

d.load_data(previous_encs_path=enc_file, get_profiles=False, get_definitions=False)

#%%
# Split encounters into a train and test set

d.split()

#%%
# Make the data lists

_, targets_train, pre_seq_train, post_seq_train, active_profiles_train, active_classes_train, depa_train, targets_test, pre_seq_test, post_seq_test, active_profiles_test, active_classes_test, depa_test, _ = d.make_lists()

#%% [markdown]
# ## Word2vec embeddings
#
# Load the previously fitted word2vec pipeline

#%%
# Load the pipeline

W2V_EMBEDDING_DIM, SEQUENCE_LENGTH, w2v = joblib.load(os.path.join(SAVE_PATH, 'w2v.joblib'))

#%% [markdown]
# ## Profile state encoder (PSE)
#
# Load the previously fitted profile state encoder

#%%
# PSE helper functions

phf = pse_helper_functions()
pse_pp = phf.pse_pp
pse_a = phf.pse_a

#%%
# Load the pipeline

USE_LSI, PSE_SHAPE, pse = joblib.load(os.path.join(SAVE_PATH, 'pse.joblib'))

#%% [markdown]
# ## Label encoder
#
# Load the previously fitted label encoder

#%%
# Load the pipeline

le = joblib.load(os.path.join(SAVE_PATH, 'le.joblib'))

#%% [markdown]
# ## Neural network
#
# Load the partially fitted neural network and resume training

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
# Load the model
model = load_model(os.path.join(SAVE_PATH, 'partially_trained_model.h5'), custom_objects={'sparse_top10_accuracy':n.sparse_top10_accuracy, 'sparse_top30_accuracy':n.sparse_top30_accuracy})

#%%
# Resume training the model

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

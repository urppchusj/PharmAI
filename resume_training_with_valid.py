# %% [markdown]
# # Resume training the model after interruption.

# %% [markdown]
# ## Setup
#
# Setup the environment
# %%
# Imports

import os
import pathlib
import pickle
from datetime import datetime
from multiprocessing import cpu_count

import joblib
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from components import (TransformedGenerator, check_ipynb, data,
                        neural_network, transformation_pipelines, visualization)

# %%[markdown]
# ## Global variables

# %%
# Save path

SAVE_DIR = 'retrospective/20190824-1409'
SAVE_PATH = os.path.join('experiments', SAVE_DIR)

# %%
# Load parameters

with open(os.path.join(SAVE_PATH, 'hp.pkl'), mode='rb') as file:
    mode, data_dir, batch_size, sequence_length, n_training_steps_per_epoch, n_validation_steps_per_epoch = pickle.load(
        file)

with open(os.path.join(SAVE_PATH, 'done_epochs.pkl'), mode='rb') as file:
    initial_epoch = pickle.load(file) + 1

# %%
# Check if running inside Jupyter notebook or not (will be used later for Keras progress bars)

in_ipynb = check_ipynb().is_inipynb()

# %%
# Define simple names for keras components

load_model = tf.keras.models.load_model

# %% [markdown]
# ## Data
#
# Load the data to resume the training

# %%
# Load the data

d = data(data_dir, mode)

if os.path.isfile(os.path.join(SAVE_PATH, 'sampled_encs.pkl')):
    enc_file = os.path.join(SAVE_PATH, 'sampled_encs.pkl')
    print('Loaded partially completed experiment was done with RESTRICTED DATA !')
else:
    enc_file = False

d.load_data(previous_encs_path=enc_file,
            get_profiles=False, get_definitions=False)

# %%
# Split encounters into a train and test set

d.split()

# %%
# Make the data lists

_, targets_train, pre_seq_train, post_seq_train, active_meds_train, active_classes_train, depa_train, targets_test, pre_seq_test, post_seq_test, active_meds_test, active_classes_test, depa_test, _ = d.make_lists()

# %% [markdown]
# ## Word2vec embeddings
#
# Load the previously fitted word2vec pipeline

# %%
# Load the pipeline and required data

w2v_embedding_dim, w2v = joblib.load(os.path.join(SAVE_PATH, 'w2v.joblib'))

# %% [markdown]
# ## Profile state encoder (PSE)
#
# Load the previously fitted profile state encoder

# %%
# Load the pipeline and required data

use_lsi, pse_shape, pse, pse_pp, pse_a = joblib.load(
    os.path.join(SAVE_PATH, 'pse.joblib'))

# %% [markdown]
# ## Label encoder
#
# Load the previously fitted label encoder

# %%
# Load the pipeline

le, output_n_classes = joblib.load(os.path.join(SAVE_PATH, 'le.joblib'))

# %% [markdown]
# ## Neural network
#
# Load the partially fitted neural network and resume training

# %%
# Build the generators

w2v_step = w2v.named_steps['w2v']

train_generator = TransformedGenerator(mode, w2v_step, use_lsi, pse, le, targets_train, pre_seq_train, post_seq_train,
                                       active_meds_train, active_classes_train, depa_train, w2v_embedding_dim, sequence_length, batch_size)

test_generator = TransformedGenerator(mode, w2v_step, use_lsi, pse, le, targets_test, pre_seq_test, post_seq_test,
                                      active_meds_test, active_classes_test, depa_test, w2v_embedding_dim, sequence_length, batch_size, shuffle=False)

# %%
# Define the callbacks

n = neural_network(mode)
callbacks = n.callbacks(SAVE_PATH)

# %%
# Load the model
model = load_model(os.path.join(SAVE_PATH, 'partially_trained_model.h5'), custom_objects={
                   'sparse_top10_accuracy': n.sparse_top10_accuracy, 'sparse_top30_accuracy': n.sparse_top30_accuracy})

# %%
# Resume training the model

# Check if running in Jupyter or not to print progress bars if in terminal and log only at epoch level if in Jupyter.
if in_ipynb:
    verbose = 2
else:
    verbose = 1

model.fit_generator(train_generator,
                    epochs=1000,
                    steps_per_epoch=n_training_steps_per_epoch,
                    callbacks=callbacks,
                    initial_epoch=initial_epoch,
                    validation_data=test_generator,
                    validation_steps=n_validation_steps_per_epoch,
                    verbose=verbose)

model.save(os.path.join(SAVE_PATH, 'model.h5'))

# %%
# Plot the loss and accuracy during training

v = visualization()

history_df = pd.read_csv(os.path.join(SAVE_PATH, 'training_history.csv'))

v.plot_accuracy_history(history_df, SAVE_PATH)
v.plot_loss_history(history_df, SAVE_PATH)

# %% [markdown]
# # Start training the model with validation.

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

import pandas as pd
import tensorflow as tf

from components import (TransformedGenerator, check_ipynb, data,
                        neural_network, transformation_pipelines,
                        visualization)

# %%[markdown]
# ## Global variables

# %%
# Execution mode

MODE = 'retrospective'

# %%
# Save path

SAVE_STAMP = datetime.now().strftime('%Y%m%d-%H%M')
SAVE_PATH = os.path.join(os.getcwd(), 'experiments', MODE, SAVE_STAMP)
pathlib.Path(SAVE_PATH).mkdir(parents=True, exist_ok=True)

# %%
# Data variables

RESTRICT_DATA = False
RESTRICT_SAMPLE_SIZE = 1000
DATA_DIR = '5yr'

# %%
# Word2vec parameters

W2V_ALPHA = 0.013
W2V_ITER = 32
W2V_EMBEDDING_DIM = 128
W2V_HS = 0
W2V_SG = 0
W2V_MIN_COUNT = 5
W2V_WORKERS = cpu_count()
EXPORT_W2V_EMBEDDINGS = True

# %%
# PSE parameters

USE_LSI = False
TSVD_N_COMPONENTS = 200

# %%
# Neural network parameters

N_LSTM = 0
N_PSE_DENSE = 0
N_DENSE = 2

LSTM_SIZE = 512
DENSE_PSE_SIZE = 512
CONCAT_LSTM_SIZE = 1024
CONCAT_TOTAL_SIZE = 1024
DENSE_SIZE = 256
DROPOUT = 0.3
L2_REG = 0
SEQUENCE_LENGTH = 30
BATCH_SIZE = 256

N_TRAINING_STEPS_PER_EPOCH = 1000
N_VALIDATIONS_STEPS_PER_EPOCH = 1000

# %%
# Check if running inside Jupyter notebook or not (will be used later for Keras progress bars)

in_ipynb = check_ipynb().is_inipynb()

# %% [markdown]
# Save parameters

with open(os.path.join(SAVE_PATH, 'hp.pkl'), mode='wb') as file:
    pickle.dump((MODE, DATA_DIR, BATCH_SIZE, SEQUENCE_LENGTH, N_TRAINING_STEPS_PER_EPOCH, N_VALIDATIONS_STEPS_PER_EPOCH), file)

# %% [markdown]
# ## Data
#
# Prepare the data

# %%
# Load the data

d = data(DATA_DIR, mode=MODE)
d.load_data(restrict_data=RESTRICT_DATA, save_path=SAVE_PATH,
            restrict_sample_size=RESTRICT_SAMPLE_SIZE)

# %%
# Split encounters into a train and test set

d.split()

# %%
# Make the data lists

profiles_train, targets_train, pre_seq_train, post_seq_train, active_meds_train, active_classes_train, depa_train, targets_test, pre_seq_test, post_seq_test, active_meds_test, active_classes_test, depa_test, definitions = d.make_lists()

# %% [markdown]
# ## Data transformation pipelines
#
# Build and fit the data transformation pipelines for word2vec embeddings,
# profile state encoder and label encoder.

# %%
tp = transformation_pipelines()

# %% [markdown]
# ### Word2vec embeddings
# Create a word2vec pipeline and train word2vec embeddings in that pipeline on the training set profiles. Optionnaly export word2vec embeddings

# %%
tp.define_w2v_pipeline(W2V_ALPHA, W2V_ITER, W2V_EMBEDDING_DIM,
                       W2V_HS, W2V_SG, W2V_MIN_COUNT, W2V_WORKERS)
w2v = tp.fitsave_w2v_pipeline(SAVE_PATH, profiles_train, W2V_EMBEDDING_DIM, SEQUENCE_LENGTH)
if EXPORT_W2V_EMBEDDINGS:
    tp.export_w2v_embeddings(SAVE_PATH, definitions_dict=definitions)

# %% [markdown]
# ### Profile state encoder (PSE)
# Encode the profile state, either as a multi-hot vector (binary count vectorizer) or using Latent Semantic Indexing

# %%
# Prepare the data before fitting
pse_data = tp.prepare_pse_data(
    active_meds_train, active_classes_train, depa_train)
tp.define_pse_pipeline(use_lsi=USE_LSI, tsvd_n_components=TSVD_N_COMPONENTS)
pse, pse_shape = tp.fitsave_pse_pipeline(SAVE_PATH, pse_data)

# %% [markdown]
# ### Label encoder
# Encode the targets

# %%
le, output_n_classes = tp.fitsave_labelencoder(SAVE_PATH, targets_train)

# %% [markdown]
# ## Neural network
#
# Train a neural network to predict each drug present in a pharmacological profile from the sequence of drug orders that came before or after it and the profile state excluding that drug.

# %%
# Build the generators
w2v_step = w2v.named_steps['w2v']
train_generator = TransformedGenerator(MODE, w2v_step, USE_LSI, pse, le, targets_train, pre_seq_train, post_seq_train,
                                       active_meds_train, active_classes_train, depa_train, W2V_EMBEDDING_DIM, SEQUENCE_LENGTH, BATCH_SIZE)

test_generator = TransformedGenerator(MODE, w2v_step, USE_LSI, pse, le, targets_test, pre_seq_test, post_seq_test,
                                      active_meds_test, active_classes_test, depa_test, W2V_EMBEDDING_DIM, SEQUENCE_LENGTH, BATCH_SIZE, shuffle=False)

# %%
# Define the callbacks
n = neural_network(MODE)
callbacks = n.callbacks(SAVE_PATH)

# %%
# Build the model
model = n.define_model(LSTM_SIZE, N_LSTM, DENSE_PSE_SIZE, CONCAT_LSTM_SIZE, CONCAT_TOTAL_SIZE, DENSE_SIZE,
                       DROPOUT, L2_REG, SEQUENCE_LENGTH, W2V_EMBEDDING_DIM, pse_shape, N_PSE_DENSE, N_DENSE, output_n_classes)
tf.keras.utils.plot_model(model, to_file=os.path.join(SAVE_PATH, 'model.png'))

# %%
# Train the model

# Check if running in Jupyter or not to print progress bars if in terminal and log only at epoch level if in Jupyter.
if in_ipynb:
    verbose = 2
else:
    verbose = 1

model.fit_generator(train_generator,
                    epochs=1000,
                    steps_per_epoch=N_TRAINING_STEPS_PER_EPOCH,
                    callbacks=callbacks,
                    validation_data=test_generator,
                    validation_steps=N_VALIDATIONS_STEPS_PER_EPOCH,
                    verbose=verbose)

model.save(os.path.join(SAVE_PATH, 'model.h5'))

# %%
# Plot the loss and accuracy during training

v = visualization()

history_df = pd.read_csv(os.path.join(SAVE_PATH, 'training_history.csv'))

v.plot_accuracy_history(history_df, SAVE_PATH)
v.plot_loss_history(history_df, SAVE_PATH)

#%% [markdown]
# # tune_model.py
#
# This file generates a Jupyter notebook of experiments to tune the hyperparameters of a model that predicts each drug in a pharmacological profile from the context. The tuning is done using keras-tuner.

#%% [markdown]
# ## Setup
#
# Setup the environment
#%%
# Imports

import os
import pathlib
import pickle
import random
from datetime import datetime
from multiprocessing import cpu_count

import gensim.utils as gsu
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from gensim.matutils import Sparse2Corpus
from gensim.models import KeyedVectors
from gensim.sklearn_api import LsiTransformer, W2VTransformer
from gensim.utils import RULE_KEEP
from joblib import dump, load
from kerastuner.tuners import RandomSearch
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import mean_squared_error as skmse
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, LabelBinarizer,
								   LabelEncoder)

#%%
# Define simple names for keras components

EarlyStopping = tf.keras.callbacks.EarlyStopping
ReduceLROnPlateau = tf.keras.callbacks.ReduceLROnPlateau
RNN = tf.keras.layers.RNN
LSTMCell = tf.keras.layers.LSTMCell
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
Input = tf.keras.layers.Input
Masking = tf.keras.layers.Masking
BatchNormalization = tf.keras.layers.BatchNormalization
concatenate = tf.keras.layers.concatenate
Adam = tf.keras.optimizers.Adam
sparse_top_k_categorical_accuracy = tf.keras.metrics.sparse_top_k_categorical_accuracy
Model = tf.keras.models.Model
pad_sequences = tf.keras.preprocessing.sequence.pad_sequences
Sequence = tf.keras.utils.Sequence

#%% [markdown]
# ## Data
#
# Prepare the data

#%%
# Data variables

SAMPLE_SIZE = 1000
DATA_DIR = '1yr'

#%%
# Data location

definitions_file = os.path.join(os.getcwd(), 'data', 'definitions.csv')
profiles_file = os.path.join(os.getcwd(), 'preprocessed_data', 'whole_profile',DATA_DIR, 'profiles_list.pkl')
targets_file = os.path.join(os.getcwd(), 'preprocessed_data', 'whole_profile', DATA_DIR, 'targets_list.pkl')
pre_seq_file = os.path.join(os.getcwd(), 'preprocessed_data', 'whole_profile', DATA_DIR, 'pre_seq_list.pkl')
post_seq_file = os.path.join(os.getcwd(), 'preprocessed_data', 'whole_profile', DATA_DIR, 'post_seq_list.pkl')
activeprofiles_file = os.path.join(os.getcwd(), 'preprocessed_data', 'whole_profile', DATA_DIR, 'active_profiles_list.pkl')
activeclasses_file = os.path.join(os.getcwd(), 'preprocessed_data', 'whole_profile', DATA_DIR, 'active_classes_list.pkl')
depa_file = os.path.join(os.getcwd(), 'preprocessed_data', 'whole_profile', DATA_DIR, 'depa_list.pkl')
enc_file = os.path.join(os.getcwd(), 'preprocessed_data', 'whole_profile', DATA_DIR, 'enc_list.pkl')

#%%
# Load the data

print('Loading profiles...')
with open(profiles_file, mode='rb') as file:
	profiles = pickle.load(file)
print('Loading targets...')
with open(targets_file, mode='rb') as file:
	targets = pickle.load(file)
print('Loading pre sequences...')
with open(pre_seq_file, mode='rb') as file:
	pre_seqs = pickle.load(file)
print('Loading post sequences...')
with open(post_seq_file, mode='rb') as file:
	post_seqs = pickle.load(file)
print('Loading active profiles...')
with open(activeprofiles_file, mode='rb') as file:
	active_profiles = pickle.load(file)
print('Loading active classes...')
with open(activeclasses_file, mode='rb') as file:
	active_classes = pickle.load(file) 
print('Loading depas...')
with open(depa_file, mode='rb') as file:
	depas = pickle.load(file) 
print('Loading encs...')
with open(enc_file, mode='rb') as file:
	enc = pickle.load(file)

#%%
# Split encounters into a train and test set
print('Splitting encounters into train and test sets...')
full_enc_train, full_enc_test = train_test_split(enc, shuffle=False, test_size=0.25)
enc_train = [full_enc_train[i] for i in sorted(random.sample(range(len(full_enc_train)), 3*(SAMPLE_SIZE//4)))]
enc_test = [full_enc_test[i] for i in sorted(random.sample(range(len(full_enc_test)), SAMPLE_SIZE//4))]

#%%
# Make the data lists

print('Building data lists...')

# Training set
print('Building training set...')
profiles_train = [profiles[enc] for enc in full_enc_train]
targets_train = [target for enc in enc_train for target in targets[enc]]
pre_seq_train = [seq for enc in enc_train for seq in pre_seqs[enc]]
post_seq_train = [seq for enc in enc_train for seq in post_seqs[enc]]
active_profiles_train = [active_profile for enc in enc_train for active_profile in active_profiles[enc]]
active_classes_train = [active_class for enc in enc_train for active_class in active_classes[enc]]
depa_train = [str(depa) for enc in enc_train for depa in depas[enc]]

# Make a list of unique targets in train set to exclude unseen targets from test set
unique_targets_train = list(set(targets_train))

# Test set
print('Building test set...')
targets_test = [target for enc in enc_test for target in targets[enc] if target in unique_targets_train]
pre_seq_test = [seq for enc in enc_test for seq, target in zip(pre_seqs[enc], targets[enc]) if target in unique_targets_train]
post_seq_test = [seq for enc in enc_test for seq, target in zip(post_seqs[enc], targets[enc]) if target in unique_targets_train]
active_profiles_test = [active_profile for enc in enc_test for active_profile, target in zip(active_profiles[enc], targets[enc]) if target in unique_targets_train]
active_classes_test = [active_class for enc in enc_test for active_class, target in zip(active_classes[enc], targets[enc]) if target in unique_targets_train]
depa_test = [str(depa) for enc in enc_test for depa, target in zip(depas[enc], targets[enc]) if target in unique_targets_train]

# Initial shuffle of training set
print('Shuffling training set...')
shuffled = list(zip(targets_train, pre_seq_train, post_seq_train, active_profiles_train, active_classes_train, depa_train))
random.shuffle(shuffled)
targets_train, pre_seq_train, post_seq_train, active_profiles_train, active_classes_train, depa_train = zip(*shuffled)

print('Training set: Obtained {} profiles, {} targets, {} pre sequences, {} post sequences, {} active profiles, {} active classes, {} depas and {} encs.'.format(len(profiles_train), len(targets_train), len(pre_seq_train), len(post_seq_train), len(active_profiles_train), len(active_classes_train), len(depa_train), len(enc_train)))

print('Validation set: Obtained {} targets, {} pre sequences, {} post sequences, {} active profiles, {} active classes, {} depas and {} encs.'.format(len(targets_test), len(pre_seq_test), len(post_seq_test), len(active_profiles_test), len(active_classes_test), len(depa_test), len(enc_test)))

#%% [markdown]
# ## Word2vec embeddings
#
# Create a word2vec pipeline and train word2vec embeddings in that pipeline on the training set profiles

#%%
# Word2vec parameters
W2V_ALPHA = 0.013
W2V_ITER = 32
W2V_EMBEDDING_DIM = 128
W2V_HS = 0
W2V_SG = 0
W2V_MIN_COUNT = 5
W2V_WORKERS = cpu_count()

#%%
# Define and train the pipeline

w2v = Pipeline([
	('w2v', W2VTransformer(alpha=W2V_ALPHA, iter=W2V_ITER, size=W2V_EMBEDDING_DIM, hs=W2V_HS, sg=W2V_SG, min_count=W2V_MIN_COUNT, workers=W2V_WORKERS)),
	])

print('Fitting word2vec embeddings...')
w2v.fit(profiles_train)

# Normalize the embeddings
w2v.named_steps['w2v'].gensim_model.init_sims(replace=True)

#%% [markdown]
# ## Profile state encoder (PSE)
#
# Encode the profile state, either as a multi-hot vector (binary count vectorizer)

#%%
# PSE helper functions

# preprocessor (join the strings with spaces to simulate a text)
def pse_pp(x):
	return ' '.join(x)

# analyzer (do not transform the strings, use them as is)
def pse_a(x):
	return x

#%%
# Prepare the data for the pipeline and fit the PSE encoder to the training set

print('Preparing data for PSE...')
pse_data = [[ap, ac, de] for ap, ac, de in zip(active_profiles_train, active_classes_train, depa_train)]
n_pse_columns = len(pse_data[0])

pse_transformers = []
for i in range(n_pse_columns):
	pse_transformers.append(('pse{}'.format(i), CountVectorizer(lowercase=False, preprocessor=pse_pp, analyzer=pse_a), i))
pse = Pipeline([
	('columntrans', ColumnTransformer(transformers=pse_transformers))
])

print('Fitting PSE...')
pse.fit(pse_data)

#%% [markdown]
# ## Label encoder
#
# Encode the targets

#%%
# Fit the encoder to the train set targets

le = LabelEncoder()
le.fit(targets_train)

#%%
# Define a sequence generator using Keras Sequence to transform the data. Hopefully, in the future, keras-tuner will be able to use a generator instead of transforming the whole dataset in RAM so this code will be more useful. Currently this is overkill, as this is simply used to transform the data using the fitted pipelines so they can be used by the model.

class TransformedGenerator(Sequence):

	def __init__(self, w2v, pse, le, y, X_w2v_pre, X_w2v_post, X_ap, X_ac, X_depa, w2v_embedding_dim, sequence_length, batch_size, shuffle=True, return_targets=True):
		self.w2v = w2v
		self.pse = pse
		self.le = le
		self.y = y
		self.X_w2v_pre = X_w2v_pre
		self.X_w2v_post = X_w2v_post
		self.X_ap = X_ap
		self.X_ac = X_ac
		self.X_depa = X_depa
		self.w2v_embedding_dim = w2v_embedding_dim
		self.sequence_length = sequence_length
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.return_targets = return_targets

	def __len__(self):
		return int(np.ceil(len(self.X_w2v_pre) / float(self.batch_size)))

	def __getitem__(self, idx):
		X = dict()
		batch_w2v_pre = self.X_w2v_pre[idx * self.batch_size:(idx+1) * self.batch_size]
		transformed_w2v_pre = [[self.w2v.gensim_model.wv.get_vector(medic) if medic in self.w2v.gensim_model.wv.index2entity else np.zeros(self.w2v_embedding_dim) for medic in seq] if len(seq) > 0 else [] for seq in batch_w2v_pre]
		transformed_w2v_pre = pad_sequences(transformed_w2v_pre, maxlen=self.sequence_length, dtype='float32')
		X['w2v_pre_input']=transformed_w2v_pre
		batch_w2v_post = self.X_w2v_post[idx * self.batch_size:(idx+1) * self.batch_size]
		transformed_w2v_post = [[self.w2v.gensim_model.wv.get_vector(medic) if medic in self.w2v.gensim_model.wv.index2entity else np.zeros(self.w2v_embedding_dim) for medic in seq] if len(seq) > 0 else [] for seq in batch_w2v_post]
		transformed_w2v_post = pad_sequences(transformed_w2v_post, maxlen=self.sequence_length, dtype='float32', padding='post', truncating='post')
		X['w2v_post_input']=transformed_w2v_post
		batch_ap = self.X_ap[idx * self.batch_size:(idx+1) * self.batch_size]
		batch_ac = self.X_ac[idx * self.batch_size:(idx+1) * self.batch_size]
		batch_depa = self.X_depa[idx * self.batch_size:(idx+1) * self.batch_size]
		batch_pse = [[bp, bc, bd] for bp, bc, bd in zip(batch_ap, batch_ac, batch_depa)]
		transformed_pse = self.pse.transform(batch_pse)
		X['pse_input']=transformed_pse
		X['pse_input']=X['pse_input'].todense()
		if self.return_targets:
			batch_y = self.y[idx * self.batch_size:(idx+1) * self.batch_size]
			transformed_y = self.le.transform(batch_y)
			y = {'main_output': transformed_y}
			return X, y
		else:
			return X
	
	def on_epoch_end(self):
		if self.shuffle == True:
			shuffled = list(zip(self.y, self.X_w2v_pre, self.X_w2v_post, self.X_ap, self.X_ac, self.X_depa))
			random.shuffle(shuffled)
			self.y, self.X_w2v_pre, self.X_w2v_post, self.X_ap, self.X_ac, self.X_depa = zip(*shuffled)

#%%
# Data trasnformation parameters

SEQUENCE_LENGTH = 30

#%% [markdown]
# Transform the data in RAM (use the sequence generator as a simple transformer)

# Build the generators
w2v_step = w2v.named_steps['w2v']

TRAIN_GENERATOR_FAKE_BATCH_SIZE = len(targets_train)
train_generator = TransformedGenerator(w2v_step, pse, le, targets_train, pre_seq_train, post_seq_train, active_profiles_train, active_classes_train, depa_train, W2V_EMBEDDING_DIM, SEQUENCE_LENGTH, TRAIN_GENERATOR_FAKE_BATCH_SIZE)

TEST_GENERATOR_FAKE_BATCH_SIZE = len(targets_test)
test_generator = TransformedGenerator(w2v_step, pse, le, targets_test, pre_seq_test, post_seq_test, active_profiles_test, active_classes_test, depa_test, W2V_EMBEDDING_DIM, SEQUENCE_LENGTH, TEST_GENERATOR_FAKE_BATCH_SIZE, shuffle=False)

print('Transforming train data...')
transformed_train = train_generator.__getitem__(0)
print('Transforming test data...')
transformed_test = test_generator.__getitem__(0)

#%% [markdown]
# ## Neural network
#
# Train a neural network to predict each drug present in a pharmacological profile from the sequence of drug orders that came before or after it and the profile state excluding that drug.

#%%
# Additional accuracy metrics. These metrics will be used to compute top 10 and top 30 accuracy during training because for this problem top 1 accuracy is going to be relatively low (because of random ordering of prescription sequence and order entry)

def sparse_top10_accuracy(y_true, y_pred):
	return (sparse_top_k_categorical_accuracy(y_true, y_pred, k=10))

def sparse_top30_accuracy(y_true, y_pred):
	return (sparse_top_k_categorical_accuracy(y_true, y_pred, k=30))

#%%
# Training variables
TSVD_N_COMPONENTS = sum([len(transformer[1].vocabulary_) for transformer in pse.named_steps['columntrans'].transformers_])
OUTPUT_N_CLASSES = len(le.classes_)

#%%
# Define the model. Built as a separate function to make code clearer.

def define_model(hp):

	LSTM_SIZE = hp.Choice('lstm_size', values=[32,128])
	DENSE_PSE_SIZE = hp.Choice('dense_pse_size', values=[32,128])
	CONCAT_LSTM_SIZE = hp.Choice('concat_lstm_size', values=[32,128])
	DENSE_SIZE = hp.Choice('dense_size', values=[32,128])
	DROPOUT = hp.Choice('dropout', values=[0,0.2,0.5])
	LEARNING_RATE = hp.Choice('learning_rate', values=[1e-3])

	to_concat_lstm = []
	to_concat = []
	inputs= []
	
	w2v_pre_input = Input(shape=(SEQUENCE_LENGTH, W2V_EMBEDDING_DIM, ), dtype='float32', name='w2v_pre_input')
	w2v_pre = Masking()(w2v_pre_input)
	w2v_pre = RNN(LSTMCell(LSTM_SIZE), return_sequences=True)(w2v_pre)
	w2v_pre = Dropout(DROPOUT)(w2v_pre)
	w2v_pre = RNN(LSTMCell(LSTM_SIZE))(w2v_pre)
	w2v_pre = Dropout(DROPOUT)(w2v_pre)
	w2v_pre = Dense(LSTM_SIZE, activation='relu')(w2v_pre)
	w2v_pre = Dropout(DROPOUT)(w2v_pre)
	to_concat_lstm.append(w2v_pre)
	inputs.append(w2v_pre_input)

	w2v_post_input = Input(shape=(SEQUENCE_LENGTH, W2V_EMBEDDING_DIM, ), dtype='float32', name='w2v_post_input')
	w2v_post = Masking()(w2v_post_input)
	w2v_post = RNN(LSTMCell(LSTM_SIZE), return_sequences=True, go_backwards=True)(w2v_post)
	w2v_post = Dropout(DROPOUT)(w2v_post)
	w2v_post = RNN(LSTMCell(LSTM_SIZE))(w2v_post)
	w2v_post = Dropout(DROPOUT)(w2v_post)
	w2v_post = Dense(LSTM_SIZE, activation='relu')(w2v_post)
	w2v_post = Dropout(DROPOUT)(w2v_post)
	to_concat_lstm.append(w2v_post)
	inputs.append(w2v_post_input)

	concatenated_lstm = concatenate(to_concat_lstm)
	concatenated_lstm = BatchNormalization()(concatenated_lstm)
	concatenated_lstm = Dense(CONCAT_LSTM_SIZE, activation='relu')(concatenated_lstm)
	concatenated_lstm = Dropout(DROPOUT)(concatenated_lstm)
	to_concat.append(concatenated_lstm)
	
	pse_input = Input(shape=(TSVD_N_COMPONENTS,), dtype='float32', name='pse_input')
	pse = Dense(DENSE_PSE_SIZE, activation='relu')(pse_input)
	pse = Dropout(DROPOUT)(pse)
	to_concat.append(pse)
	inputs.append(pse_input)

	concatenated = concatenate(to_concat)
	concatenated = BatchNormalization()(concatenated)
	concatenated = Dense(DENSE_SIZE, activation='relu')(concatenated)
	concatenated = Dropout(DROPOUT)(concatenated)
	concatenated = BatchNormalization()(concatenated)
	concatenated = Dense(DENSE_SIZE, activation='relu')(concatenated)
	concatenated = Dropout(DROPOUT)(concatenated)
	concatenated = BatchNormalization()(concatenated)
	output = Dense(OUTPUT_N_CLASSES, activation='softmax', name='main_output')(concatenated)

	model = Model(inputs = inputs, outputs = output)
	model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy', sparse_top10_accuracy, sparse_top30_accuracy])
	
	return model

#%%
# Prepare the tuner

tuner = RandomSearch(
	define_model,
	objective='val_loss',
	max_trials=2,
	executions_per_trial=1,
	project_name='hp_tuning')

tuner.search_space_summary()

#%%
# Define callbacks

rlr_callback = ReduceLROnPlateau(monitor='val_loss', patience=3, min_delta=0.0005)
earlystop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, verbose=1, restore_best_weights=True)

#%%
# Search
tuner.search(transformed_train[0], transformed_train[1],
	epochs=1000,
	batch_size=32,
	verbose=2,
	callbacks=[rlr_callback, earlystop_callback],
	validation_data=(transformed_test[0], transformed_test[1]))

#%%
# Print the results
tuner.results_summary()

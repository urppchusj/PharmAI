import os
import pathlib
import pickle
import random

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras, test


class check_ipynb:

	def __init__(self):
		pass

	def is_inipynb(self):
		try:
			get_ipython()
			print('Execution in Jupyter Notebook detected.')
			return True
		except:
			print('Execution outside of Jupyter Notebook detected.')
			return False


class data:

	def __init__(self, datadir):
		self.definitions_file = os.path.join(os.getcwd(), 'data', 'definitions.csv')
		self.profiles_file = os.path.join(os.getcwd(), 'preprocessed_data', 'whole_profile',datadir, 'profiles_list.pkl')
		self.targets_file = os.path.join(os.getcwd(), 'preprocessed_data', 'whole_profile', datadir, 'targets_list.pkl')
		self.pre_seq_file = os.path.join(os.getcwd(), 'preprocessed_data', 'whole_profile', datadir, 'pre_seq_list.pkl')
		self.post_seq_file = os.path.join(os.getcwd(), 'preprocessed_data', 'whole_profile', datadir, 'post_seq_list.pkl')
		self.activeprofiles_file = os.path.join(os.getcwd(), 'preprocessed_data', 'whole_profile', datadir, 'active_profiles_list.pkl')
		self.activeclasses_file = os.path.join(os.getcwd(), 'preprocessed_data', 'whole_profile', datadir, 'active_classes_list.pkl')
		self.depa_file = os.path.join(os.getcwd(), 'preprocessed_data', 'whole_profile', datadir, 'depa_list.pkl')
		self.enc_file = os.path.join(os.getcwd(), 'preprocessed_data', 'whole_profile', datadir, 'enc_list.pkl')

	def load_data(self, restrict_data=False, restrict_sample_size=None, previous_encs_path=False, get_profiles=True, get_definitions=True):
		
		if get_profiles:
			print('Loading profiles...')
			with open(self.profiles_file, mode='rb') as file:
				self.profiles = pickle.load(file)
		else:
			self.profiles=None
		print('Loading targets...')
		with open(self.targets_file, mode='rb') as file:
			self.targets = pickle.load(file)
		print('Loading pre sequences...')
		with open(self.pre_seq_file, mode='rb') as file:
			self.pre_seqs = pickle.load(file)
		print('Loading post sequences...')
		with open(self.post_seq_file, mode='rb') as file:
			self.post_seqs = pickle.load(file)
		print('Loading active profiles...')
		with open(self.activeprofiles_file, mode='rb') as file:
			self.active_profiles = pickle.load(file)
		print('Loading active classes...')
		with open(self.activeclasses_file, mode='rb') as file:
			self.active_classes = pickle.load(file) 
		print('Loading depas...')
		with open(self.depa_file, mode='rb') as file:
			self.depas = pickle.load(file) 
		print('Loading encs...')
		if previous_encs_path:
			self.enc_file = previous_encs_path
		with open(self.enc_file, mode='rb') as file:
				self.enc = pickle.load(file)
		if restrict_data:
			print('Data restriction flag enabled, sampling {} encounters...'.format(restrict_sample_size))
			self.enc = [self.enc[i] for i in sorted(random.sample(range(len(self.enc)), restrict_sample_size))]

		if get_definitions:
			print('Loading definitions...')
			definitions_col_names = ['medinb', 'mediname', 'genenb', 'genename', 'classnb', 'classname']
			definitions_dtypes = {'medinb':str, 'mediname':str, 'genenb':str, 'genename':str, 'classnb':str, 'classename':str}
			classes_data = pd.read_csv(self.definitions_file, sep=';', names=definitions_col_names, dtype=definitions_dtypes)
			self.definitions = dict(zip(list(classes_data.medinb), list(classes_data.mediname)))
		else:
			self.definitions=None
		
	def split(self):
		print('Splitting encounters into train and val sets...')
		self.enc_train, self.enc_val = train_test_split(self.enc, shuffle=False, test_size=0.25)

	def make_lists(self, get_val=True):
		print('Building data lists...')

		# Training set
		print('Building training set...')
		if get_val == False:
			self.enc_train = self.enc
		if self.profiles != None:
			self.profiles_train = [self.profiles[enc] for enc in self.enc_train]
		else:
			self.profiles_train=[]
		self.targets_train = [target for enc in self.enc_train for target in self.targets[enc]]
		self.pre_seq_train = [seq for enc in self.enc_train for seq in self.pre_seqs[enc]]
		self.post_seq_train = [seq for enc in self.enc_train for seq in self.post_seqs[enc]]
		self.active_profiles_train = [active_profile for enc in self.enc_train for active_profile in self.active_profiles[enc]]
		self.active_classes_train = [active_class for enc in self.enc_train for active_class in self.active_classes[enc]]
		self.depa_train = [str(depa) for enc in self.enc_train for depa in self.depas[enc]]

		# Make a list of unique targets in train set to exclude unseen targets from validation set
		unique_targets_train = list(set(self.targets_train))

		# Validation set
		if get_val:
			print('Building validation set...')
			self.targets_val = [target for enc in self.enc_val for target in self.targets[enc] if target in unique_targets_train]
			self.pre_seq_val = [seq for enc in self.enc_val for seq, target in zip(self.pre_seqs[enc], self.targets[enc]) if target in unique_targets_train]
			self.post_seq_val = [seq for enc in self.enc_val for seq, target in zip(self.post_seqs[enc], self.targets[enc]) if target in unique_targets_train]
			self.active_profiles_val = [active_profile for enc in self.enc_val for active_profile, target in zip(self.active_profiles[enc], self.targets[enc]) if target in unique_targets_train]
			self.active_classes_val = [active_class for enc in self.enc_val for active_class, target in zip(self.active_classes[enc], self.targets[enc]) if target in unique_targets_train]
			self.depa_val = [str(depa) for enc in self.enc_val for depa, target in zip(self.depas[enc], self.targets[enc]) if target in unique_targets_train]
		else:
			self.targets_val = None
			self.pre_seq_val = None
			self.post_seq_val = None
			self.active_profiles_val = None
			self.active_classes_val = None
			self.depa_val = None
		
		# Initial shuffle of training set
		print('Shuffling training set...')
		shuffled = list(zip(self.targets_train, self.pre_seq_train, self.post_seq_train, self.active_profiles_train, self.active_classes_train, self.depa_train))
		random.shuffle(shuffled)
		self.targets_train, self.pre_seq_train, self.post_seq_train, self.active_profiles_train, self.active_classes_train, self.depa_train = zip(*shuffled)

		print('Training set: Obtained {} profiles, {} targets, {} pre sequences, {} post sequences, {} active profiles, {} active classes, {} depas and {} encs.'.format(len(self.profiles_train), len(self.targets_train), len(self.pre_seq_train), len(self.post_seq_train), len(self.active_profiles_train), len(self.active_classes_train), len(self.depa_train), len(self.enc_train)))

		if get_val == True:
			print('Validation set: Obtained {} targets, {} pre sequences, {} post sequences, {} active profiles, {} active classes, {} depas and {} encs.'.format(len(self.targets_val), len(self.pre_seq_val), len(self.post_seq_val), len(self.active_profiles_val), len(self.active_classes_val), len(self.depa_val), len(self.enc_val)))

		return self.profiles_train, self.targets_train, self.pre_seq_train, self.post_seq_train, self.active_profiles_train, self.active_classes_train, self.depa_train, self.targets_val, self.pre_seq_val, self.post_seq_val, self.active_profiles_val, self.active_classes_val, self.depa_val, self.definitions


class pse_helper_functions:

	def __init__(self):
		pass

	# preprocessor (join the strings with spaces to simulate a text)
	def pse_pp(self, x):
		return ' '.join(x)

	# analyzer (do not transform the strings, use them as is)
	def pse_a(self, x):
		return x


class TransformedGenerator(keras.utils.Sequence):

	def __init__(self, w2v, use_lsi, pse, le, y, X_w2v_pre, X_w2v_post, X_ap, X_ac, X_depa, w2v_embedding_dim, sequence_length, batch_size, shuffle=True, return_targets=True):
		self.w2v = w2v
		self.use_lsi = use_lsi
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
		transformed_w2v_pre = keras.preprocessing.sequence.pad_sequences(transformed_w2v_pre, maxlen=self.sequence_length, dtype='float32')
		X['w2v_pre_input']=transformed_w2v_pre
		batch_w2v_post = self.X_w2v_post[idx * self.batch_size:(idx+1) * self.batch_size]
		transformed_w2v_post = [[self.w2v.gensim_model.wv.get_vector(medic) if medic in self.w2v.gensim_model.wv.index2entity else np.zeros(self.w2v_embedding_dim) for medic in seq] if len(seq) > 0 else [] for seq in batch_w2v_post]
		transformed_w2v_post = keras.preprocessing.sequence.pad_sequences(transformed_w2v_post, maxlen=self.sequence_length, dtype='float32', padding='post', truncating='post')
		X['w2v_post_input']=transformed_w2v_post
		batch_ap = self.X_ap[idx * self.batch_size:(idx+1) * self.batch_size]
		batch_ac = self.X_ac[idx * self.batch_size:(idx+1) * self.batch_size]
		batch_depa = self.X_depa[idx * self.batch_size:(idx+1) * self.batch_size]
		batch_pse = [[bp, bc, bd] for bp, bc, bd in zip(batch_ap, batch_ac, batch_depa)]
		transformed_pse = self.pse.transform(batch_pse)
		X['pse_input']=transformed_pse
		if self.use_lsi == False:
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


class neural_network:

	def __init__(self):
		pass

	def sparse_top10_accuracy(self, y_true, y_pred):
		sparse_top_k_categorical_accuracy = keras.metrics.sparse_top_k_categorical_accuracy
		return (sparse_top_k_categorical_accuracy(y_true, y_pred, k=10))

	def sparse_top30_accuracy(self, y_true, y_pred):
		sparse_top_k_categorical_accuracy = keras.metrics.sparse_top_k_categorical_accuracy
		return (sparse_top_k_categorical_accuracy(y_true, y_pred, k=30))

	def callbacks(self, save_path):

		# assign simple names
		CSVLogger = keras.callbacks.CSVLogger
		EarlyStopping = keras.callbacks.EarlyStopping
		ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau
		ModelCheckpoint = keras.callbacks.ModelCheckpoint

		rlr_callback = ReduceLROnPlateau(monitor='val_loss', patience=3, min_delta=0.0005)
		earlystop_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, restore_best_weights=True)
		mc_callback = ModelCheckpoint(os.path.join(save_path, 'partially_trained_model.h5'), verbose=1)
		csv_callback = CSVLogger(os.path.join(save_path, 'training_history.csv'), append=True)

		return [rlr_callback, earlystop_callback, mc_callback, csv_callback]

	def define_model(self, sequence_size, n_add_seq_layers, dense_pse_size, concat_sequence_size, concat_total_size, dense_size, dropout, l2_reg, sequence_length, w2v_embedding_dim, pse_shape, n_add_pse_dense, n_dense, output_n_classes):
		
		# assign simple names
		if test.is_gpu_available():
			LSTM = keras.layers.CuDNNLSTM
		else:
			LSTM = keras.layers.LSTM

		Dense = keras.layers.Dense
		Dropout = keras.layers.Dropout
		Input = keras.layers.Input
		BatchNormalization = keras.layers.BatchNormalization
		concatenate = keras.layers.concatenate
		l2 = keras.regularizers.l2
		Model = keras.models.Model

		to_concat_sequence = []
		to_concat = []
		inputs= []
		
		w2v_pre_input = Input(shape=(sequence_length, w2v_embedding_dim, ), dtype='float32', name='w2v_pre_input')
		w2v_pre = LSTM(sequence_size, return_sequences=True)(w2v_pre_input)
		w2v_pre = Dropout(dropout)(w2v_pre)
		for _ in range(n_add_seq_layers):
			w2v_pre = LSTM(sequence_size, return_sequences=True)(w2v_pre)
			w2v_pre = Dropout(dropout)(w2v_pre)
		w2v_pre = LSTM(sequence_size)(w2v_pre)
		w2v_pre = Dropout(dropout)(w2v_pre)
		w2v_pre = Dense(sequence_size, activation='relu')(w2v_pre)
		w2v_pre = Dropout(dropout)(w2v_pre)
		to_concat_sequence.append(w2v_pre)
		inputs.append(w2v_pre_input)

		w2v_post_input = Input(shape=(sequence_length, w2v_embedding_dim, ), dtype='float32', name='w2v_post_input')
		w2v_post = LSTM(sequence_size, return_sequences=True, go_backwards=True)(w2v_post_input)
		w2v_post = Dropout(dropout)(w2v_post)
		for _ in range(n_add_seq_layers):
			w2v_post = LSTM(sequence_size, return_sequences=True)(w2v_post)
			w2v_post = Dropout(dropout)(w2v_post)
		w2v_post = LSTM(sequence_size)(w2v_post)
		w2v_post = Dropout(dropout)(w2v_post)
		w2v_post = Dense(sequence_size, activation='relu', kernel_regularizer=l2(l2_reg))(w2v_post)
		w2v_post = Dropout(dropout)(w2v_post)
		to_concat_sequence.append(w2v_post)
		inputs.append(w2v_post_input)

		concatenated_sequence = concatenate(to_concat_sequence)
		concatenated_sequence = BatchNormalization()(concatenated_sequence)
		concatenated_sequence = Dense(concat_sequence_size, activation='relu', kernel_regularizer=l2(l2_reg))(concatenated_sequence)
		concatenated_sequence = Dropout(dropout)(concatenated_sequence)
		to_concat.append(concatenated_sequence)
		
		pse_input = Input(shape=(pse_shape,), dtype='float32', name='pse_input')
		pse = Dense(dense_pse_size, activation='relu', kernel_regularizer=l2(l2_reg))(pse_input)
		pse = Dropout(dropout)(pse)
		for _ in range(n_add_pse_dense):
			pse = BatchNormalization()(pse)
			pse = Dense(dense_pse_size, activation='relu', kernel_regularizer=l2(l2_reg))(pse)
			pse = Dropout(dropout)(pse)
		to_concat.append(pse)
		inputs.append(pse_input)

		concatenated = concatenate(to_concat)
		for _ in range(n_dense):
			concatenated = BatchNormalization()(concatenated)
			concatenated = Dense(concat_total_size, activation='relu', kernel_regularizer=l2(l2_reg))(concatenated)
			concatenated = Dropout(dropout)(concatenated)
		concatenated = BatchNormalization()(concatenated)
		output = Dense(output_n_classes, activation='softmax', name='main_output')(concatenated)

		model = Model(inputs = inputs, outputs = output)
		model.compile(optimizer='Adam', loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy', self.sparse_top10_accuracy, self.sparse_top30_accuracy])
		print(model.summary())
		
		return model


class visualization:

	def __init__(self):
		self.in_ipynb = check_ipynb().is_inipynb()

	def plot_accuracy_history(self, df, save_path):
		acc_df = df[['sparse_top10_accuracy', 'val_sparse_top10_accuracy', 'sparse_top30_accuracy', 'val_sparse_top30_accuracy', 'sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].copy()
		acc_df.rename(inplace=True, index=str, columns={'sparse_top30_accuracy':'Train top 30 accuracy', 'val_sparse_top30_accuracy':'Val top 30 accuracy', 'sparse_top10_accuracy':'Train top 10 accuracy', 'val_sparse_top10_accuracy':'Val top 10 accuracy', 'sparse_categorical_accuracy':'Train top 1 accuracy', 'val_sparse_categorical_accuracy':'Val top 1 accuracy'})
		acc_df = acc_df.stack().reset_index()
		acc_df.rename(inplace=True, index=str, columns={'level_0':'Epoch', 'level_1':'Metric', 0:'Result'})
		acc_df['Epoch'] = acc_df['Epoch'].astype('int8')
		sns.set(style='darkgrid')
		sns.relplot(x='Epoch', y='Result', hue='Metric', kind='line', data=acc_df)
		if self.in_ipynb:
			plt.show()
		else:
			plt.savefig(os.path.join(save_path, 'acc_history.png'))
		plt.gcf().clear()

	def plot_loss_history(self,df, save_path):
		loss_df = df[['loss', 'val_loss']].copy()
		loss_df.rename(inplace=True, index=str, columns={'loss':'Train loss', 'val_loss':'Val loss'})
		loss_df = loss_df.stack().reset_index()
		loss_df.rename(inplace=True, index=str, columns={'level_0':'Epoch', 'level_1':'Metric', 0:'Result'})
		loss_df['Epoch'] = loss_df['Epoch'].astype('int8')
		sns.set(style='darkgrid')
		sns.relplot(x='Epoch', y='Result', hue='Metric', kind='line', data=loss_df)
		if self.in_ipynb:
			plt.show()
		else:
			plt.savefig(os.path.join(save_path, 'loss_history.png'))
		plt.gcf().clear()

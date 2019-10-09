import os
import pickle
import random

import gensim.utils as gsu
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from gensim.matutils import Sparse2Corpus
from gensim.models import KeyedVectors
from gensim.sklearn_api import LsiTransformer, W2VTransformer
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import (ShuffleSplit, TimeSeriesSplit,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, MultiLabelBinarizer
from tensorflow import keras, test, math, constant, dtypes, float32, metrics


class check_ipynb:

    '''
    Verifies if current execution is in a Jupyter Notebook.
    Prints result and returns True if in Jupyter Notebook, else false.
    '''

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

    '''
    Functions related to data loading and preparation.
    '''

    def __init__(self, datadir, mode, keep_time_order=True):
        # Prepare the data paths given a directory.
        self.mode = mode
        self.keep_time_order = keep_time_order
        # File names correspond to what is created by the preprocessor.
        # Definitions file allows matching drug ids to drug names
        self.definitions_file = os.path.join(
            os.getcwd(), 'data', 'definitions.csv')
        self.profiles_file = os.path.join(
            os.getcwd(), 'preprocessed_data', self.mode, datadir, 'profiles_list.pkl')
        self.targets_file = os.path.join(
            os.getcwd(), 'preprocessed_data', self.mode, datadir, 'targets_list.pkl')
        self.pre_seq_file = os.path.join(
            os.getcwd(), 'preprocessed_data', self.mode, datadir, 'pre_seq_list.pkl')
        self.post_seq_file = os.path.join(
            os.getcwd(), 'preprocessed_data', self.mode, datadir, 'post_seq_list.pkl')
        self.activemeds_file = os.path.join(os.getcwd(
        ), 'preprocessed_data', self.mode, datadir, 'active_meds_list.pkl')
        self.activeclasses_file = os.path.join(os.getcwd(
        ), 'preprocessed_data', self.mode, datadir, 'active_classes_list.pkl')
        self.depa_file = os.path.join(
            os.getcwd(), 'preprocessed_data', self.mode, datadir, 'depa_list.pkl')
        self.enc_file = os.path.join(
            os.getcwd(), 'preprocessed_data', self.mode, datadir, 'enc_list.pkl')

    def load_data(self, restrict_data=False, save_path=None, restrict_sample_size=None, previous_encs_path=False, get_profiles=True, get_definitions=True):

        # This allows to prevent loading profiles for nothing when
        # resuming training and word2vec embeddings do not need
        # to be retrained.
        if get_profiles:
            print('Loading profiles...')
            with open(self.profiles_file, mode='rb') as file:
                self.profiles = pickle.load(file)
        else:
            self.profiles = None
        # Load all the files
        if self.mode == 'retrospective-autoenc':
            self.targets = None
        else:
            print('Loading targets...')
            with open(self.targets_file, mode='rb') as file:
                self.targets = pickle.load(file)
        print('Loading pre sequences...')
        with open(self.pre_seq_file, mode='rb') as file:
            self.pre_seqs = pickle.load(file)
        print('Loading post sequences...')
        with open(self.post_seq_file, mode='rb') as file:
            self.post_seqs = pickle.load(file)
        print('Loading active meds...')
        with open(self.activemeds_file, mode='rb') as file:
            self.active_meds = pickle.load(file)
        print('Loading active classes...')
        try:
            with open(self.activeclasses_file, mode='rb') as file:
                self.active_classes = pickle.load(file)
            self.use_classes = True
        except:
            self.use_classes = False
        print('Loading departments...')
        with open(self.depa_file, mode='rb') as file:
            self.depas = pickle.load(file)
        print('Loading encounters...')
        # If reloading in order to resume training, this allows to reload
        # the same set of encounters when encounters have been sampled
        # using the restrict_data flag, to resume training with exactly
        # the same.
        if previous_encs_path:
            self.enc_file = previous_encs_path
        with open(self.enc_file, mode='rb') as file:
            self.enc = pickle.load(file)
        # The restrict_data flags allows to sample a number of encounters
        # defined by restrict_sample_size, to allow for faster execution
        # when testing code.
        if restrict_data:
            print('Data restriction flag enabled, sampling {} encounters...'.format(
                restrict_sample_size))
            self.enc = [self.enc[i] for i in sorted(
                random.sample(range(len(self.enc)), restrict_sample_size))]
            with open(os.path.join(save_path, 'sampled_encs.pkl'), mode='wb') as file:
                pickle.dump(self.enc, file)

        if get_definitions:
            # Build a dict mapping drug ids to their names
            print('Loading definitions...')
            definitions_col_names = ['medinb', 'mediname',
                                     'genenb', 'genename', 'classnb', 'classname']
            definitions_dtypes = {'medinb': str, 'mediname': str,
                                  'genenb': str, 'genename': str, 'classnb': str, 'classename': str}
            classes_data = pd.read_csv(
                self.definitions_file, sep=';', names=definitions_col_names, dtype=definitions_dtypes)
            self.definitions = dict(
                zip(list(classes_data.medinb), list(classes_data.mediname)))
        else:
            self.definitions = None

    def split(self, split_seed=None):
        print('Splitting encounters into train and val sets...')
        self.enc_train, self.enc_val = train_test_split(
            self.enc, shuffle=not self.keep_time_order, random_state=split_seed, test_size=0.25)

    def cross_val_split(self, n_folds, split_seed=None):
        print('Splitting encounters into train and validation sets for {} cross-validation folds...'.format(n_folds))
        self.enc_train_list = []
        self.enc_val_list = []
        if self.keep_time_order:
            splitter = TimeSeriesSplit(n_splits=n_folds)
        else:
            splitter = ShuffleSplit(n_splits=n_folds, random_state=split_seed)
        for train_indices, val_indices in splitter.split(self.enc):
            self.enc_train_list.append([self.enc[i] for i in train_indices])
            self.enc_val_list.append([self.enc[i] for i in val_indices])

    def make_lists(self, get_valid=True, cross_val_fold=None, shuffle_train_set=True):
        print('Building data lists...')

        # If building lists in cross-validation (cross_val_fold > 0),
        # use the encounters in the cross_val lists instead of the
        # entire encounter list
        if cross_val_fold is not None:
            self.enc_train = self.enc_train_list[cross_val_fold]
            self.enc_val = self.enc_val_list[cross_val_fold]

        # Training set
        print('Building training set...')
        # If the get_valid flag is set to False, put all encounters
        # in the training set. This can be used for evaluation
        # of a trained model, in this case the "training" set is
        # actually an evaluation set that does not get split.
        if get_valid == False:
            self.enc_train = self.enc
        # Allocate profiles only if they have been loaded
        if self.profiles != None:
            self.profiles_train = [self.profiles[enc]
                                   for enc in self.enc_train]
        else:
            self.profiles_train = []
        if self.targets != None:
            self.targets_train = [
                target for enc in self.enc_train for target in self.targets[enc]]
        else:
            self.targets_train = []
        self.pre_seq_train = [
            seq for enc in self.enc_train for seq in self.pre_seqs[enc]]
        self.post_seq_train = [
            seq for enc in self.enc_train for seq in self.post_seqs[enc]]
        self.active_meds_train = [
            active_med for enc in self.enc_train for active_med in self.active_meds[enc]]
        if self.use_classes:
            self.active_classes_train = [
                active_class for enc in self.enc_train for active_class in self.active_classes[enc]]
        else:
            self.active_classes_train = []
        self.depa_train = [[str(dep) for dep in depa]
                           for enc in self.enc_train for depa in self.depas[enc]]

        # Validation set is built only if necessary
        if get_valid:
            if len(self.targets_train) > 0 :
                # Make a list of unique targets in train set to exclude unseen targets from validation set
                unique_targets_train = list(set(self.targets_train))
                print('Building validation set...')
                # Filter out samples with previously unseen labels.
                self.targets_val = [
                    target for enc in self.enc_val for target in self.targets[enc] if target in unique_targets_train]
                self.pre_seq_val = [seq for enc in self.enc_val for seq, target in zip(
                    self.pre_seqs[enc], self.targets[enc]) if target in unique_targets_train]
                self.post_seq_val = [seq for enc in self.enc_val for seq, target in zip(
                    self.post_seqs[enc], self.targets[enc]) if target in unique_targets_train]
                self.active_meds_val = [active_med for enc in self.enc_val for active_med, target in zip(
                    self.active_meds[enc], self.targets[enc]) if target in unique_targets_train]
                if self.use_classes:
                    self.active_classes_val = [active_class for enc in self.enc_val for active_class, target in zip(
                        self.active_classes[enc], self.targets[enc]) if target in unique_targets_train]
                else:
                    self.active_classes_val = []
                self.depa_val = [[str(dep) for dep in depa] for enc in self.enc_val for depa, target in zip(
                    self.depas[enc], self.targets[enc]) if target in unique_targets_train]
            else:
                print('Building validation set...')
                self.targets_val = []
                self.pre_seq_val = [seq for enc in self.enc_val for seq in self.pre_seqs[enc]]
                self.post_seq_val = [seq for enc in self.enc_val for seq in self.post_seqs[enc]]
                self.active_meds_val = [active_med for enc in self.enc_val for active_med in self.active_meds[enc]]
                if self.use_classes:
                    self.active_classes_val = [active_class for enc in self.enc_val for active_class in self.active_classes[enc]]
                else:
                    self.active_classes_val = []
                self.depa_val = [[str(dep) for dep in depa] for enc in self.enc_val for depa in self.depas[enc]]
        else:
            self.targets_val = None
            self.pre_seq_val = None
            self.post_seq_val = None
            self.active_meds_val = None
            self.active_classes_val = None
            self.depa_val = None
        

        if shuffle_train_set:
            # Initial shuffle of training set
            print('Shuffling training set...')
            if self.use_classes:
                if self.mode == 'prospective':
                    shuffled = list(zip(self.targets_train, self.pre_seq_train,
                                        self.active_meds_train, self.active_classes_train, self.depa_train))
                    random.shuffle(shuffled)
                    self.targets_train, self.pre_seq_train, self.active_meds_train, self.active_classes_train, self.depa_train = zip(*shuffled)
                elif self.mode == 'retrospective':
                    shuffled = list(zip(self.targets_train, self.pre_seq_train, self.post_seq_train,
                                        self.active_meds_train, self.active_classes_train, self.depa_train))
                    random.shuffle(shuffled)
                    self.targets_train, self.pre_seq_train, self.post_seq_train, self.active_meds_train, self.active_classes_train, self.depa_train = zip(*shuffled)
                elif self.mode == 'retrospective-autoenc':
                    shuffled = list(zip(self.pre_seq_train, self.active_meds_train, self.active_classes_train, self.depa_train))
                    random.shuffle(shuffled)
                    self.pre_seq_train, self.active_meds_train, self.active_classes_train, self.depa_train = zip(*shuffled)
            else:
                if self.mode == 'prospective':
                    shuffled = list(zip(self.targets_train, self.pre_seq_train,
                                        self.active_meds_train, self.depa_train))
                    random.shuffle(shuffled)
                    self.targets_train, self.pre_seq_train, self.active_meds_train, self.depa_train = zip(*shuffled)
                elif self.mode == 'retrospective':
                    shuffled = list(zip(self.targets_train, self.pre_seq_train, self.post_seq_train,
                                        self.active_meds_train, self.depa_train))
                    random.shuffle(shuffled)
                    self.targets_train, self.pre_seq_train, self.post_seq_train, self.active_meds_train, self.depa_train = zip(*shuffled)
                elif self.mode == 'retrospective-autoenc':
                    shuffled = list(zip(self.pre_seq_train, self.active_meds_train, self.depa_train))
                    random.shuffle(shuffled)
                    self.pre_seq_train, self.active_meds_train, self.depa_train = zip(*shuffled)

       # Print out the number of samples obtained to make sure they match.
        print('Training set: Obtained {} profiles, {} targets, {} pre sequences, {} post sequences, {} active meds, {} active classes, {} depas and {} encs.'.format(len(self.profiles_train), len(
            self.targets_train), len(self.pre_seq_train), len(self.post_seq_train), len(self.active_meds_train), len(self.active_classes_train), len(self.depa_train), len(self.enc_train)))

        if get_valid == True:
            print('Validation set: Obtained {} targets, {} pre sequences, {} post sequences, {} active meds, {} active classes, {} depas and {} encs.'.format(len(
                self.targets_val), len(self.pre_seq_val), len(self.post_seq_val), len(self.active_meds_val), len(self.active_classes_val), len(self.depa_val), len(self.enc_val)))

        return self.profiles_train, self.targets_train, self.pre_seq_train, self.post_seq_train, self.active_meds_train, self.active_classes_train, self.depa_train, self.targets_val, self.pre_seq_val, self.post_seq_val, self.active_meds_val, self.active_classes_val, self.depa_val, self.definitions


class transformation_pipelines:

    '''
    Scikit-learn data transformation pipelines that take the preprocessed
    data as inputs and transform them into appropriate representations
    for the neural network
    '''

    def __init__(self):
        pass

    # Define the word2vec pipeline
    def define_w2v_pipeline(self, alpha, iter, embedding_dim, hs, sg, min_count, workers):
        self.w2v = Pipeline([
            ('w2v', W2VTransformer(alpha=alpha, iter=iter, size=embedding_dim,
                                   hs=hs, sg=sg, min_count=min_count, workers=workers)),
        ])

    # Fit the pipeline, normalize the embeddings and save the
    # pipeline as well as required values to proprely load and use.
    def fitsave_w2v_pipeline(self, save_path, profiles_train, n_fold):
        print('Fitting word2vec embeddings...')
        self.w2v.fit(profiles_train)
        # Normalize the embeddings
        self.w2v.named_steps['w2v'].gensim_model.init_sims(replace=True)
        # save the fitted word2vec pipe
        joblib.dump((n_fold, self.w2v),
                    os.path.join(save_path, 'w2v.joblib'))
        return self.w2v

    # Export the word2vec embeddings with associated metadata to
    # visualize or perform other tasks with the embeddings alone.
    def export_w2v_embeddings(self, save_path, definitions_dict=None):
        print('Exporting word2vec embeddings...')
        self.w2v.named_steps['w2v'].gensim_model.wv.save_word2vec_format(
            os.path.join(save_path, 'w2v.model'))
        model = KeyedVectors.load_word2vec_format(
            os.path.join(save_path, 'w2v.model'), binary=False)
        outfiletsv = os.path.join(save_path, 'w2v_embeddings.tsv')
        outfiletsvmeta = os.path.join(save_path, 'w2v_metadata.tsv')

        with open(outfiletsv, 'w+') as file_vector:
            with open(outfiletsvmeta, 'w+') as file_metadata:
                for word in model.index2word:
                    file_metadata.write(gsu.to_utf8(word).decode(
                        'utf-8') + gsu.to_utf8('\n').decode('utf-8'))
                    vector_row = '\t'.join(str(x) for x in model[word])
                    file_vector.write(vector_row + '\n')

        print("2D tensor file saved to %s", outfiletsv)
        print("Tensor metadata file saved to %s", outfiletsvmeta)

        with open(outfiletsvmeta, mode='r', encoding='utf-8', errors='strict') as metadata_file:
            metadata = metadata_file.read()
        converted_string = ''
        for element in metadata.splitlines():
            string = element.strip()
            converted_string += definitions_dict[string] + '\n'
        with open(os.path.join(save_path, 'w2v_defined_metadata.tsv'), mode='w', encoding='utf-8', errors='strict') as converted_metadata:
            converted_metadata.write(converted_string)

    # Transform the profile state lists (active meds, active classes and
    # ordering department) into lists for input into the scikit-learn
    # column transformer
    def prepare_pse_data(self, active_meds, active_classes, departments):
        print('Preparing data for PSE...')
        if len(active_classes) == 0:
            pse_data = [[am, de] for am, de in zip(
                active_meds, departments)]
        else:
            pse_data = [[am, ac, de] for am, ac, de in zip(
                active_meds, active_classes, departments)]
        self.n_pse_columns = len(pse_data[0])
        return pse_data

    # Define the profile state encoder pipeline, encode as multihot if use_lsi
    # is false and perform latent semantic indexing (decomposed into tfidf and
    # truncated SVD (gensim LsiTransformer) because uses too much ram with
    # sci-kit learn lsi transformer)
    def define_pse_pipeline(self, use_lsi=False, tsvd_n_components=0):
        self.use_lsi = use_lsi
        self.tsvd_n_components = tsvd_n_components
        pse_transformers = []
        for i in range(self.n_pse_columns):
            pse_transformers.append(('pse{}'.format(i), CountVectorizer(
                lowercase=False, preprocessor=self.pse_pp, analyzer=self.pse_a), i))
        pse_pipeline_transformers = [
            ('columntrans', ColumnTransformer(transformers=pse_transformers))
        ]
        if self.use_lsi:
            pse_pipeline_transformers.extend([
                ('tfidf', TfidfTransformer()),
                ('sparse2corpus', FunctionTransformer(func=Sparse2Corpus,
                                                      accept_sparse=True, validate=False, kw_args={'documents_columns': False})),
                ('tsvd', LsiTransformer(self.tsvd_n_components))
            ])
        self.pse = Pipeline(pse_pipeline_transformers)

    # Fit and save the pipeline as well as required values to properly load and use.
    def fitsave_pse_pipeline(self, save_path, pse_data, n_fold):
        print('Fitting PSE...')
        self.pse.fit(pse_data)
        # If encoding as multi-hot (no latent semantic indexing)
        # compute the shape of the multi-hot (required by the neural network)
        if self.use_lsi == False:
            pse_shape = sum([len(transformer[1].vocabulary_)
                             for transformer in self.pse.named_steps['columntrans'].transformers_])
        else:
            pse_shape = self.tsvd_n_components
        # save the fitted profile state encoder
        joblib.dump((pse_shape, n_fold, self.pse, self.pse_pp, self.pse_a),
                    os.path.join(save_path, 'pse.joblib'))
        return self.pse, pse_shape

    # string preprocessor (join the strings with spaces to simulate a text)
    def pse_pp(self, x):
        return ' '.join(x)

    # string analyzer (do not transform the strings, use them as is because they are not words.)
    def pse_a(self, x):
        return x

    # Encode the labels, save the pipeline
    def fitsave_labelencoder(self, save_path, targets, n_fold, mode):

        if mode == 'retrospective-autoenc':
            le = MultiLabelBinarizer()
        else:
            le = LabelEncoder()
        le.fit(targets)
        output_n_classes = len(le.classes_)
        joblib.dump((output_n_classes, n_fold, le),
                    os.path.join(save_path, 'le.joblib'))
        return le, output_n_classes


class TransformedGenerator(keras.utils.Sequence):

    '''
    This is a Sequence generator that takes the fitted scikit-learn pipelines, the data
    and some parameters required to do the transformation properly, and them transforms
    them batch by batch into numpy arrays. This is necessary because transforming the 
    entire dataset at once uses an ungodly amount of RAM and takes forever.
    '''

    def __init__(self, mode, w2v, use_lsi, pse, le, y, X_w2v_pre, X_w2v_post, X_am, X_ac, X_depa, w2v_embedding_dim, sequence_length, batch_size, shuffle=True, return_targets=True):
        # Mode
        self.mode = mode
        # Fitted scikit-learn pipelines
        self.w2v = w2v
        self.use_lsi = use_lsi
        self.pse = pse
        self.le = le
        # Data
        self.y = y
        self.X_w2v_pre = X_w2v_pre
        self.X_w2v_post = X_w2v_post
        self.X_am = X_am
        self.X_ac = X_ac
        if len(self.X_ac) == 0:
            self.use_classes = False
        else:
            self.use_classes = True
        self.X_depa = X_depa
        # Transformation parameters
        self.w2v_embedding_dim = w2v_embedding_dim
        self.sequence_length = sequence_length
        # Training hyperparemeters
        self.batch_size = batch_size
        # Do you want to shuffle ? True if you train, False if you evaluate
        self.shuffle = shuffle
        # Do you want the targets ? True if you're training or evaluating,
        # False if you're predicting
        self.return_targets = return_targets

    def __len__(self):
        # Required by tensorflow, compute the length of the generator
        # which is the number of batches given the batch size
        return int(np.ceil(len(self.X_w2v_pre) / float(self.batch_size)))

    def __getitem__(self, idx):
        # Transformation happens here.
        # Features go into a dict
        X = dict()
        # Transform the sequence into word2vec embeddings
        # Get a batch
        batch_w2v_pre = self.X_w2v_pre[idx *
                                       self.batch_size:(idx+1) * self.batch_size]
        transformed_w2v_pre = [[self.w2v.gensim_model.wv.get_vector(medic) if medic in self.w2v.gensim_model.wv.index2entity else np.zeros(
            self.w2v_embedding_dim) for medic in seq] if len(seq) > 0 else [] for seq in batch_w2v_pre]
        transformed_w2v_pre = keras.preprocessing.sequence.pad_sequences(
            transformed_w2v_pre, maxlen=self.sequence_length, dtype='float32')
        # Name the w2v feature dict key according to prediction mode
        if self.mode == 'prospective' or self.mode == 'retrospective-autoenc':
            w2v_pre_namestring = 'w2v_input'
        elif self.mode == 'retrospective':
            w2v_pre_namestring = 'w2v_pre_input'
        X[w2v_pre_namestring] = transformed_w2v_pre
        # Compute post sequences only in retrospective mode
        if self.mode == 'retrospective':
            batch_w2v_post = self.X_w2v_post[idx *
                                             self.batch_size:(idx+1) * self.batch_size]
            transformed_w2v_post = [[self.w2v.gensim_model.wv.get_vector(medic) if medic in self.w2v.gensim_model.wv.index2entity else np.zeros(
                self.w2v_embedding_dim) for medic in seq] if len(seq) > 0 else [] for seq in batch_w2v_post]
            transformed_w2v_post = keras.preprocessing.sequence.pad_sequences(
                transformed_w2v_post, maxlen=self.sequence_length, dtype='float32', padding='post', truncating='post')
            X['w2v_post_input'] = transformed_w2v_post
        # Transform the active meds, pharmacological classes and department into a multi-hot vector
        # Get batches
        batch_am = self.X_am[idx * self.batch_size:(idx+1) * self.batch_size]
        if self.use_classes:
            batch_ac = self.X_ac[idx * self.batch_size:(idx+1) * self.batch_size]
        batch_depa = self.X_depa[idx *
                                 self.batch_size:(idx+1) * self.batch_size]
        # Prepare the batches for input into the ColumnTransformer step of the pipeline
        if self.use_classes:
            batch_pse = [[bp, bc, bd]
                         for bp, bc, bd in zip(batch_am, batch_ac, batch_depa)]
        else:
            batch_pse = [[bp, bd]
                     for bp, bd in zip(batch_am, batch_depa)]
        # Transform
        transformed_pse = self.pse.transform(batch_pse)
        X['pse_input'] = transformed_pse
        # Output of the pipeline is a sparse matrix, convert to dense only if inputting
        # as multi hot. If using LSI, transformation pipeling will generate dense matrix.
        if self.use_lsi == False:
            X['pse_input'] = X['pse_input'].todense()
        if self.return_targets:
            # Get a batch
            batch_y = self.y[idx * self.batch_size:(idx+1) * self.batch_size]
            # Transform the batch
            if self.mode == 'retrospective-autoenc':
                transformed_y = self.le.transform(batch_y)
            else:
                transformed_y = self.le.transform(batch_y)
            y = {'main_output': transformed_y}
            return X, y
        else:
            return X

    def on_epoch_end(self):
        # Shuffle after each training epoch so that the data is not always
        # seen in the same order
        if self.shuffle == True:
            if self.use_classes:
                if self.mode == 'prospective':
                    shuffled = list(zip(self.y, self.X_w2v_pre,
                                    self.X_am, self.X_ac, self.X_depa))
                    random.shuffle(shuffled)
                    self.y, self.X_w2v_pre, self.X_am, self.X_ac, self.X_depa = zip(
                    *shuffled)
                elif self.mode == 'retrospective':
                    shuffled = list(zip(self.y, self.X_w2v_pre,
                                        self.X_w2v_post, self.X_am, self.X_ac, self.X_depa))
                    random.shuffle(shuffled)
                    self.y, self.X_w2v_pre, self.X_w2v_post, self.X_am, self.X_ac, self.X_depa = zip(
                        *shuffled)
                elif self.mode == 'retrospective-autoenc':
                    shuffled = list(zip(self.y, self.X_w2v_pre, self.X_am, self.X_ac, self.X_depa))
                    random.shuffle(shuffled)
                    self.y, self.X_w2v_pre, self.X_am, self.X_ac, self.X_depa = zip(
                        *shuffled)
            else:
                if self.mode == 'prospective':
                    shuffled = list(zip(self.y, self.X_w2v_pre,
                                        self.X_am, self.X_depa))
                    random.shuffle(shuffled)
                    self.y, self.X_w2v_pre, self.X_am, self.X_depa = zip(
                        *shuffled)
                elif self.mode == 'retrospective':
                    shuffled = list(zip(self.y, self.X_w2v_pre,
                                        self.X_w2v_post, self.X_am, self.X_depa))
                    random.shuffle(shuffled)
                    self.y, self.X_w2v_pre, self.X_w2v_post, self.X_am, self.X_depa = zip(
                        *shuffled)
                elif self.mode == 'retrospective-autoenc':
                    shuffled = list(zip(self.y, self.X_w2v_pre,
                                        self.X_am, self.X_depa))
                    random.shuffle(shuffled)
                    self.y, self.X_w2v_pre, self.X_am, self.X_depa = zip(
                        *shuffled)


class neural_network:

    '''
    Functions related to the neural network
    '''

    def __init__(self, mode):
        # Define the neural network mode (prospective or retrospective)
        self.mode = mode

    # Custom accuracy metrics
    def sparse_top10_accuracy(self, y_true, y_pred):
        sparse_top_k_categorical_accuracy = keras.metrics.sparse_top_k_categorical_accuracy
        return (sparse_top_k_categorical_accuracy(y_true, y_pred, k=10))

    def sparse_top30_accuracy(self, y_true, y_pred):
        sparse_top_k_categorical_accuracy = keras.metrics.sparse_top_k_categorical_accuracy
        return (sparse_top_k_categorical_accuracy(y_true, y_pred, k=30))

    def autoencoder_accuracy(self, y_true, y_pred):
        dichot_ypred = dtypes.cast(math.greater_equal(y_pred, constant(0.5)), float32)
        maximums = math.count_nonzero(math.maximum(y_true, dichot_ypred), 1, dtype=float32)
        correct = math.count_nonzero(math.multiply(y_true, dichot_ypred), 1, dtype=float32)
        return math.reduce_mean(math.xdivy(correct, maximums))

    # Callbacks during training
    def callbacks(self, save_path, n_fold, callback_mode='train_with_valid', learning_rate_schedule=None):

        # Learning rate schedule is a dict where the keys are
        # the epoch at which the learning rate changes and values
        # are the new learning rate.
        self.learning_rate_schedule = learning_rate_schedule

        # Assign simple names
        CSVLogger = keras.callbacks.CSVLogger
        EarlyStopping = keras.callbacks.EarlyStopping
        ReduceLROnPlateau = keras.callbacks.ReduceLROnPlateau
        ModelCheckpoint = keras.callbacks.ModelCheckpoint
        LearningRateScheduler = keras.callbacks.LearningRateScheduler

        # Define the callbacks
        callbacks = []

        callbacks.append(EpochLoggerCallback(save_path))
        callbacks.append(ModelCheckpoint(os.path.join(
            save_path, 'partially_trained_model_{}.h5'.format(n_fold)), verbose=1))
        # Train with valid and cross-val callbacks
        if callback_mode == 'train_with_valid' or callback_mode == 'cross_val':
            callbacks.append(ReduceLROnPlateau(
                monitor='val_loss', patience=3, min_delta=0.0005))
            callbacks.append(EarlyStopping(monitor='val_loss', min_delta=0.0001,
                                           patience=5, verbose=1, restore_best_weights=True))
        # Train with valid and train no valid callbacks
        if callback_mode == 'train_with_valid' or callback_mode == 'train_no_valid':
            callbacks.append(CSVLogger(os.path.join(
                save_path, 'training_history.csv'), append=True))
        if callback_mode == 'train_no_valid':
            callbacks.append(LearningRateScheduler(self.schedule, verbose=1))
        return callbacks

    def schedule(self, i, cur_lr):
        # The schedule is hardcoded here from the results
        # of a training with validation
        if i in self.learning_rate_schedule.keys():
            new_lr = self.learning_rate_schedule[i]
        else:
            new_lr = cur_lr
        return new_lr

    def define_model(self, sequence_size, n_add_seq_layers, dense_pse_size, concat_sequence_size, concat_total_size, dense_size, dropout, l2_reg, sequence_length, w2v_embedding_dim, pse_shape, n_add_pse_dense, n_dense, output_n_classes):

        # Assign simple names
        # Use CuDNN implementation of LSTM if GPU is available, LSTM if it isn't
        # (non-CuDNN implementation is slower even on GPU)
        # TODO upgrade to tensorflow 2: this is supposedly done automatically
        #if test.is_gpu_available():
        #    LSTM = keras.layers.CuDNNLSTM
        #else:
        #    LSTM = keras.layers.LSTM
        LSTM = keras.layers.LSTM

        Dense = keras.layers.Dense
        Dropout = keras.layers.Dropout
        Input = keras.layers.Input
        BatchNormalization = keras.layers.BatchNormalization
        concatenate = keras.layers.concatenate
        l2 = keras.regularizers.l2
        Model = keras.models.Model

        # Assign word2vec pre sequence input name according to mode
        if self.mode == 'prospective' or self.mode =='retrospective-autoenc':
            w2v_pre_input_name = 'w2v_input'
        elif self.mode == 'retrospective':
            w2v_pre_input_name = 'w2v_pre_input'

        to_concat_sequence = []
        to_concat = []
        inputs = []

        # The pre-target sequence word2vec inputs and layers before concatenation
        w2v_pre_input = Input(shape=(
            sequence_length, w2v_embedding_dim, ), dtype='float32', name=w2v_pre_input_name)
        w2v_pre = LSTM(sequence_size, return_sequences=True)(w2v_pre_input)
        w2v_pre = Dropout(dropout)(w2v_pre)
        for _ in range(n_add_seq_layers):
            w2v_pre = LSTM(sequence_size, return_sequences=True)(w2v_pre)
            w2v_pre = Dropout(dropout)(w2v_pre)
        w2v_pre = LSTM(sequence_size)(w2v_pre)
        w2v_pre = Dropout(dropout)(w2v_pre)
        w2v_pre = Dense(sequence_size, activation='relu')(w2v_pre)
        w2v_pre = Dropout(dropout)(w2v_pre)
        if self.mode == 'retrospective':
            to_concat_sequence.append(w2v_pre)
        elif self.mode == 'prospective' or self.mode == 'retrospective-autoenc':
            to_concat.append(w2v_pre)
        inputs.append(w2v_pre_input)

        # The pre-target sequence word2vec inputs and layers before concatenation
        # (used only in retrospective mode where we have info about what happened
        # after the target)
        if self.mode == 'retrospective':
            w2v_post_input = Input(shape=(
                sequence_length, w2v_embedding_dim, ), dtype='float32', name='w2v_post_input')
            w2v_post = LSTM(sequence_size, return_sequences=True,
                            go_backwards=True)(w2v_post_input)
            w2v_post = Dropout(dropout)(w2v_post)
            for _ in range(n_add_seq_layers):
                w2v_post = LSTM(sequence_size, return_sequences=True)(w2v_post)
                w2v_post = Dropout(dropout)(w2v_post)
            w2v_post = LSTM(sequence_size)(w2v_post)
            w2v_post = Dropout(dropout)(w2v_post)
            w2v_post = Dense(sequence_size, activation='relu',
                             kernel_regularizer=l2(l2_reg))(w2v_post)
            w2v_post = Dropout(dropout)(w2v_post)
            to_concat_sequence.append(w2v_post)
            inputs.append(w2v_post_input)

            concatenated_sequence = concatenate(to_concat_sequence)
            concatenated_sequence = BatchNormalization()(concatenated_sequence)
            concatenated_sequence = Dense(
                concat_sequence_size, activation='relu', kernel_regularizer=l2(l2_reg))(concatenated_sequence)
            concatenated_sequence = Dropout(dropout)(concatenated_sequence)
            to_concat.append(concatenated_sequence)

        # The multi-hot (or LSI-transformed) vector input (pse = profile state encoder) and layers before concatenation
        pse_input = Input(shape=(pse_shape,),
                          dtype='float32', name='pse_input')
        pse = Dense(dense_pse_size, activation='relu',
                    kernel_regularizer=l2(l2_reg))(pse_input)
        pse = Dropout(dropout)(pse)
        for _ in range(n_add_pse_dense):
            pse = BatchNormalization()(pse)
            pse = Dense(dense_pse_size, activation='relu',
                        kernel_regularizer=l2(l2_reg))(pse)
            pse = Dropout(dropout)(pse)
        to_concat.append(pse)
        inputs.append(pse_input)

        # Concatenation and dense layers
        concatenated = concatenate(to_concat)
        for n in range(n_dense):
            concatenated = BatchNormalization()(concatenated)
            if n == n_dense - 1 and self.mode == 'retrospective-autoenc':
                concatenated = Dense(concat_total_size*2, activation='relu',
                                    kernel_regularizer=l2(l2_reg))(concatenated)
            else:
                concatenated = Dense(concat_total_size, activation='relu',
                                    kernel_regularizer=l2(l2_reg))(concatenated)
            concatenated = Dropout(dropout)(concatenated)
        concatenated = BatchNormalization()(concatenated)
        if self.mode == 'retrospective-autoenc':
            output = Dense(output_n_classes, activation='sigmoid',
                        name='main_output')(concatenated)
        else:
            output = Dense(output_n_classes, activation='softmax',
                        name='main_output')(concatenated)

        # Compile the model
        model = Model(inputs=inputs, outputs=output)
        if self.mode == 'retrospective-autoenc':
            model.compile(optimizer='Adam', loss=['binary_crossentropy'], metrics=[self.autoencoder_accuracy, metrics.AUC(num_thresholds=10, curve='PR', name='aupr')])
        else:
            model.compile(optimizer='Adam', loss=['sparse_categorical_crossentropy'], metrics=[
                        'sparse_categorical_accuracy', self.sparse_top10_accuracy, self.sparse_top30_accuracy])

        return model


class EpochLoggerCallback(keras.callbacks.Callback):

    '''
    Custom callback that logs done epochs so that training
    can be resumed and continue for the correct number of
    total epochs
    '''

    def __init__(self, save_path):
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        self.done_epochs = epoch
        with open(os.path.join(self.save_path, 'done_epochs.pkl'), mode='wb') as file:
            pickle.dump(self.done_epochs, file)


class visualization:

    '''
    Functions that plot graphs
    '''

    def __init__(self):
        # Will be useful to decide to either show the plot (in
                # Jupyer Notebook) or save as a file (outside notebook)
        self.in_ipynb = check_ipynb().is_inipynb()

    def plot_accuracy_history(self, df, save_path):
        # Select only useful columns
        acc_df = df[['sparse_top10_accuracy', 'val_sparse_top10_accuracy', 'sparse_top30_accuracy',
                     'val_sparse_top30_accuracy', 'sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].copy()
        # Rename columns to clearer names
        acc_df.rename(inplace=True, index=str, columns={
            'sparse_top30_accuracy': 'Train top 30 accuracy',
            'val_sparse_top30_accuracy': 'Val top 30 accuracy',
            'sparse_top10_accuracy': 'Train top 10 accuracy',
            'val_sparse_top10_accuracy': 'Val top 10 accuracy',
            'sparse_categorical_accuracy': 'Train top 1 accuracy',
            'val_sparse_categorical_accuracy': 'Val top 1 accuracy',
        })
        # Structure the dataframe as expected by Seaborn
        acc_df = acc_df.stack().reset_index()
        acc_df.rename(inplace=True, index=str, columns={
                      'level_0': 'Epoch', 'level_1': 'Metric', 0: 'Result'})
        # Make sure the epochs are int to avoid weird ordering effects in the plot
        acc_df['Epoch'] = acc_df['Epoch'].astype('int8')
        # Plot
        sns.set(style='darkgrid')
        sns.relplot(x='Epoch', y='Result', hue='Metric',
                    kind='line', data=acc_df)
        # Output the plot
        if self.in_ipynb:
            plt.show()
        else:
            plt.savefig(os.path.join(save_path, 'acc_history.png'))
        # Clear
        plt.gcf().clear()

    def plot_autoenc_accuracy_history(self, df, save_path):
        # Select only useful columns
        acc_df = df[['autoencoder_accuracy', 'val_autoencoder_accuracy']].copy()
        # Rename columns to clearer names
        acc_df.rename(inplace=True, index=str, columns={
            'autoencoder_accuracy': 'Autoencoder accuracy',
            'val_autoencoder_accuracy': 'Val autoencoder accuracy',
            'aupr': 'Area under precision-recall',
            'val_aupr': 'Val area under precision-recall',
        })
        # Structure the dataframe as expected by Seaborn
        acc_df = acc_df.stack().reset_index()
        acc_df.rename(inplace=True, index=str, columns={
                      'level_0': 'Epoch', 'level_1': 'Metric', 0: 'Result'})
        # Make sure the epochs are int to avoid weird ordering effects in the plot
        acc_df['Epoch'] = acc_df['Epoch'].astype('int8')
        # Plot
        sns.set(style='darkgrid')
        sns.relplot(x='Epoch', y='Result', hue='Metric',
                    kind='line', data=acc_df)
        # Output the plot
        if self.in_ipynb:
            plt.show()
        else:
            plt.savefig(os.path.join(save_path, 'acc_history.png'))
        # Clear
        plt.gcf().clear()

    def plot_loss_history(self, df, save_path):
        loss_df = df[['loss', 'val_loss']].copy()
        loss_df.rename(inplace=True, index=str, columns={
                       'loss': 'Train loss', 'val_loss': 'Val loss'})
        loss_df = loss_df.stack().reset_index()
        loss_df.rename(inplace=True, index=str, columns={
                       'level_0': 'Epoch', 'level_1': 'Metric', 0: 'Result'})
        loss_df['Epoch'] = loss_df['Epoch'].astype('int8')
        sns.set(style='darkgrid')
        sns.relplot(x='Epoch', y='Result', hue='Metric',
                    kind='line', data=loss_df)
        if self.in_ipynb:
            plt.show()
        else:
            plt.savefig(os.path.join(save_path, 'loss_history.png'))
            plt.gcf().clear()

    def plot_crossval_accuracy_history(self, df, save_path):
        # Select only useful columns
        cv_results_df_filtered = df[['sparse_top30_accuracy', 'val_sparse_top30_accuracy', 'sparse_top10_accuracy',
                                     'val_sparse_top10_accuracy', 'sparse_categorical_accuracy', 'val_sparse_categorical_accuracy']].copy()
        # Rename columns to clearer names
        cv_results_df_filtered.rename(inplace=True, index=str, columns={
            'sparse_top30_accuracy': 'Train top 30 accuracy',
            'val_sparse_top30_accuracy': 'Val top 30 accuracy',
            'sparse_top10_accuracy': 'Train top 10 accuracy',
            'val_sparse_top10_accuracy': 'Val top 10 accuracy',
            'sparse_categorical_accuracy': 'Train top 1 accuracy',
            'val_sparse_categorical_accuracy': 'Val top 1 accuracy',
        })
        # Structure the dataframe as expected by Seaborn
        cv_results_graph_df = cv_results_df_filtered.stack().reset_index()
        cv_results_graph_df.rename(inplace=True, index=str, columns={
            'level_0': 'Split', 'level_1': 'Metric', 0: 'Result'})
        # Make sure the splits are int to avoid weird ordering effects in the plot
        cv_results_graph_df['Split'] = cv_results_graph_df['Split'].astype(
            'int8')
        # Plot
        sns.set(style='darkgrid')
        sns.relplot(x='Split', y='Result', hue='Metric',
                    kind='line', data=cv_results_graph_df)
        # Output the plot
        if self.in_ipynb:
            plt.show()
        else:
            plt.savefig(os.path.join(
                save_path, 'cross_val_acc_history.png'))
        # Clear
        plt.gcf().clear()

    def plot_crossval_loss_history(self, df, save_path):
        # Select only useful columns
        cv_results_df_filtered = df[['loss', 'val_loss']].copy()
        # Rename columns to clearer names
        cv_results_df_filtered.rename(inplace=True, index=str, columns={
            'loss': 'Train loss', 'val_loss': 'Val loss'})
        # Structure the dataframe as expected by Seaborn
        cv_results_graph_df = cv_results_df_filtered.stack().reset_index()
        cv_results_graph_df.rename(inplace=True, index=str, columns={
            'level_0': 'Split', 'level_1': 'Metric', 0: 'Result'})
        # Make sure the splits are int to avoid weird ordering effects in the plot
        cv_results_graph_df['Split'] = cv_results_graph_df['Split'].astype(
            'int8')
        # Plot
        sns.set(style='darkgrid')
        sns.relplot(x='Split', y='Result', hue='Metric',
                    kind='line', data=cv_results_graph_df)
        # Output the plot
        if self.in_ipynb:
            plt.show()
        else:
            plt.savefig(os.path.join(
                save_path, 'cross_val_loss_history.png'))
        # Clear
        plt.gcf().clear()

    def plot_crossval_autoenc_accuracy_history(self, df, save_path):
        # Select only useful columns
        cv_results_df_filtered = df[['autoencoder_accuracy', 'val_autoencoder_accuracy']].copy()
        # Rename columns to clearer names
        cv_results_df_filtered.rename(inplace=True, index=str, columns={
            'autoencoder_accuracy': 'Autoencoder accuracy',
            'val_autoencoder_accuracy': 'Val autoencoder accuracy',
            'aupr': 'Area under precision-recall',
            'val_aupr': 'Val area under precision-recall',
        })
        # Structure the dataframe as expected by Seaborn
        cv_results_graph_df = cv_results_df_filtered.stack().reset_index()
        cv_results_graph_df.rename(inplace=True, index=str, columns={
            'level_0': 'Split', 'level_1': 'Metric', 0: 'Result'})
        # Make sure the splits are int to avoid weird ordering effects in the plot
        cv_results_graph_df['Split'] = cv_results_graph_df['Split'].astype(
            'int8')
        # Plot
        sns.set(style='darkgrid')
        sns.relplot(x='Split', y='Result', hue='Metric',
                    kind='line', data=cv_results_graph_df)
        # Output the plot
        if self.in_ipynb:
            plt.show()
        else:
            plt.savefig(os.path.join(
                save_path, 'cross_val_acc_history.png'))
        # Clear
        plt.gcf().clear()

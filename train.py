# %% [markdown]
# # Start or resume training a model.
# Can be done with cross validation or as a single training run with or without validation.

# %% [markdown]
# ## Setup
#
# Setup the environment
# %%
# Imports

import os
import pathlib
import pickle
import random
import statistics
import warnings
from datetime import datetime
from multiprocessing import cpu_count
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from tensorflow import keras

from components import (TransformedGenerator, check_ipynb, data,
                        neural_network, transformation_pipelines,
                        gan_continue_checker, visualization, Sampling)

warnings.filterwarnings('ignore',category=UserWarning)

# %%[markdown]
# ## Parameters

# %%
# Save path and parameter loading

# Empty string to start training a new model, else will load parameters from this directory; will start a new model if can't load.
LOAD_FROM = ''

# Check if can load hyperparameters from specified dir, else create new dir
try:
    new_model = False
    save_path = os.path.join('experiments', LOAD_FROM)
    with open(os.path.join(save_path, 'hp.pkl'), mode='rb') as file:
        parameters_dict = pickle.load(
            file)
    param = SimpleNamespace(**parameters_dict)
    # Cannot proprely save and resume for now with gan.
    # TODO fix model saving and resuming training with gan. Remove this code when fixed.
    if param.MODE == 'retrospective-gan':
        print('WARNING ! Saving and resuming training in adversarial mode is buggy for now. Do not do this.')
        quit()
    print('Parameters of previous training successfully loaded. Resuming...')
except:
    new_model = True
    random_seed = random.randint(0,10000)
    # Parameters for new model
    parameters_dict = {

        # Execution parameters
        # retrospective for medication profile analysis, prospective for order prediction,
        # retrospective-autoenc for medication profile analysis with autoencoder,
        # retrospective-gan for medication profile analysis with autoencoder GAN
        'MODE': 'retrospective-autoenc',
        # SPLIT MODE, use 'year' for preprocessed data keyed by year or 'enc' for preprocessed data keyed by enc
        'SPLIT_MODE':'year',

        # Parameters for year split modes
        'NUM_TRAINING_YEARS':4,
        'MAX_YEAR_IN_SET':2017,

        # Keep chronological sequence when splitting for validation
        'KEEP_TIME_ORDER':True, # True for local dataset, False for mimic, no effect for year split mode
        'VAL_SPLIT_SEED':random_seed, # Seed to get identical splits when resuming training if KEEP_TIME_ORDER is False
        # True to do cross-val, false to do single training run with validation

        'CROSS_VALIDATE': True,
        'N_CROSS_VALIDATION_FOLDS': 3,
        # validate when doing a single training run. Cross-validation has priority over this
        'VALIDATE': True,

        # Data parameters
        # False prepares all data, True samples a number of encounters for faster execution, useful for debugging or testing
        'RESTRICT_DATA': True,
        'RESTRICT_SAMPLE_SIZE':2000, # The number of encounters to sample in the restricted data / sample by year in year split mode
        'DATA_DIR': '20yr_byyear',  # Where to find the preprocessed data.

        # Word2vec parameters
        'W2V_ALPHA': 0.013, # for local dataset 0.013, for mimic 0.013
        'W2V_ITER': 32, # for local dataset 32, for mimic 32
        'W2V_EMBEDDING_DIM': 128, # for local dataset 128, for mimic 64
        'W2V_HS': 0, # for local dataset 0, for mimic 0
        'W2V_SG': 0, # for local dataset 0, for mimic 1
        'W2V_MIN_COUNT': 5,
        # Exports only in a single training run, no effect in cross-validation
        'EXPORT_W2V_EMBEDDINGS': False,

        # Profile state encoder (PSE) parameters
        'USE_LSI': False,  # False: encode profile state as multi-hot. True: perform Tf-idf then tsvd on the profile state
        'TSVD_N_COMPONENTS': 200,  # Number of components on the lsi-transformed profile state

        # Neural network parameters (for prospective or retrospective modes)
        # Number of additional LSTM layers (minimum 1 not included in this count)
        'N_LSTM': 0,
        # Number of dense layers after lstm (minimum 0):
        'POST_LSTM_DENSE': 0,
        # Number of additional batchnorm/dense/dropout layers after PSE before concat (minimum 1 not included in this count)
        'N_PSE_DENSE': 1,
        # Number of batchnorm/dense/dropout layers after concatenation (minimum 1 not included in this count)
        'N_DENSE': 2,
        'LSTM_SIZE': 256, # 512 for retrospective, 128 for prospective
        'DENSE_PSE_SIZE': 256, # 256 for retrospective, 128 for prospective
        'CONCAT_LSTM_SIZE': 8, # 512 for retrospective, irrelevant for prospective
        'CONCAT_TOTAL_SIZE': 128, # 512 for retrospective, 256 for prospective
        'DENSE_SIZE': 256, #  128 for retrospective, 256 for prospective
        'DROPOUT': 0, # 0.3 for retrospective, 0.2 for prospective
        'L2_REG': 0,
        'SEQUENCE_LENGTH': 30,

        # Neural network paramters for retrospective-gan or retrospective-autoenc mode
        # Number of batchnorm/dense/dropout blocks after input into autoencoder
        'N_ENC_DEC_BLOCKS':1,
        # Size of largest encoder (first) and decoder (last) dense layer
        'AUTOENC_MAX_SIZE':256,
        # Denominator for division of initial size for subsequent layers
        'AUTOENC_SIZE_RATIO':2,
        # Autoencoder latent representation layer size
        'AUTOENC_SQUEEZE_SIZE':128,
        # Number of blocks in the feature extractor [GAN only](discriminator adds an additional layer with size 1 as an output layer)
        'FEAT_EXT_N_BLOCKS':2,
        # Size of dense layers in the feature extractor [GAN only]
        'FEAT_EXT_SIZE':4,
        # Discriminator will be like the encoder part of the autoencoder but with a single node instead of the latent representation layer size
        # Loss weights are the relative weights of each loss in this order: contextual loss (binary), adversarial loss (mse), validity (must be zero) and encoder loss (mse)
        'LOSS_WEIGHTS':[0.65, 0.27, 0, 0.08],

        # Neural network training parameters,
        'BATCH_SIZE': 256,
        'MAX_TRAINING_EPOCHS':1000, # Default 1000, should never get there, reduce for faster execution when testing or debugging.
        'SINGLE_RUN_EPOCHS':12, # How many epochs to train when doing a single run without validation. 16 for local retrospective. 7 for mimic prospective.
        'LEARNING_RATE_SCHEDULE':{8:1e-4, 11:1e-5}, # Dict where keys are epoch index (epoch - 1) where the learning rate decreases and values are the new learning rate. {14:1e-4} for local data retrospective. {} for mimic prospective.
        'N_TRAINING_STEPS_PER_EPOCH': None, # 1000 for retrospective, None for prospective (use whole generator)
        'N_VALIDATION_STEPS_PER_EPOCH': None, # 1000 for retrospective, None for prospective (use whole generator)
    }
    param = SimpleNamespace(**parameters_dict)

    # Save the parameters
    save_stamp = datetime.now().strftime('%Y%m%d-%H%M')
    save_path = os.path.join(
        os.getcwd(), 'experiments', param.MODE, save_stamp)
    pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(save_path, 'parameters.txt'), mode='w') as file:
        for k, v in parameters_dict.items():
            file.write(str(k) + ': ' + str(v) + '\n')

    with open(os.path.join(save_path, 'hp.pkl'), mode='wb') as file:
        pickle.dump(parameters_dict, file)
    print('Could not load from previous training. Starting new model...')

# Check at which fold to resume the training, else start at beginning
try:
    with open(os.path.join(save_path, 'done_crossval_folds.pkl'), mode='rb') as file:
        initial_fold = pickle.load(file) + 1
except:
    initial_fold = 0

if param.CROSS_VALIDATE:
    print('Starting cross-validation at fold: {}'.format(initial_fold))
    if initial_fold >= param.N_CROSS_VALIDATION_FOLDS:
        print('Cross-validation already completed: would start at fold index {} while {} folds planned. Quitting...'.format(initial_fold, param.N_CROSS_VALIDATION_FOLDS))
        quit()

# Check at which epoch to resume the training, else start at beginning
try:
    with open(os.path.join(save_path, 'done_epochs.pkl'), mode='rb') as file:
        initial_epoch = pickle.load(file) + 1
except:
    initial_epoch = 0
print('Neural network training will begin at epoch: {}'.format(initial_epoch))
if param.CROSS_VALIDATE == False and os.path.isfile(os.path.join(save_path, 'model.h5')):
    print('Training with validation already completed, final model already saved. Quitting...')
    quit()


# %%
# Check if running inside Jupyter notebook or not (will be used later for Keras progress bars)

in_ipynb = check_ipynb().is_inipynb()

# %% [markdown]
# ## Data
#
# Load the data to start or resume the training

# %%
# Load the data

d = data(param.DATA_DIR, param.MODE, param.KEEP_TIME_ORDER, param.VAL_SPLIT_SEED)

if os.path.isfile(os.path.join(save_path, 'sampled_encs.pkl')):
    enc_file = os.path.join(save_path, 'sampled_encs.pkl')
    print('Loaded partially completed experiment was done with RESTRICTED DATA !')
else:
    enc_file = False

# Retrospective-gan and retrospective-autoenc modes do not require profiles (uses only active meds)
if param.MODE == 'retrospective-gan' or param.MODE == 'retrospective-autoenc':
    get_profiles = False
else:
    get_profiles = True

if new_model:
    d.load_data(restrict_data=param.RESTRICT_DATA, save_path=save_path,
                restrict_sample_size=param.RESTRICT_SAMPLE_SIZE, get_profiles=get_profiles)
else:
    d.load_data(previous_encs_path=enc_file)

# %% [markdown]
# Split encounters into a train and test set, if doing cross-validation
# prepare the appropriate number of folds

if param.SPLIT_MODE == 'enc':
    if param.CROSS_VALIDATE:
        d.cross_val_split(param.N_CROSS_VALIDATION_FOLDS, split_seed=param.VAL_SPLIT_SEED)
    elif param.VALIDATE:
        d.split(split_seed=param.VAL_SPLIT_SEED)

# %% [markdown]
# ## Training execution
# Executed within a loop for cross-validationm if not cross-validating
# will run the loop only one time.

# %%
if param.CROSS_VALIDATE:
    loop_iters = param.N_CROSS_VALIDATION_FOLDS
else:
    loop_iters = 1

# %% [markdown]
# ## Loop

# %%
for i in range(initial_fold, loop_iters):

    # Make the data lists
    if param.CROSS_VALIDATE:
        cross_val_fold = i
        print('Performing cross-validation, fold: {}'.format(i))
        get_valid = True
    elif param.VALIDATE:
        get_valid = True
        cross_val_fold = None
    else:
        get_valid=False
        cross_val_fold = None

    if param.SPLIT_MODE == 'enc':
        profiles_train, targets_train, pre_seq_train, post_seq_train, active_meds_train, active_classes_train, depa_train, targets_test, pre_seq_test, post_seq_test, active_meds_test, active_classes_test, depa_test, definitions = d.make_lists(get_valid=get_valid,
        cross_val_fold=cross_val_fold)
    elif param.SPLIT_MODE == 'year':
        valid_year_begin = param.MAX_YEAR_IN_SET - i
        valid_years = range(valid_year_begin, valid_year_begin+1)
        train_years_end = valid_year_begin - 1
        train_years_begin = train_years_end - param.NUM_TRAINING_YEARS + 1
        training_years = range(train_years_begin, train_years_end + 1)
        print('Performing cross-validation fold {} with training data years: {} - {} and validation year: {}\n\n'.format(i, train_years_begin, train_years_end-1, valid_year_begin))
        profiles_train, targets_train, pre_seq_train, post_seq_train, active_meds_train, active_classes_train, depa_train, targets_test, pre_seq_test, post_seq_test, active_meds_test, active_classes_test, depa_test, definitions = d.make_lists_by_year(train_years=training_years, valid_years=valid_years, shuffle_train_set=True)

    if param.MODE == 'retrospective-autoenc' or param.MODE == 'retrospective-gan':
        targets_train = active_meds_train
        targets_test = active_meds_test

    # Try loading previously fitted transformation pipelines. If they do not exist or do not
    # correspond to the current fold, build and fit new pipelines for word2vec embeddings,
    # profile state encoder and label encoder.
    tp = transformation_pipelines()

    # Word2vec embeddings
    # Create a word2vec pipeline and train word2vec embeddings in that pipeline
    # on the training set profiles. Optionnaly export word2vec embeddings.
    if param.MODE not in ['retrospective-autoenc', 'retrospective-gan']:
        try:
            n_fold, w2v = joblib.load(os.path.join(save_path, 'w2v.joblib'))
            assert n_fold == i
            print('Successfully loaded word2vec pipeline for current fold.')
        except:
            print('Could not load word2vec pipeline for current fold...')
            tp.define_w2v_pipeline(param.W2V_ALPHA, param.W2V_ITER, param.W2V_EMBEDDING_DIM,
                                param.W2V_HS, param.W2V_SG, param.W2V_MIN_COUNT, cpu_count())
            w2v = tp.fitsave_w2v_pipeline(save_path, profiles_train, i)
            if param.CROSS_VALIDATE == False and param.EXPORT_W2V_EMBEDDINGS == True:
                tp.export_w2v_embeddings(save_path, definitions_dict=definitions)

    # Profile state encoder (PSE)
    # Encode the profile state, either as a multi-hot vector (binary count vectorizer) or using Latent Semantic Indexing.
    try:
        pse_shape, n_fold, pse, pse_pp, pse_a = joblib.load(
            os.path.join(save_path, 'pse.joblib'))
        assert n_fold == i
        print('Successfully loaded PSE pipeline for current fold.')
    except:
        print('Could not load PSE pipeline for current fold...')
        if param.MODE == 'retrospective-gan' or param.MODE == 'retrospective-autoenc':
            pse_data = tp.prepare_pse_data(active_meds_train, [], [])
        else:
            pse_data = tp.prepare_pse_data(
                active_meds_train, active_classes_train, depa_train)
        tp.define_pse_pipeline(use_lsi=param.USE_LSI,
                               tsvd_n_components=param.TSVD_N_COMPONENTS)
        pse, pse_shape = tp.fitsave_pse_pipeline(save_path, pse_data, i)

    # Label encoder
    # Encode the label using a MultiLabelBinarizer if retrospective-autoenc or retrospective-gan mode, otherwise LabelEncoder
    try:
        output_n_classes, n_fold, le = joblib.load(
            os.path.join(save_path, 'le.joblib'))
        assert n_fold == i
        print('Successfully loaded label encoder for current fold.')
    except:
        print('Could not load label encoder for current fold...')
        le, output_n_classes = tp.fitsave_labelencoder(
            save_path, targets_train, i, param.MODE)

    # Neural network

    # Build the generators, prepare the variables for fitting
    if param.MODE not in ['retrospective-gan', 'retrospective-autoenc']:
        w2v_step = w2v.named_steps['w2v']
    else:
        w2v_step = None
    train_generator = TransformedGenerator(param.MODE, w2v_step, param.USE_LSI, pse, le, targets_train, pre_seq_train, post_seq_train,
                                           active_meds_train, active_classes_train, depa_train, param.W2V_EMBEDDING_DIM, param.SEQUENCE_LENGTH, param.BATCH_SIZE)

    if param.CROSS_VALIDATE or param.VALIDATE:
        test_generator = TransformedGenerator(param.MODE, w2v_step, param.USE_LSI, pse, le, targets_test, pre_seq_test, post_seq_test,
                                           active_meds_test, active_classes_test, depa_test, param.W2V_EMBEDDING_DIM, param.SEQUENCE_LENGTH, param.BATCH_SIZE)
        n_validation_steps_per_epoch = param.N_VALIDATION_STEPS_PER_EPOCH
        training_epochs = param.MAX_TRAINING_EPOCHS
    else:
        test_generator = None
        n_validation_steps_per_epoch = None
        training_epochs = param.SINGLE_RUN_EPOCHS

    # Define the network and train
    n = neural_network(param.MODE)

    if param.MODE == 'retrospective-gan':

        custom_objects_dict = {'autoencoder_accuracy':n.autoencoder_accuracy, 'autoencoder_false_neg_rate':n.autoencoder_false_neg_rate, 'Sampling':Sampling}
        val_monitor_losses = []
        # TODO load previously saved models and resume
        # The code to save and resume training the gan is buggy and doesn't proprely restore the model weights. Do not use for now.
        # TODO fix model saving and resuming training with gan
        '''
        if i == initial_fold:
            try:
                gan_encoder = tf.keras.models.load_model(os.path.join(save_path, 'partially_trained_encoder_{}_{}.h5'.format(i, initial_epoch - 1)), custom_objects=custom_objects_dict)
                gan_decoder = tf.keras.models.load_model(os.path.join(save_path, 'partially_trained_decoder_{}_{}.h5'.format(i, initial_epoch - 1)), custom_objects=custom_objects_dict)
                gan_discriminator = tf.keras.models.load_model(os.path.join(save_path, 'partially_trained_discriminator_{}_{}.h5'.format(i, initial_epoch - 1)), custom_objects=custom_objects_dict)
                gan_adv_autoencoder = tf.keras.models.load_model(os.path.join(save_path, 'partially_trained_adversarial_model_{}_{}.h5'.format(i, initial_epoch - 1)), custom_objects=custom_objects_dict)
                print('Successfully loaded partially trained models from fold {} epoch {}'.format(i, initial_epoch))
            except:
                gan_encoder, gan_decoder, gan_discriminator, gan_adv_autoencoder = n.aaa(param.N_ENC_DEC_BLOCKS, param.AUTOENC_MAX_SIZE, param.AUTOENC_SIZE_RATIO, param.AUTOENC_SQUEEZE_SIZE, pse_shape, param.DROPOUT)
        else:
            gan_encoder, gan_decoder, gan_discriminator, gan_adv_autoencoder = n.aaa(param.N_ENC_DEC_BLOCKS, param.AUTOENC_MAX_SIZE, param.AUTOENC_SIZE_RATIO, param.AUTOENC_SQUEEZE_SIZE, pse_shape, param.DROPOUT)
        '''
        gan_encoder, gan_decoder, gan_discriminator, gan_feature_extractor, gan_adv_autoencoder = n.aaa(param.N_ENC_DEC_BLOCKS, param.AUTOENC_MAX_SIZE, param.AUTOENC_SIZE_RATIO, param.AUTOENC_SQUEEZE_SIZE, param.FEAT_EXT_N_BLOCKS, param.FEAT_EXT_SIZE, pse_shape, param.DROPOUT, param.LOSS_WEIGHTS)

        if (param.CROSS_VALIDATE == True and i == 0) or (param.CROSS_VALIDATE == False):
            tf.keras.utils.plot_model(
                gan_adv_autoencoder, to_file=os.path.join(save_path, 'model.png'))
        
        # Custom training loop
        c = gan_continue_checker(i, save_path)
        for epoch in range(initial_epoch, training_epochs):

            d_losses = []
            g_losses = []

            print('EPOCH {}\n\n'.format(epoch + 1))
            print('TRAINING...')

            for batch_idx  in tqdm(range(len(train_generator))):
            
                # Train discriminator

                batch_data_X, batch_data_y = train_generator.__getitem__(batch_idx)

                ones_label = np.ones((len(batch_data_X['pse_input']),1))
                zeros_label = np.zeros((len(batch_data_X['pse_input']),1))

                latent_from_data = gan_encoder.predict(batch_data_X['pse_input'])
                #latent_generated = np.random.normal(size=(len(batch_data_X['pse_input']), param.AUTOENC_SQUEEZE_SIZE))

                reconstructed_from_data = gan_decoder.predict(latent_from_data)
                #reconstructed_generated = gan_decoder.predict(latent_generated)

                #d_loss_generated = gan_discriminator.train_on_batch(reconstructed_generated, ones_label)
                d_loss_generated = gan_discriminator.train_on_batch(reconstructed_from_data, ones_label)
                #d_loss_from_data = gan_discriminator.train_on_batch(reconstructed_from_data, zeros_label)
                d_loss_from_data = gan_discriminator.train_on_batch(batch_data_X['pse_input'], zeros_label)
                d_loss = 0.5 * np.add(d_loss_generated, d_loss_from_data)

                # Train generator

                feature_extracted = gan_feature_extractor.predict(reconstructed_from_data)
                g_loss = gan_adv_autoencoder.train_on_batch(batch_data_X['pse_input'], [batch_data_y['main_output'], feature_extracted, ones_label, latent_from_data])

                d_losses.append(d_loss)
                g_losses.append(g_loss)

            # Compute the metrics for the epoch
            d_losses = np.array(d_losses)
            d_losses = np.mean(d_losses, axis=0)

            g_losses = np.array(g_losses)
            g_losses = np.mean(g_losses, axis=0)

            all_names = gan_discriminator.metrics_names + gan_adv_autoencoder.metrics_names
            all_losses = np.hstack((d_losses, g_losses)).tolist()

            print('EPOCH {} TRAINING RESULTS:'.format(epoch+1))
            print('\n'.join(['{} : {:.5f}'.format(name,metric) for name,metric in zip(all_names, all_losses)]))
            print('\n')

            # Test data
            if param.CROSS_VALIDATE or param.VALIDATE:
                g_val_losses = []

                print('VALIDATION...')
                for batch_idx  in tqdm(range(len(test_generator))):
                    
                    batch_data_X, batch_data_y = test_generator.__getitem__(batch_idx)
                    ones_label = np.ones((len(batch_data_X['pse_input']),1))
                    latent_repr = gan_encoder.predict(batch_data_X['pse_input'])
                    feature_extracted = gan_feature_extractor.predict(gan_decoder.predict(latent_repr))

                    g_loss = gan_adv_autoencoder.test_on_batch(batch_data_X['pse_input'], [batch_data_y['main_output'], feature_extracted, ones_label, latent_repr])

                    g_val_losses.append(g_loss)
                    # In current config model_2_loss is the loss of the autoencoder and it is at index 1 of g_loss and g_val_losses

                g_val_losses = np.array(g_val_losses)
                g_val_losses = np.mean(g_val_losses, axis=0)
                val_monitor_losses.append(g_val_losses[0])
                val_names = ['val_' + name for name in gan_adv_autoencoder.metrics_names]

                print('EPOCH {} VALIDATION RESULTS:'.format(epoch+1))
                print('\n'.join(['{} : {:.5f}'.format(name,metric) for name,metric in zip(val_names, g_val_losses)]))
                print('\n')

                all_names = all_names + val_names
                all_losses = np.hstack((all_losses, g_val_losses)).tolist()
                '''
                # Sample n fake profiles
                n_profiles = 3
                
                fake_reconsts = gan_decoder.predict(np.random.normal(size=(n_profiles, param.AUTOENC_SQUEEZE_SIZE)))
                fake_reconst_matrix = (fake_reconsts >= 0.5) * 1
                fake_reconst_labels = le.inverse_transform(fake_reconst_matrix)
                fake_profiles = [[definitions[med] for med in profile] for profile in fake_reconst_labels]
                print('These profiles do not exist:\n')
                for profile in fake_profiles:
                    profile.sort()
                    print('Profile: \n{}'.format(profile))
                print('\n\n')
                
                # Test n reconstructions

                sampled_profiles = random.sample(active_meds_test, n_profiles)
                transformed_profiles = pse.transform([[bp] for bp in sampled_profiles]).todense()
                reconstructed = gan_decoder.predict(gan_encoder.predict(transformed_profiles))
                reconst_matrix = (reconstructed >= 0.5) * 1
                reconstructed_labels = le.inverse_transform(reconst_matrix)
                reconstructed_profiles = [[definitions[med] for med in profile] for profile in reconstructed_labels]
                orig_profiles = [[definitions[med] for med in profile] for profile in sampled_profiles]
                atypical_meds = [[definitions[med] for med in orig_profile if med not in reconst_profile] for orig_profile, reconst_profile in zip(sampled_profiles, reconstructed_labels)]
                print('These profiles do exist:\n')
                for profile, atypical_med in zip(orig_profiles, atypical_meds):
                    profile.sort()
                    atypical_med.sort()
                    print('Profile: \n{}\nAtypicals: \n{}\n'.format(profile, atypical_med))
                '''

                continue_check = c.gan_continue_check(val_monitor_losses,epoch)
                if 'early_stop' in continue_check:
                    break
                elif 'reduce_lr' in continue_check:
                    cur_lr = gan_adv_autoencoder.optimizer.lr.numpy()
                    keras.backend.set_value(gan_adv_autoencoder.optimizer.lr, cur_lr/10)
                    
            epoch_results_df = pd.DataFrame.from_dict(
                {epoch: dict(zip(all_names, all_losses))}, orient='index')
            epoch_results_df['epoch']=epoch
            epoch_results_df['lr']=gan_adv_autoencoder.optimizer.lr.numpy()
            # save the dataframe to csv file, create new file at first epoch, else append
            if epoch == 0:
                epoch_results_df.to_csv(os.path.join(save_path, 'training_history.csv'))
            else:
                epoch_results_df.to_csv(os.path.join(
                    save_path, 'training_history.csv'), mode='a', header=False)

            gan_encoder.save(os.path.join(save_path, 'partially_trained_encoder_{}_{}.h5'.format(i,epoch)), save_format='tf')
            gan_decoder.save(os.path.join(save_path, 'partially_trained_decoder_{}_{}.h5'.format(i,epoch)), save_format='tf')
            gan_discriminator.save(os.path.join(save_path, 'partially_trained_discriminator_{}_{}.h5'.format(i,epoch)), save_format='tf')
            gan_feature_extractor.save(os.path.join(save_path, 'partially_trained_feature_extractor_{}_{}.h5'.format(i,epoch)), save_format='tf')
            gan_adv_autoencoder.save(os.path.join(save_path, 'partially_trained_adversarial_model_{}_{}.h5'.format(i,epoch)), save_format='tf')

            # Save that the epoch completed
            with open(os.path.join(save_path, 'done_epochs.pkl'), mode='wb') as file:
                pickle.dump(epoch, file)

        if param.CROSS_VALIDATE:
            # Save that the fold completed successfully
            with open(os.path.join(save_path, 'done_crossval_folds.pkl'), mode='wb') as file:
                pickle.dump(i, file)
            # make a dataframe with the fold metrics
            fold_results_df = pd.read_csv(os.path.join(save_path, 'training_history.csv'))
            fold_results_df = fold_results_df.loc[fold_results_df['epoch'] == c.absolute_min_loss_epoch].copy()
            # save the dataframe to csv file, create new file at first fold, else append
            if i == 0:
                fold_results_df.to_csv(os.path.join(save_path, 'cv_results.csv'))
            else:
                fold_results_df.to_csv(os.path.join(
                    save_path, 'cv_results.csv'), mode='a', header=False)
        # Else save the models
        else:
            gan_encoder.save(os.path.join(save_path, 'encoder.h5'))
            gan_decoder.save(os.path.join(save_path, 'decoder.h5'))
            gan_discriminator.save(os.path.join(save_path, 'discriminator.h5'))
            gan_adv_autoencoder.save(os.path.join(save_path, 'adversarial_model.h5'))

    else:

        # Try loading a partially trained neural network for current fold,
        # or define a new neural network
        if param.MODE == 'retrospective-autoenc':
            custom_objects_dict = {'autoencoder_accuracy':n.autoencoder_accuracy, 'autoencoder_false_neg_rate':n.autoencoder_false_neg_rate}
        else:
            custom_objects_dict = {'sparse_top10_accuracy': n.sparse_top10_accuracy, 'sparse_top30_accuracy': n.sparse_top30_accuracy}
        try:
            model = tf.keras.models.load_model(os.path.join(save_path, 'partially_trained_model_{}.h5'.format(i)), custom_objects=custom_objects_dict)
        except:
            if param.MODE == 'retrospective-autoenc':
                model = n.simple_autoencoder(param.N_ENC_DEC_BLOCKS, param.AUTOENC_MAX_SIZE, param.AUTOENC_SIZE_RATIO, param.AUTOENC_SQUEEZE_SIZE, pse_shape, param.DROPOUT)
            else:
                model = n.define_model(param.LSTM_SIZE, param.N_LSTM, param.POST_LSTM_DENSE, param.DENSE_PSE_SIZE, param.CONCAT_LSTM_SIZE, param.CONCAT_TOTAL_SIZE, param.DENSE_SIZE, param.DROPOUT, param.L2_REG, param.SEQUENCE_LENGTH, param.W2V_EMBEDDING_DIM, pse_shape, param.N_PSE_DENSE, param.N_DENSE, output_n_classes)

            if (param.CROSS_VALIDATE == True and i == 0) or (param.CROSS_VALIDATE == False):
                tf.keras.utils.plot_model(
                    model, to_file=os.path.join(save_path, 'model.png'))
        
        # Define the callbacks
        if param.CROSS_VALIDATE:
            callback_mode = 'cross_val'
        elif param.VALIDATE:
            callback_mode = 'train_with_valid'
        else:
            callback_mode = 'train_no_valid'
        callbacks = n.callbacks(save_path, i, callback_mode=callback_mode, learning_rate_schedule=param.LEARNING_RATE_SCHEDULE)

        # Train the model

        # Check if running in Jupyter or not to print progress bars if in terminal and log only at epoch level if in Jupyter.
        if in_ipynb:
            verbose = 2
        else:
            verbose = 1

        model.fit_generator(train_generator,
                            epochs=training_epochs,
                            steps_per_epoch=param.N_TRAINING_STEPS_PER_EPOCH,
                            callbacks=callbacks,
                            initial_epoch=initial_epoch,
                            validation_data=test_generator,
                            validation_steps=n_validation_steps_per_epoch,
                            verbose=verbose)

        # If doing cross-validation, get the metrics for the best
        # epoch.
        if param.CROSS_VALIDATE:
            train_results = model.evaluate_generator(
                train_generator, verbose=verbose)
            val_results = model.evaluate_generator(test_generator, verbose=verbose)

            # Make a list of validation metrics names
            val_metrics = [
                'val_' + metric_name for metric_name in model.metrics_names]

            # Concatenate all results in a list
            all_results = [*train_results, *val_results]
            all_metric_names = [*model.metrics_names, *val_metrics]

            # make a dataframe with the fold metrics
            fold_results_df = pd.DataFrame.from_dict(
                {i: dict(zip(all_metric_names, all_results))}, orient='index')
            # save the dataframe to csv file, create new file at first fold, else append
            if i == 0:
                fold_results_df.to_csv(os.path.join(save_path, 'cv_results.csv'))
            else:
                fold_results_df.to_csv(os.path.join(
                    save_path, 'cv_results.csv'), mode='a', header=False)
            # Save that the fold completed successfully
            with open(os.path.join(save_path, 'done_crossval_folds.pkl'), mode='wb') as file:
                pickle.dump(i, file)
        # Else save the model
        else:
            model.save(os.path.join(save_path, 'model.h5'))

# %%
# Plot the loss and accuracy during training

v = visualization()

# If cross-validating, plot evaluation metrics (best epoch)
# by fold. Else, plot training history by epoch.
if param.CROSS_VALIDATE:
    plotting_df = pd.read_csv(os.path.join(save_path, 'cv_results.csv'))
    if param.MODE == 'retrospective-gan':
        plotting_df.rename(inplace=True, index=str, columns={
            'model_3_autoencoder_accuracy': 'autoencoder_accuracy',
            'val_model_3_autoencoder_accuracy': 'val_autoencoder_accuracy',
            'model_3_aupr': 'aupr',
            'val_model_3_aupr': 'val_aupr',
            'model_3_autoencoder_false_neg_rate': 'autoencoder_false_neg_rate',
            'val_model_3_autoencoder_false_neg_rate': 'val_autoencoder_false_neg_rate',
            'model_4_accuracy':'model_accuracy',
            'val_model_4_accuracy':'val_model_accuracy',
        })
        v.plot_crossval_autoenc_accuracy_history(plotting_df, save_path)
        v.plot_crossval_gan_discacc_history(plotting_df, save_path)
    elif param.MODE == 'retrospective-autoenc':
        v.plot_crossval_autoenc_accuracy_history(plotting_df, save_path)
        v.plot_crossval_loss_history(plotting_df, save_path)
    else:
        v.plot_crossval_accuracy_history(plotting_df, save_path)
        v.plot_crossval_loss_history(plotting_df, save_path)
elif param.VALIDATE:
    plotting_df = pd.read_csv(os.path.join(save_path, 'training_history.csv'))
    if param.MODE == 'retrospective-gan':
        plotting_df.rename(inplace=True, index=str, columns={
            'model_3_autoencoder_accuracy': 'autoencoder_accuracy',
            'val_model_3_autoencoder_accuracy': 'val_autoencoder_accuracy',
            'model_3_aupr': 'aupr',
            'val_model_3_aupr': 'val_aupr',
            'model_3_autoencoder_false_neg_rate': 'autoencoder_false_neg_rate',
            'val_model_3_autoencoder_false_neg_rate': 'val_autoencoder_false_neg_rate',
            'model_4_accuracy':'model_accuracy',
            'val_model_4_accuracy':'val_model_accuracy',
        })
        v.plot_autoenc_accuracy_history(plotting_df, save_path)
        v.plot_gan_discacc_history(plotting_df, save_path)
    elif param.MODE == 'retrospective-autoenc':
        v.plot_autoenc_accuracy_history(plotting_df, save_path)
        v.plot_loss_history(plotting_df, save_path)
    else:
        v.plot_accuracy_history(plotting_df, save_path)
        v.plot_loss_history(plotting_df, save_path)

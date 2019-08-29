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
from datetime import datetime
from multiprocessing import cpu_count
from types import SimpleNamespace

import joblib
import pandas as pd
import tensorflow as tf

from components import (TransformedGenerator, check_ipynb, data,
                        neural_network, transformation_pipelines, visualization)

# %%[markdown]
# ## Parameters

# %%
# Save path and parameter loading

# Empty string to start training a new model, else will load parameters from this directory; will start a new model if can't load.
LOAD_FROM = 'prospective/20190829-1302'

# Check if can load hyperparameters from specified dir, else create new dir
try:
    new_model = False
    save_path = os.path.join('experiments', LOAD_FROM)
    with open(os.path.join(save_path, 'hp.pkl'), mode='rb') as file:
        parameters_dict = pickle.load(
            file)
    param = SimpleNamespace(**parameters_dict)
    print('Parameters of previous training successfully loaded. Resuming...')
except:
    new_model = True
    random_seed = random.randint(0,10000)
    # Parameters for new model
    parameters_dict = {

        # Execution parameters
        # retrospective for medication profile analysis, prospective for order prediction
        'MODE': 'prospective',
        # Keep chronological sequence when splitting for validation
        'KEEP_TIME_ORDER':False, # True for local dataset, False for mimic
        'VAL_SPLIT_SEED':random_seed, # Seed to get identical splits when resuming training if KEEP_TIME_ORDER is False
        # True to do cross-val, false to do single training run with validation
        'CROSS_VALIDATE': False,
        'N_CROSS_VALIDATION_FOLDS': 5,
        # Validate when doing a single training run. Cross-validation has priority over this
        'VALIDATE': True,

        # Data parameters
        # False prepares all data, True samples a number of encounters for faster execution, useful for debugging or testing
        'RESTRICT_DATA': False,
        'RESTRICT_SAMPLE_SIZE':1000, # The number of encounters to sample in the restricted data.
        'DATA_DIR': 'mimic',  # Where to find the preprocessed data.

        # Word2vec parameters
        'W2V_ALPHA': 0.013, # for local dataset 0.013, for mimic 0.013
        'W2V_ITER': 32, # for local dataset 32, for mimic 32
        'W2V_EMBEDDING_DIM': 64, # for local dataset 128, for mimic 64
        'W2V_HS': 0, # for local dataset 0, for mimic 0
        'W2V_SG': 1, # for local dataset 0, for mimic 1
        'W2V_MIN_COUNT': 5,
        # Exports only in a single training run, no effect in cross-validation
        'EXPORT_W2V_EMBEDDINGS': False,

        # Profile state encoder (PSE) parameters
        'USE_LSI': False,  # False: encode profile state as multi-hot. True: perform Tf-idf then tsvd on the profile state
        'TSVD_N_COMPONENTS': 200,  # Number of components on the lsi-transformed profile state

        # Neural network parameters
        # Number of additional LSTM layers (minimum 2 not included in this count)
        'N_LSTM': 0,
        # Number of additional batchnorm/dense/dropout layers after PSE before concat (minimum 1 not included in this count)
        'N_PSE_DENSE': 0,
        # Number of batchnorm/dense/dropout layers after concatenation (minimum 1 not included in this count)
        'N_DENSE': 2,
        'LSTM_SIZE': 128, # 512 for retrospective, 128 for prospective
        'DENSE_PSE_SIZE': 128, # 256 for retrospective, 128 for prospective
        'CONCAT_LSTM_SIZE': 512, # 512 for retrospective, irrelevant for prospective
        'CONCAT_TOTAL_SIZE': 256, # 512 for retrospective, 256 for prospective
        'DENSE_SIZE': 256, # 128 for retrospective, 256 for prospective
        'DROPOUT': 0.2, # 0.3 for retrospective, 0.2 for prospective
        'L2_REG': 0,
        'SEQUENCE_LENGTH': 30,

        # Neural network training parameters,
        'BATCH_SIZE': 256,
        'MAX_TRAINING_EPOCHS':1000, # Default 1000, should never get there, reduce for faster execution when testing or debugging.
        'SINGLE_RUN_EPOCHS':16, # How many epochs to train when doing a single run without validation
        'LEARNING_RATE_SCHEDULE':{14:1e-4}, # Dict where keys are epoch index (epoch - 1) where the learning rate decreases and values are the new learning rate.
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

d = data(param.DATA_DIR, param.MODE, param.KEEP_TIME_ORDER)

if os.path.isfile(os.path.join(save_path, 'sampled_encs.pkl')):
    enc_file = os.path.join(save_path, 'sampled_encs.pkl')
    print('Loaded partially completed experiment was done with RESTRICTED DATA !')
else:
    enc_file = False

if new_model:
    d.load_data(restrict_data=param.RESTRICT_DATA, save_path=save_path,
                restrict_sample_size=param.RESTRICT_SAMPLE_SIZE)
else:
    d.load_data(previous_encs_path=enc_file)

# %% [markdown]
# Split encounters into a train and test set, if doing cross-validation
# prepare the appropriate number of folds

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

    profiles_train, targets_train, pre_seq_train, post_seq_train, active_meds_train, active_classes_train, depa_train, targets_test, pre_seq_test, post_seq_test, active_meds_test, active_classes_test, depa_test, definitions = d.make_lists(get_valid=get_valid,
        cross_val_fold=cross_val_fold)

    # Try loading previously fitted transformation pipelines. If they do not exist or do not
    # correspond to the current fold, build and fit new pipelines for word2vec embeddings,
    # profile state encoder and label encoder.
    tp = transformation_pipelines()

    # Word2vec embeddings
    # Create a word2vec pipeline and train word2vec embeddings in that pipeline
    # on the training set profiles. Optionnaly export word2vec embeddings.
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
        pse_data = tp.prepare_pse_data(
            active_meds_train, active_classes_train, depa_train)
        tp.define_pse_pipeline(use_lsi=param.USE_LSI,
                               tsvd_n_components=param.TSVD_N_COMPONENTS)
        pse, pse_shape = tp.fitsave_pse_pipeline(save_path, pse_data, i)

    # Profile state encoder (PSE)
    # Encode the profile state, either as a multi-hot vector (binary count vectorizer) or using Latent Semantic Indexing
    try:
        output_n_classes, n_fold, le = joblib.load(
            os.path.join(save_path, 'le.joblib'))
        assert n_fold == i
        print('Successfully loaded label encoder for current fold.')
    except:
        print('Could not load label encoder for current fold...')
        le, output_n_classes = tp.fitsave_labelencoder(
            save_path, targets_train, i)

    # Neural network

    # Build the generators, prepare the variables for fitting
    w2v_step = w2v.named_steps['w2v']
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

    # Define the callbacks
    n = neural_network(param.MODE)
    if param.CROSS_VALIDATE:
        callback_mode = 'cross_val'
    elif param.VALIDATE:
        callback_mode = 'train_with_valid'
    else:
        callback_mode = 'train_no_valid'
    callbacks = n.callbacks(save_path, i, callback_mode=callback_mode, learning_rate_schedule=param.LEARNING_RATE_SCHEDULE)

    # Try loading a partially trained neural network for current fold,
    # or define a new neural network
    try:
        model = tf.keras.models.load_model(os.path.join(save_path, 'partially_trained_model_{}.h5'.format(i)), custom_objects={
            'sparse_top10_accuracy': n.sparse_top10_accuracy, 'sparse_top30_accuracy': n.sparse_top30_accuracy})
    except:
        model = n.define_model(param.LSTM_SIZE, param.N_LSTM, param.DENSE_PSE_SIZE, param.CONCAT_LSTM_SIZE, param.CONCAT_TOTAL_SIZE, param.DENSE_SIZE,
                               param.DROPOUT, param.L2_REG, param.SEQUENCE_LENGTH, param.W2V_EMBEDDING_DIM, pse_shape, param.N_PSE_DENSE, param.N_DENSE, output_n_classes)
        if (param.CROSS_VALIDATE == True and i == 0) or (param.CROSS_VALIDATE == False):
            tf.keras.utils.plot_model(
                model, to_file=os.path.join(save_path, 'model.png'))

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
    cv_results_df = pd.read_csv(os.path.join(save_path, 'cv_results.csv'))
    v.plot_crossval_accuracy_history(cv_results_df, save_path)
    v.plot_crossval_loss_history(cv_results_df, save_path)
elif param.VALIDATE:
    history_df = pd.read_csv(os.path.join(save_path, 'training_history.csv'))
    v.plot_accuracy_history(history_df, save_path)
    v.plot_loss_history(history_df, save_path)

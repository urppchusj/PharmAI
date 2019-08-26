# %% [markdown]
# # Start training the model

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
CROSS_VALIDATE = True
N_CROSS_VALIDATION_FOLDS = 5

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
DENSE_PSE_SIZE = 256
CONCAT_LSTM_SIZE = 512
CONCAT_TOTAL_SIZE = 512
DENSE_SIZE = 128
DROPOUT = 0.3
L2_REG = 0
SEQUENCE_LENGTH = 30
BATCH_SIZE = 256

N_TRAINING_STEPS_PER_EPOCH = 1000
N_VALIDATION_STEPS_PER_EPOCH = 1000

# %%
# Save the parameters
param_dict = dict(((k,eval(k)) for k in ('MODE', 'CROSS_VALIDATE', 'N_CROSS_VALIDATION_FOLDS', 'RESTRICT_DATA', 'RESTRICT_SAMPLE_SIZE', 'DATA_DIR', 'W2V_ALPHA', 'W2V_ITER', 'W2V_EMBEDDING_DIM', 'W2V_HS', 'W2V_SG', 'W2V_MIN_COUNT', 'W2V_WORKERS', 'EXPORT_W2V_EMBEDDINGS', 'USE_LSI', 'TSVD_N_COMPONENTS', 'N_LSTM', 'N_PSE_DENSE', 'N_DENSE', 'LSTM_SIZE', 'DENSE_PSE_SIZE', 'CONCAT_LSTM_SIZE', 'CONCAT_TOTAL_SIZE', 'DENSE_SIZE', 'DROPOUT', 'L2_REG', 'SEQUENCE_LENGTH', 'BATCH_SIZE', 'N_TRAINING_STEPS_PER_EPOCH', 'N_VALIDATION_STEPS_PER_EPOCH')))

with open(os.path.join(SAVE_PATH, 'parameters.txt'), mode='w') as file:
    for k, v in param_dict.items():
        file.write(str(k) + ': ' + str(v) + '\n')

# %%
# Check if running inside Jupyter notebook or not (will be used later for Keras progress bars)

in_ipynb = check_ipynb().is_inipynb()

# %% [markdown]
# Save hyperparameters

with open(os.path.join(SAVE_PATH, 'hp.pkl'), mode='wb') as file:
    pickle.dump((MODE, CROSS_VALIDATE, N_CROSS_VALIDATION_FOLDS, DATA_DIR, W2V_ALPHA, W2V_ITER, W2V_EMBEDDING_DIM, W2V_HS, W2V_SG, W2V_MIN_COUNT, EXPORT_W2V_EMBEDDINGS, USE_LSI, TSVD_N_COMPONENTS, N_LSTM, N_PSE_DENSE, N_DENSE, LSTM_SIZE, DENSE_PSE_SIZE, CONCAT_LSTM_SIZE, CONCAT_TOTAL_SIZE, DENSE_SIZE, DROPOUT, L2_REG, BATCH_SIZE, SEQUENCE_LENGTH, N_TRAINING_STEPS_PER_EPOCH, N_VALIDATION_STEPS_PER_EPOCH), file)

# %% [markdown]
#  ## Data
# Load the data

# %%
d = data(DATA_DIR, mode=MODE)
d.load_data(restrict_data=RESTRICT_DATA, save_path=SAVE_PATH,
            restrict_sample_size=RESTRICT_SAMPLE_SIZE)

# %% [markdown]
# Split encounters into a train and test set, if doing cross-validation
# prepare the appropriate number of folds

if CROSS_VALIDATE:
    d.cross_val_split(N_CROSS_VALIDATION_FOLDS)
else:
    d.split()

# %% [markdown]
# ## Training execution
# Executed within a loop for cross-validationm if not cross-validating
# will run the loop only one time.

# %%
if CROSS_VALIDATE:
    loop_iters = N_CROSS_VALIDATION_FOLDS
else:
    loop_iters = 1

# %% [markdown]
# ## Loop

# %%
for i in range(loop_iters):
    
    # Make the data lists
    if CROSS_VALIDATE:
        cross_val_fold = i
        print('Performing cross-validation, fold: {}'.format(i))
    else:
        cross_val_fold = None
    profiles_train, targets_train, pre_seq_train, post_seq_train, active_meds_train, active_classes_train, depa_train, targets_test, pre_seq_test, post_seq_test, active_meds_test, active_classes_test, depa_test, definitions = d.make_lists(cross_val_fold=cross_val_fold)

    # Build and fit the data transformation pipelines for word2vec embeddings,
    # profile state encoder and label encoder.
    tp = transformation_pipelines()
    
    # Word2vec embeddings
    # Create a word2vec pipeline and train word2vec embeddings in that pipeline
    # on the training set profiles. Optionnaly export word2vec embeddings.
    tp.define_w2v_pipeline(W2V_ALPHA, W2V_ITER, W2V_EMBEDDING_DIM,
                       W2V_HS, W2V_SG, W2V_MIN_COUNT, W2V_WORKERS)
    w2v = tp.fitsave_w2v_pipeline(SAVE_PATH, profiles_train, W2V_EMBEDDING_DIM, i)
    if CROSS_VALIDATE == False and EXPORT_W2V_EMBEDDINGS == True:
        tp.export_w2v_embeddings(SAVE_PATH, definitions_dict=definitions)


    # Profile state encoder (PSE)
    # Encode the profile state, either as a multi-hot vector (binary count vectorizer) or using Latent Semantic Indexing
    pse_data = tp.prepare_pse_data(
        active_meds_train, active_classes_train, depa_train)
    tp.define_pse_pipeline(use_lsi=USE_LSI, tsvd_n_components=TSVD_N_COMPONENTS)
    pse, pse_shape = tp.fitsave_pse_pipeline(SAVE_PATH, pse_data, i)

    # Label encoder
    le, output_n_classes = tp.fitsave_labelencoder(SAVE_PATH, targets_train, i)

    # Neural network
    # Train a neural network to predict each drug present in a pharmacological
    # profile from the sequence of drug orders that came before or after it
    # and the profile state excluding that drug.

    # Build the generators
    w2v_step = w2v.named_steps['w2v']
    train_generator = TransformedGenerator(MODE, w2v_step, USE_LSI, pse, le, targets_train, pre_seq_train, post_seq_train,
                                        active_meds_train, active_classes_train, depa_train, W2V_EMBEDDING_DIM, SEQUENCE_LENGTH, BATCH_SIZE)

    test_generator = TransformedGenerator(MODE, w2v_step, USE_LSI, pse, le, targets_test, pre_seq_test, post_seq_test,
                                        active_meds_test, active_classes_test, depa_test, W2V_EMBEDDING_DIM, SEQUENCE_LENGTH, BATCH_SIZE, shuffle=False)

    # Define the callbacks
    n = neural_network(MODE)
    callbacks = n.callbacks(SAVE_PATH)

    # Build the model
    model = n.define_model(LSTM_SIZE, N_LSTM, DENSE_PSE_SIZE, CONCAT_LSTM_SIZE, CONCAT_TOTAL_SIZE, DENSE_SIZE,
                        DROPOUT, L2_REG, SEQUENCE_LENGTH, W2V_EMBEDDING_DIM, pse_shape, N_PSE_DENSE, N_DENSE, output_n_classes)
    if (CROSS_VALIDATE == True and i == 0) or (CROSS_VALIDATE == False):
        tf.keras.utils.plot_model(model, to_file=os.path.join(SAVE_PATH, 'model.png'))

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
                        validation_steps=N_VALIDATION_STEPS_PER_EPOCH,
                        verbose=verbose)
    
    # If doing cross-validation, get the metrics for the best
    # epoch.
    if CROSS_VALIDATE:
        train_results = model.evaluate_generator(train_generator, verbose=verbose)
        val_results = model.evaluate_generator(test_generator, verbose=verbose)

        # Make a list of validation metrics names
        val_metrics = ['val_' + metric_name for metric_name in model.metrics_names]

        # Concatenate all results in a list
        all_results = [*train_results, *val_results]
        all_metric_names = [*model.metrics_names, *val_metrics]

        # make a dataframe with the fold metrics
        fold_results_df = pd.DataFrame.from_dict({i:dict(zip(all_metric_names, all_results))}, orient='index')
        # save the dataframe to csv file, create new file at first fold, else append
        if i == 0:
            fold_results_df.to_csv(os.path.join(SAVE_PATH, 'cv_results.csv'))
        else:
            fold_results_df.to_csv(os.path.join(SAVE_PATH, 'cv_results.csv'), mode='a', header=False)
        # Save that the fold completed successfully
        with open(os.path.join(SAVE_PATH, 'done_crossval_folds.pkl'), mode='wb') as file:
            pickle.dump(i, file)
    # Else save the model
    else:
        model.save(os.path.join(SAVE_PATH, 'model.h5'))

# %%
# Plot the loss and accuracy during training

v = visualization()

# If cross-validating, plot evaluation metrics (best epoch)
# by fold. Else, plot training history by epoch.
if CROSS_VALIDATE:
    cv_results_df = pd.read_csv(os.path.join(SAVE_PATH, 'cv_results.csv'))
    v.plot_crossval_accuracy_history(cv_results_df, SAVE_PATH)
    v.plot_crossval_loss_history(cv_results_df, SAVE_PATH)
else:
    history_df = pd.read_csv(os.path.join(SAVE_PATH, 'training_history.csv'))
    v.plot_accuracy_history(history_df, SAVE_PATH)
    v.plot_loss_history(history_df, SAVE_PATH)




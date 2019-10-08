#%%[markdown]
# # Extract the drug information from the MIMIC PRESCRIPTIONS table
#
# This is not actually used anywhere, this is used only to get a human
# readable extraction of drug names as they will be generated
# during preprocessing by the mimic preprocessor.

#%%[narkdown]
#
# ## Imports

#%%
import os
import pathlib

import numpy as np
import pandas as pd

#%%[markdown]
# ## Global variables

#%%
SAVE_PATH = os.path.join(os.getcwd(), 'mimic', 'data')

profile_dtypes = {'ROW_ID':np.int32, 'SUBJECT_ID':str, 'HADM_ID':str, 'ICUSTAY_ID':str, 'STARTDATE':str, 'ENDDATE':str, 'DRUG_TYPE':str, 'DRUG':str, 'DRUG_NAME_POE':str, 'DRUG_NAME_GENERIC':str, 'FORMULARY_DRUG_CD':str, 'GSN':str, 'NDC':str, 'PROD_STRENGTH':str, 'DOSE_VAL_RX':str, 'FORM_VAL_DISP':str, 'FORM_UNIT_DISP':str, 'ROUTE':str}

#%%[markdown]
# ## Load the table

#%%
data = pd.read_csv('mimic/data/PRESCRIPTIONS.csv', index_col='ROW_ID', dtype=profile_dtypes)

#%%[markdown]
# ## Filter out unneeded columns

#%%
data = data.loc[data['DRUG_TYPE']=='MAIN'].copy()
data = data[['DRUG', 'PROD_STRENGTH', 'FORMULARY_DRUG_CD']].drop_duplicates()

#%%[markdown]
# ## Concatenate two columns into a single name per drug

#%%
data['DRUG_EXACT_NAME'] = data['DRUG'] + ' ' + data['PROD_STRENGTH']
data['DRUG_EXACT_NAME'] = data['DRUG_EXACT_NAME'].str.replace(' ', '_')

#%%[markdown]
# ## Save the results as a definitions file for use by the preprocessor

#%%
data.to_csv(os.path.join(SAVE_PATH, 'definitions.csv'), columns=['FORMULARY_DRUG_CD', 'DRUG_EXACT_NAME'])

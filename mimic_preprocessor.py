import argparse as ap
import logging
import os
import pathlib
import pickle
from collections import defaultdict
from datetime import datetime
from itertools import chain

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


class preprocessor():

    def __init__(self, mode, logging_level=logging.DEBUG):

        # Settings
        self.mode = mode
        # Where the preprocessed files will be saved
        self.data_save_path = os.path.join(
            os.getcwd(), 'preprocessed_data', mode, 'mimic')
        # dtypes of data file columns
        profile_dtypes = {'ROW_ID': np.int32, 'SUBJECT_ID': str, 'HADM_ID': str, 'ICUSTAY_ID': str, 'STARTDATE': str, 'ENDDATE': str, 'DRUG_TYPE': str, 'DRUG': str, 'DRUG_NAME_POE': str, 'DRUG_NAME_GENERIC': str, 'FORMULARY_DRUG_CD': str,
        'GSN': str, 'NDC': str, 'PROD_STRENGTH': str, 'DOSE_VAL_RX': str, 'FORM_VAL_DISP': str, 'FORM_UNIT_DISP': str, 'ROUTE': str}
        services_dtypes = {'ROW_ID': np.int32, 'SUBJECT_ID': str, 'HADM_ID': str,
            'TRANSFERTIME': str, 'PREV_SERVICE': str, 'CURR_SERVICE': str}
        admissions_dtypes = {'ROW_ID': np.int32,
            'SUBJECT_ID': str, 'HADM_ID': str, 'ADMITTIME': str}

        # Congigure logger
        print('Configuring logger...')
        self.logging_path = os.path.join(
            os.getcwd(), 'logs', 'preprocessing', self.mode)
        pathlib.Path(self.logging_path).mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging_level,
            format="%(asctime)s [%(levelname)s]  %(message)s",
            handlers=[
                    logging.FileHandler(os.path.join(
                        self.logging_path, datetime.now().strftime('%Y%m%d-%H%M') + 'mimic.log')),
                    logging.StreamHandler()
                ])
        logging.debug('Logger successfully configured.')

        # Load raw data
        logging.info('Loading data...')
        self.raw_profile_data = pd.read_csv(
            'mimic_data/PRESCRIPTIONS.csv', index_col='ROW_ID', dtype=profile_dtypes)
        depa_data = pd.read_csv('mimic_data/SERVICES.csv',
                                index_col='ROW_ID', dtype=services_dtypes)
        depa_data = depa_data[['HADM_ID', 'TRANSFERTIME', 'CURR_SERVICE']]
        admit_data = pd.read_csv(
            'mimic_data/ADMISSIONS.csv', index_col='ROW_ID', dtype=admissions_dtypes)
        admit_data = admit_data[['HADM_ID', 'ADMITTIME']]

        # Calculate synthetic features

        logging.info('Calculating synthetic features...')
        # Keep only drugs, remove IV fluids
        self.raw_profile_data = self.raw_profile_data.loc[self.raw_profile_data['DRUG_TYPE'] == 'MAIN'].copy(
        )
        # Convert datetime string in services to actual datetimes and sort.
        # Here we normalize the transfer datetimes (set all time parts to
        # 00:00:00) because in MIMIC drug start and end times are all at
        # 00:00:00. This introduces noise in the data but is necessary because
        # it is impossible to know when the drug was prescribed in relation
        # to the transfer. We approximate that all orders were prescribed on
        # the last department the patient was on that day.
        depa_data['TRANSFER_DATETIME'] = pd.to_datetime(
            depa_data['TRANSFERTIME'], format='%Y%m%d %H:%M:%S')
        depa_data['TRANSFER_DATETIME'] = depa_data['TRANSFER_DATETIME'].dt.normalize()
        depa_data = depa_data.drop(['TRANSFERTIME'], axis=1)
        depa_data.sort_values(
            ['TRANSFER_DATETIME', 'HADM_ID'], ascending=True, inplace=True)
        # Convert datetime string in admissions to actual datetimes and sort
        admit_data['datetime_begenc'] = pd.to_datetime(
            admit_data['ADMITTIME'], format='%Y%m%d %H:%M:%S')
        admit_data = admit_data.drop(['ADMITTIME'], axis=1)
        admit_data.sort_values(
            ['HADM_ID', 'datetime_begenc'], ascending=True, inplace=True)
        # Convert datetime strings in prescriptions to actual datetimes and sort
        self.raw_profile_data['datetime_beg'] = pd.to_datetime(
            self.raw_profile_data['STARTDATE'], format='%Y%m%d %H:%M:%S')
        self.raw_profile_data = self.raw_profile_data.drop(
            ['STARTDATE'], axis=1)
        self.raw_profile_data['datetime_end'] = pd.to_datetime(
            self.raw_profile_data['ENDDATE'], format='%Y%m%d %H:%M:%S')
        self.raw_profile_data = self.raw_profile_data.drop(['ENDDATE'], axis=1)
        self.raw_profile_data = self.raw_profile_data.sort_values(
            ['datetime_beg', 'HADM_ID'])
        # Add admit start times to prescriptions data
        self.raw_profile_data = self.raw_profile_data.join(
            admit_data.set_index('HADM_ID'), on='HADM_ID')
        # Drop empty formulary drug code (n=1928) datetime_beg (n=3181), datetime_end (n=5151)
        self.raw_profile_data.dropna(
            subset=['FORMULARY_DRUG_CD', 'datetime_beg', 'datetime_end'], inplace=True)

        # Add service at time of prescription to prescription data
        self.raw_profile_data = pd.merge_asof(
            self.raw_profile_data, depa_data, left_on='datetime_beg', right_on='TRANSFER_DATETIME', by='HADM_ID')
        # Rename HADM_ID to enc for compatibility with rest of preprocessor and convert to int
        self.raw_profile_data.rename(columns={'HADM_ID': 'enc'}, inplace=True)
        self.raw_profile_data['enc'] = self.raw_profile_data['enc'].astype(
            np.int32)
        # Sort by admit datetime, then encounter, then prescription datetime
        self.raw_profile_data.sort_values(
            ['datetime_begenc', 'enc', 'datetime_beg'], ascending=True, inplace=True)

        # Fun part begins, create the addition number to sequentially number prescriptions
        self.raw_profile_data['addition_number'] = self.raw_profile_data.groupby(
            'enc').enc.rank(method='first').astype(int)
        # Move that to index
        self.raw_profile_data.set_index(
            ['enc', 'addition_number'], drop=True, inplace=True)

    def get_profiles(self):
        # Rebuild profiles at every addition
        logging.info('Recreating profiles... (takes a while)')
        profiles_dict = defaultdict(list)
        targets_dict = defaultdict(list)
        pre_seq_dict = defaultdict(list)
        post_seq_dict = defaultdict(list)
        active_profiles_dict = defaultdict(list)
        depa_dict = defaultdict(list)
        enc_list = []
        # Prepare a variable of the number of encounters in the dataset
        length = self.raw_profile_data.index.get_level_values(0).nunique()
        # Iterate over encounters, send each encounter to self.build_enc_profiles
        for n, enc in zip(range(0, length), self.raw_profile_data.groupby(level='enc', sort=False)):
            enc_list.append(enc[0])
            profiles_dict[enc[0]] = enc[1]['FORMULARY_DRUG_CD'].tolist()
            enc_profiles = self.build_enc_profiles(enc)
            # Convert each profile to list
            for profile in enc_profiles.groupby(level='profile', sort=False):
                logging.info('Handling encounter number {} profile number {}: {:.2f} %\r'.format(
                    enc[0], profile[0], 100*n / length))
                targets_to_append_list, pre_seq_to_append_list, post_seq_to_append_list, active_profile_to_append_list, depa_to_append_list = self.make_profile_lists(
                    profile)
                targets_dict[enc[0]].extend(targets_to_append_list)
                pre_seq_dict[enc[0]].extend(pre_seq_to_append_list)
                post_seq_dict[enc[0]].extend(post_seq_to_append_list)
                active_profiles_dict[enc[0]].extend(
                    active_profile_to_append_list)
                depa_dict[enc[0]].extend(depa_to_append_list)
        logging.info('Done!')
        return profiles_dict, targets_dict, pre_seq_dict, post_seq_dict, active_profiles_dict, depa_dict, enc_list

    def build_enc_profiles(self, enc):
        enc_profiles_list = []
        prev_add_time = enc[1]['datetime_beg'][0]
        max_enc = enc[1].index.get_level_values('addition_number').max()
        # Iterate over additions in the encounter
        for addition in enc[1].itertuples():
            # For each addition, generate a profile of all medications with a datetime of beginning
            # before or at the same time of the addition
            if self.mode == 'retrospective':
                # In retrospective mode, generate profiles only when no drug
                # was added for 1 hour, representing a "stable" profile for
                # retrospective analysis of all drugs in the profile
                cur_add_time = addition.datetime_beg
                if addition.Index[1] == max_enc:
                    pass
                # Here the offset is set at 1 hour but since every order is at the same time
                # any offset less than 24 hours would work.
                elif cur_add_time < prev_add_time + pd.DateOffset(hours=1):
                    continue
            profile_at_time = enc[1].loc[(enc[1]['datetime_beg'] <= addition.datetime_beg)].copy()
            # Determine if each medication was active at the time of addition.
            # Here this is done differently than in the original approach. In
            # the original approach we used > as an operator. Here because
            # the time is always 00:00:00 we use >= . This approximates that
            # all orders that were prescribed on that day were prescribed at
            # once at the beginning of the day and all discontinuations occured
            # at once at the end of the day. This is obviously inaccurate and
            # introduces noise but it cannot be done more precisely on this
            # dataset.
            profile_at_time['active'] = np.where(
                profile_at_time['datetime_end'] >= addition.datetime_beg, 1, 0)
            # Manipulate indexes to have three levels: encounter, profile and addition
            profile_at_time['profile'] = addition.Index[1]
            profile_at_time.set_index('profile', inplace=True, append=True)
            profile_at_time = profile_at_time.swaplevel(
                i='profile', j='addition_number')
            enc_profiles_list.append(profile_at_time)
            # Used by retrospective mode to calculate how much time elapsed since last addition.
            if self.mode == 'retrospective':
                prev_add_time = cur_add_time
        enc_profiles = pd.concat(enc_profiles_list)
        return enc_profiles

    def make_profile_lists(self, profile):
        targets_list = []
        pre_seq_list = []
        post_seq_list = []
        active_profile_to_append_list = []
        depa_to_append_list = []
        profile_list = profile[1]['FORMULARY_DRUG_CD'].tolist()
        if self.mode == 'retrospective':
            for target_med in profile[1].itertuples():
                # make a list with all medications in profile
                if target_med.active == 0:
                    continue
                mask = profile[1].index.get_level_values(
                    2) == target_med.Index[2]
                target = profile[1][mask]['FORMULARY_DRUG_CD'].astype(str).values[0]
                target_index = len(profile_list) - 1 - \
                    profile_list[::-1].index(target)
                pre_seq = profile_list[:target_index]
                post_seq = profile_list[target_index+1:]
                # remove row of target from profile
                filtered_profile = profile[1].drop(
                    profile[1].index[target_index])
                # select only active medications and make another list with those
                active_profile = filtered_profile.loc[filtered_profile['FORMULARY_DRUG_CD'] == 1].copy(
                )
                # make sets of contents of active profile to prepare for multi-hot encoding
                active_profile_to_append = active_profile['FORMULARY_DRUG_CD'].tolist()
                depa_to_append = active_profile['CURR_SERVICE'].unique().tolist()
                targets_list.append(target)
                pre_seq_list.append(pre_seq)
                post_seq_list.append(post_seq)
                active_profile_to_append_list.append(active_profile_to_append)
                depa_to_append_list.append(depa_to_append)
        elif self.mode == 'prospective':
            # make a list with all medications in profile
            mask = profile[1].index.get_level_values(
                'profile') == profile[1].index.get_level_values('addition_number')
            target = profile[1][mask]['FORMULARY_DRUG_CD'].astype(str).values[0]
            pre_seq = profile[1]['FORMULARY_DRUG_CD'].tolist()
            target_index = len(pre_seq) - 1 - pre_seq[::-1].index(target)
            pre_seq.pop(target_index)
            # remove row of target from profile
            filtered_profile = profile[1].drop(profile[1].index[target_index])
            # select only active medications and make another list with those
            active_profile = filtered_profile.loc[filtered_profile['active'] == 1].copy(
            )
            # make lists of contents of active profile to prepare for multi-hot encoding
            active_profile_to_append = active_profile['FORMULARY_DRUG_CD'].tolist()
            depa_to_append = active_profile['CURR_SERVICE'].unique().tolist()
            targets_list.append(target)
            pre_seq_list.append(pre_seq)
            active_profile_to_append_list.append(active_profile_to_append)
            depa_to_append_list.append(depa_to_append)
        return targets_list, pre_seq_list, post_seq_list, active_profile_to_append_list, depa_to_append_list


    def preprocess(self):
        # Preprocess the data
        profiles_dict, targets_dict, pre_seq_dict, post_seq_dict, active_meds_dict, depa_dict, enc_list = self.get_profiles()
        # Split the encounters into a train and test set
        enc_train, enc_test = train_test_split(enc_list, test_size=0.2)
        # Split all the dicts according to the encounter split
        profiles_dict_train = {k: v for k,v in profiles_dict.items() if k in enc_train}
        targets_dict_train = {k: v for k,v in targets_dict.items() if k in enc_train}
        pre_seq_dict_train = {k: v for k,v in pre_seq_dict.items() if k in enc_train}
        post_seq_dict_train = {k: v for k,v in post_seq_dict.items() if k in enc_train}
        active_meds_dict_train = {k: v for k,v in active_meds_dict.items() if k in enc_train}
        depa_dict_train = {k: v for k,v in depa_dict.items() if k in enc_train}
        targets_dict_test = {k: v for k,v in targets_dict.items() if k in enc_test}
        pre_seq_dict_test = {k: v for k,v in pre_seq_dict.items() if k in enc_test}
        post_seq_dict_test = {k: v for k,v in post_seq_dict.items() if k in enc_test}
        active_meds_dict_test = {k: v for k,v in active_meds_dict.items() if k in enc_test}
        depa_dict_test = {k: v for k,v in depa_dict.items() if k in enc_test}
        # Save preprocessed data to pickle files
        pathlib.Path(self.data_save_path).mkdir(parents=True, exist_ok=True)
        # Train
        with open(os.path.join(self.data_save_path, 'profiles_list.pkl'), mode='wb') as file:
            pickle.dump(profiles_dict_train, file)
        with open(os.path.join(self.data_save_path, 'targets_list.pkl'), mode='wb') as file:
            pickle.dump(targets_dict_train, file)
        with open(os.path.join(self.data_save_path, 'pre_seq_list.pkl'), mode='wb') as file:
            pickle.dump(pre_seq_dict_train, file)
        with open(os.path.join(self.data_save_path, 'post_seq_list.pkl'), mode='wb') as file:
            pickle.dump(post_seq_dict_train, file)
        with open(os.path.join(self.data_save_path, 'active_meds_list.pkl'), mode='wb') as file:
            pickle.dump(active_meds_dict_train, file)
        with open(os.path.join(self.data_save_path, 'depa_list.pkl'), mode='wb') as file:
            pickle.dump(depa_dict_train, file)
        with open(os.path.join(self.data_save_path, 'enc_list.pkl'), mode='wb') as file:
            pickle.dump(enc_train, file)
        # Test
        with open(os.path.join(self.data_save_path, 'test_targets_list.pkl'), mode='wb') as file:
            pickle.dump(targets_dict_test, file)
        with open(os.path.join(self.data_save_path, 'test_pre_seq_list.pkl'), mode='wb') as file:
            pickle.dump(pre_seq_dict_test, file)
        with open(os.path.join(self.data_save_path, 'test_post_seq_list.pkl'), mode='wb') as file:
            pickle.dump(post_seq_dict_test, file)
        with open(os.path.join(self.data_save_path, 'test_active_meds_list.pkl'), mode='wb') as file:
            pickle.dump(active_meds_dict_test, file)
        with open(os.path.join(self.data_save_path, 'test_depa_list.pkl'), mode='wb') as file:
            pickle.dump(depa_dict_test, file)
        with open(os.path.join(self.data_save_path, 'test_enc_list.pkl'), mode='wb') as file:
            pickle.dump(enc_test, file)


###########
# EXECUTE #
###########

if __name__ == '__main__':
    parser = ap.ArgumentParser(
        description='Preprocess the data extracted from the pharmacy database before input into the machine learning model', formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument('--mode', metavar='Type_String', type=str, nargs='?',
                        help='Preprocessing mode. Use "prospective" to generate the preprocessed data for prediction of the next medication order. Use "retrospective" for preprocessed data for retrospective profile analysis. No default.')

    args = parser.parse_args()
    mode = args.mode

    if mode not in ['prospective', 'retrospective']:
        logging.critical('Mode: {} not implemented. Quitting...'.format(mode))
        quit()

    pp = preprocessor(mode)
    pp.preprocess()

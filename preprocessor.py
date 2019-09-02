import argparse as ap
import os
import pathlib
import pickle
from collections import defaultdict
from datetime import datetime
from itertools import chain

import numpy as np
import pandas as pd
from tqdm import tqdm


class preprocessor():

    def __init__(self, source_file, definitions_file, restrict_data, mode):

        # Settings
        self.mode = mode
        self.data_save_path = os.path.join(
            os.getcwd(), 'preprocessed_data', restrict_data + 'yr_' + self.mode)
        self.profile_col_names = ['enc', 'date_beg', 'time_beg', 'date_end', 'time_end', 'medinb',
                            'dose', 'freq', 'genenb', 'date_begenc', 'date_endenc', 'time_endenc', 'depa', 'protoc', 'tram']
        self.profile_dtypes = {'enc': np.int32, 'date_beg': str, 'time_beg': str, 'date_end': str, 'time_end': str, 'medinb': str, 'dose': np.float32,
                         'freq': np.int32, 'genenb': str, 'date_begenc': str, 'date_endenc': str, 'time_endenc': str, 'depa': str, 'protoc': str, 'tram': str}
        self.definitions_col_names = ['medinb', 'mediname',
                                'genenb', 'genename', 'classnb', 'classname']
        self.definitions_dtypes = {'medinb': np.int32, 'mediname': str,
                             'genenb': str, 'genename': str, 'classnb': str, 'classename': str}

        # Load raw data
        print('Loading data...')
        self.raw_profile_data = pd.read_csv(
            source_file, sep=';', names=self.profile_col_names, index_col=None, dtype=self.profile_dtypes)
        classes_data = pd.read_csv(
            definitions_file, sep=';', names=self.definitions_col_names, index_col=0, dtype=self.definitions_dtypes)

        # Calculate synthetic features
        '''
        Convert medinb from text to int
        Add classes from the definitions file and decompose into 4 class levels
        Convert dates and times from text to datetime
        Calculate addition numbers which be used later for sequence generation
        Drop data that is not useful anymore
        '''
        print('Calculating synthetic features...')
        self.raw_profile_data['medinb_int'] = self.raw_profile_data['medinb'].astype(
            np.int32)
        self.raw_profile_data['classnb'] = self.raw_profile_data['medinb_int'].map(
            classes_data['classnb'])
        del classes_data
        self.raw_profile_data['class1_part'] = self.raw_profile_data['classnb'].str.slice(
            start=0, stop=2).astype(np.int32)
        self.raw_profile_data['class2_part'] = self.raw_profile_data['classnb'].str.slice(
            start=3, stop=5).astype(np.int32)
        self.raw_profile_data['class3_part'] = self.raw_profile_data['classnb'].str.slice(
            start=6, stop=8).astype(np.int32)
        self.raw_profile_data['class4_part'] = self.raw_profile_data['classnb'].str.slice(
            start=9, stop=11).astype(np.int32)
        self.raw_profile_data['class1_whole'] = self.raw_profile_data['classnb'].str.slice(
            start=0, stop=2)
        self.raw_profile_data['class2_whole'] = self.raw_profile_data['classnb'].str.slice(
            start=0, stop=5)
        self.raw_profile_data['class3_whole'] = self.raw_profile_data['classnb'].str.slice(
            start=0, stop=8)
        self.raw_profile_data['class4_whole'] = self.raw_profile_data['classnb'].str.slice(
            start=0, stop=11)
        self.raw_profile_data['datetime_beg'] = pd.to_datetime(
            self.raw_profile_data['date_beg']+' '+self.raw_profile_data['time_beg'], format='%Y%m%d %H:%M')
        self.raw_profile_data = self.raw_profile_data.drop(
            ['date_beg', 'time_beg'], axis=1)
        self.raw_profile_data['datetime_end'] = pd.to_datetime(
            self.raw_profile_data['date_end']+' '+self.raw_profile_data['time_end'], format='%Y%m%d %H:%M')
        self.raw_profile_data = self.raw_profile_data.drop(
            ['date_end', 'time_end'], axis=1)
        self.raw_profile_data['date_begenc'] = pd.to_datetime(
            self.raw_profile_data['date_begenc'], format='%Y%m%d')
        self.raw_profile_data['datetime_endenc'] = pd.to_datetime(
            self.raw_profile_data['date_endenc']+' '+self.raw_profile_data['time_endenc'], format='%Y%m%d %H:%M')
        self.raw_profile_data = self.raw_profile_data.drop(
            ['date_endenc', 'time_endenc'], axis=1)
        self.raw_profile_data.sort_values(['date_begenc', 'enc', 'datetime_beg', 'class1_part',
                                           'class2_part', 'class3_part', 'class4_part'], ascending=True, inplace=True)
        self.raw_profile_data['addition_number'] = self.raw_profile_data.groupby(
            'enc').enc.rank(method='first').astype(int)
        self.raw_profile_data.set_index(
            ['enc', 'addition_number'], drop=True, inplace=True)
        maxyear = max(
            self.raw_profile_data['date_begenc'].apply(lambda x: x.year))
        self.raw_profile_data = self.raw_profile_data.loc[self.raw_profile_data['date_begenc'] > datetime(
            maxyear-int(restrict_data)+1, 1, 1)].copy()

    def get_profiles(self):
        # Rebuild profiles at every addition
        print('Recreating profiles... (takes a while)')
        profiles_dict = defaultdict(list)
        targets_dict = defaultdict(list)
        pre_seq_dict = defaultdict(list)
        post_seq_dict = defaultdict(list)
        active_profiles_dict = defaultdict(list)
        active_classes_dict = defaultdict(list)
        depa_dict = defaultdict(list)
        enc_list = []
        # Iterate over encounters, send each encounter to self.build_enc_profiles
        for enc in tqdm(self.raw_profile_data.groupby(level='enc', sort=False)):
            enc_list.append(enc[0])
            profiles_dict[enc[0]] = enc[1]['medinb'].tolist()
            enc_profiles = self.build_enc_profiles(enc)
            # Convert each profile to list
            for profile in enc_profiles.groupby(level='profile', sort=False):
                targets_to_append_list, pre_seq_to_append_list, post_seq_to_append_list, active_profile_to_append_list, class_1_to_append_list, class_2_to_append_list, class_3_to_append_list, class_4_to_append_list, depa_to_append_list = self.make_profile_lists(
                    profile)
                targets_dict[enc[0]].extend(targets_to_append_list)
                pre_seq_dict[enc[0]].extend(pre_seq_to_append_list)
                post_seq_dict[enc[0]].extend(post_seq_to_append_list)
                active_profiles_dict[enc[0]].extend(
                    active_profile_to_append_list)
                depa_dict[enc[0]].extend(depa_to_append_list)
                for class_1_to_append, class_2_to_append, class_3_to_append, class_4_to_append in zip(class_1_to_append_list, class_2_to_append_list, class_3_to_append_list, class_4_to_append_list):
                    active_classes_dict[enc[0]].append(list(chain.from_iterable(
                        [class_1_to_append, class_2_to_append, class_3_to_append, class_4_to_append])))
        print('Done!')
        return profiles_dict, targets_dict, pre_seq_dict, post_seq_dict, active_profiles_dict, active_classes_dict, depa_dict, enc_list

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
                elif cur_add_time < prev_add_time + pd.DateOffset(hours=1):
                    continue
            profile_at_time = enc[1].loc[(
                enc[1]['datetime_beg'] <= addition.datetime_beg)].copy()
            # Determine if each medication was active at the time of addition
            profile_at_time['active'] = np.where(
                profile_at_time['datetime_end'] > addition.datetime_beg, 1, 0)
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
        class_1_to_append_list = []
        class_2_to_append_list = []
        class_3_to_append_list = []
        class_4_to_append_list = []
        depa_to_append_list = []
        profile_list = profile[1]['medinb'].tolist()
        if self.mode == 'retrospective':
            for target_med in profile[1].itertuples():
                # make a list with all medications in profile
                if target_med.active == 0:
                    continue
                mask = profile[1].index.get_level_values(
                    2) == target_med.Index[2]
                target = profile[1][mask]['medinb'].astype(str).values[0]
                target_index = len(profile_list) - 1 - \
                    profile_list[::-1].index(target)
                pre_seq = profile_list[:target_index]
                post_seq = profile_list[target_index+1:]
                # remove row of target from profile
                filtered_profile = profile[1].drop(
                    profile[1].index[target_index])
                # select only active medications and make another list with those
                active_profile = filtered_profile.loc[filtered_profile['active'] == 1].copy(
                )
                # make sets of contents of active profile to prepare for multi-hot encoding
                active_profile_to_append = active_profile['medinb'].tolist()
                class_1_to_append = active_profile['class1_whole'].tolist()
                class_2_to_append = active_profile['class2_whole'].tolist()
                class_3_to_append = active_profile['class3_whole'].tolist()
                class_4_to_append = active_profile['class4_whole'].tolist()
                depa_to_append = active_profile['depa'].unique().tolist()
                targets_list.append(target)
                pre_seq_list.append(pre_seq)
                post_seq_list.append(post_seq)
                active_profile_to_append_list.append(active_profile_to_append)
                class_1_to_append_list.append(class_1_to_append)
                class_2_to_append_list.append(class_2_to_append)
                class_3_to_append_list.append(class_3_to_append)
                class_4_to_append_list.append(class_4_to_append)
                depa_to_append_list.append(depa_to_append)
        elif self.mode == 'prospective':
            # make a list with all medications in profile
            mask = profile[1].index.get_level_values(
                'profile') == profile[1].index.get_level_values('addition_number')
            target = profile[1][mask]['medinb'].astype(str).values[0]
            pre_seq = profile[1]['medinb'].tolist()
            target_index = len(pre_seq) - 1 - pre_seq[::-1].index(target)
            pre_seq.pop(target_index)
            # remove row of target from profile
            filtered_profile = profile[1].drop(profile[1].index[target_index])
            # select only active medications and make another list with those
            active_profile = filtered_profile.loc[filtered_profile['active'] == 1].copy(
            )
            # make lists of contents of active profile to prepare for multi-hot encoding
            active_profile_to_append = active_profile['medinb'].tolist()
            class_1_to_append = active_profile['class1_whole'].tolist()
            class_2_to_append = active_profile['class2_whole'].tolist()
            class_3_to_append = active_profile['class3_whole'].tolist()
            class_4_to_append = active_profile['class4_whole'].tolist()
            depa_to_append = active_profile['depa'].unique().tolist()
            targets_list.append(target)
            pre_seq_list.append(pre_seq)
            active_profile_to_append_list.append(active_profile_to_append)
            class_1_to_append_list.append(class_1_to_append)
            class_2_to_append_list.append(class_2_to_append)
            class_3_to_append_list.append(class_3_to_append)
            class_4_to_append_list.append(class_4_to_append)
            depa_to_append_list.append(depa_to_append)
        return targets_list, pre_seq_list, post_seq_list, active_profile_to_append_list, class_1_to_append_list, class_2_to_append_list, class_3_to_append_list, class_4_to_append_list, depa_to_append_list

    def preprocess(self):
        # Preprocess the data
        profiles_dict, targets_dict, pre_seq_dict, post_seq_dict, active_profiles_dict, active_classes_dict, depa_dict, enc_list = self.get_profiles()
        # Save preprocessed data to pickle file
        pathlib.Path(self.data_save_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(self.data_save_path, 'targets_list.pkl'), mode='wb') as file:
            pickle.dump(targets_dict, file)
        with open(os.path.join(self.data_save_path, 'profiles_list.pkl'), mode='wb') as file:
            pickle.dump(profiles_dict, file)
        with open(os.path.join(self.data_save_path, 'pre_seq_list.pkl'), mode='wb') as file:
            pickle.dump(pre_seq_dict, file)
        with open(os.path.join(self.data_save_path, 'post_seq_list.pkl'), mode='wb') as file:
            pickle.dump(post_seq_dict, file)
        with open(os.path.join(self.data_save_path, 'active_meds_list.pkl'), mode='wb') as file:
            pickle.dump(active_profiles_dict, file)
        with open(os.path.join(self.data_save_path, 'active_classes_list.pkl'), mode='wb') as file:
            pickle.dump(active_classes_dict, file)
        with open(os.path.join(self.data_save_path, 'depa_list.pkl'), mode='wb') as file:
            pickle.dump(depa_dict, file)
        with open(os.path.join(self.data_save_path, 'enc_list.pkl'), mode='wb') as file:
            pickle.dump(enc_list, file)


###########
# EXECUTE #
###########

if __name__ == '__main__':
    parser = ap.ArgumentParser(
        description='Preprocess the data extracted from the pharmacy database before input into the machine learning model', formatter_class=ap.RawTextHelpFormatter)
    parser.add_argument('--mode', metavar='Type_String', type=str, nargs='?',
                        help='Preprocessing mode. Use "prospective" to generate the preprocessed data for prediction of the next medication order. Use "retrospective" for preprocessed data for retrospective profile analysis. No default.')
    parser.add_argument('--numyears', metavar='Type_String', type=str, nargs="?",
                        default='5', help='Number of years in the data to process. Defaults to 5')
    parser.add_argument('--sourcefile', metavar='Type_String', type=str, nargs="?",
                        default='data/20050101-20180101pet.csv', help='Source file load. Defaults to "data/20050101-20180101pet.csv".')
    parser.add_argument('--definitionsfile', metavar='Type_String', type=str, nargs="?",
                        default='data/definitions.csv', help='Source file load. Defaults to "data/definitions.csv".')

    args = parser.parse_args()
    mode = args.mode
    num_years = args.numyears
    source_file = args.sourcefile
    definitions_file = args.definitionsfile

    if mode not in ['prospective', 'retrospective']:
        print('Mode: {} not implemented. Quitting...'.format(mode))
        quit()
    if not int(num_years):
        print(
            'Argument --numyears {} is not an integer. Quitting...'.format(num_years))
        quit()
    try:
        if(not os.path.isfile(source_file)):
            print(
                'Data file: {} not found. Quitting...'.format(source_file))
            quit()
    except TypeError:
        print('Invalid data file given. Quitting...')
        quit()
    try:
        if(not os.path.isfile(definitions_file)):
            print(
                'Definitions file: {} not found. Quitting...'.format(definitions_file))
            quit()
    except TypeError:
        print('Invalid data file given. Quitting...')
        quit()

    pp = preprocessor(source_file, definitions_file,
                      restrict_data=num_years, mode=mode)
    pp.preprocess()

__author__ = 'Mike'

import os
import numpy as np
import pandas as pd
import random


class Container:

    def __init__(self, path, epoch_size):
        self.n_classes = 2  # TODO: remove deprecated variable
        self.data = {}

        # meta information
        self.epoch_size = epoch_size

        dataset_folders = os.listdir(path)

        for dataset in dataset_folders:
            self.data[dataset] = gather_data(dataset, path, epoch_size, verbose=True)

    def create_features_and_labels(self, channels, test_size=0.1, one_hot=True):
        # TODO: revise
        merged_data = {}

        for dataset in self.data:
            for trial in self.data[dataset].keys():
                new_key = dataset + '_' + str(trial)
                merged_data[new_key] = self.data[dataset][trial]

        return self.prepare_features_and_labels(merged_data, channels, test_size, one_hot)

    def create_features_and_labels_for_dataset(self, dataset, channels, test_size=0.1, one_hot=True):
        # TODO: revise
        return self.prepare_features_and_labels(self.data[dataset], channels, test_size, one_hot)

    def prepare_features_and_labels(self, data, channels, test_size, one_hot):
        # TODO: revise
        # get size of testing set
        testing_size = int(test_size * len(data.keys()))

        # shuffle trial keys
        shuffled_trial_keys = list(data.keys())
        random.shuffle(shuffled_trial_keys)

        # stack labels into nparray
        labels = np.hstack((data[trial][1]['Class'] for trial in shuffled_trial_keys))

        # stack features into nparray
        features = np.vstack((np.hstack(
            (np.dstack((data[trial][time_point][channel] for channel in channels)) for time_point in
             data[trial].keys())) for trial in shuffled_trial_keys))

        if one_hot:
            labels = make_labels_one_hot(labels, self.n_classes)

        train_features = features[:-testing_size]
        train_labels = labels[:-testing_size]

        test_features = features[-testing_size:]
        test_labels = labels[-testing_size:]

        return train_features, train_labels, test_features, test_labels


    def add_constant(self, dataset, name, constant):
        # TODO: revise
        for case in self.data[dataset]:
            for time_point in self.data[dataset][case]:
                self.data[dataset][case][time_point][name] = constant

    def add_variable(self, dataset, df, variable):
        # TODO: revise
        """
        Inserts a variable from a pandas data frame into a dataset of EEG data container.
        :param dataset: Name of dataset from EEG data container
        :param df: Pandas data frame with variable (case in dataset must match index of df)
        :param variable: Name of variable that will be inserted
        :return:
        """
        if len(self.data[dataset]) > len(df):
            raise Exception("Number of entries df must be greater or equal than number of cases in dataset.")

        for case in self.data[dataset]:
            for time_point in self.data[dataset][case]:
                self.data[dataset][case][time_point][variable] = df.loc[case - 1, variable]

    def merge_datasets(self, a, b, name):
        self.data[name] = merge_two_dicts(self.data[a], self.data[b])
        del self.data[a]
        del self.data[b]


def gather_data(dataset, path, epoch_size, verbose=False):
    """
    Reads EEG data epochs from folder that contains one or more files that are treated as one dataset. One dataset
    can comprise one or more files that each contain EEG data of e.g. different trial types of one dataset (i.e. one
    dataset corresponds to one dataset) or different datasets of one experimental condition (i.e. one dataset
    corresponds to one experimental condition).
    :param epoch_size: Number size of epoch
    :param dataset: String ID of dataset
    :param path: Folder path to datasets
    :param begin: Number begin of ERP data stream in a data file
    :param verbose: Boolean defines the verbosity of data reading process
    :return: Dictionary with dataset IDs as keys and collected data from data files as values
    """
    path = os.path.abspath(os.path.join(path, dataset))

    dataset_files = os.listdir(path)

    epochs = pd.DataFrame()

    if verbose:
        print("Dataset ID %s, record files:" % dataset)

    for file in dataset_files:
        if file.endswith(".dat"):

            dataset_id = guess_dataset_id(file)
            target_class = guess_target(file)
            congruency = guess_congruency_info(file)

            file_path = os.path.abspath(os.path.join(path, file))
            epochs_file = read_brainvision_file(file_path)

            if verbose:
                print("> %s, %d trial(s)" % (file, len(epochs_file)/epoch_size))

            if not epochs.empty:
                epochs = pd.concat(epochs, epochs_file, ignore_index=True)
            else:
                epochs = epochs_file

    n_epochs = len(epochs)//epoch_size

    if verbose:
        print("> Total: %d trial(s)" % n_epochs)
    else:
        print("Dataset ID: %s, record files: %d" % (dataset, len(dataset_files)))

    # get hierarchical dataframe with epochs
    epochs_dfs = []
    epochs_index = []

    epochs = epochs.groupby(np.arange(len(epochs))//epoch_size)
    for k, g in epochs:
        g.reset_index(drop=True, inplace=True)
        epochs_dfs.append(g)
        epochs_index.append(k)

    epochs_trials = pd.concat(epochs_dfs, keys=epochs_index)

    return epochs_trials


def guess_target(string):
    """
    Guesses target information from string.
    :param string
    :return: Number code for target class (1 = correct, 0 = error)
    """
    if string[-5] == "e":
        target = 0
    elif string[-5] == "c":
        target = 1
    else:
        target = None
    return target


def guess_congruency_info(string):
    """
    Guesses congruency type from string.
    :param string
    :return: Number code for type of congruency (1 = congruent, 0 = not congruent)
    """
    trial_type = string[-9:-6]
    if trial_type == str(101):
        congruency = 1
    elif trial_type == str(102):
        congruency = 1
    elif trial_type == str(103):
        congruency = 0
    elif trial_type == str(104):
        congruency = 0
    else:
        congruency = None
    return congruency


def guess_dataset_id(string):
    """
    Guesses dataset id from string.
    :param string
    :return: dataset ID
    """

    return string[0:6]


def read_brainvision_file(file):
    """ Reads data from generic data format file (.dat) and creates a pandas dataframe with channels as coumns and
    voltage values in rows.
    :param file: File name of generic data format file
    :return: Dataframe with channels as columns and voltage values in rows
    """
    data_file = open(file, "r")
    epochs = {}
    for line in data_file:

        # separate channel keys and values
        ch_values = line.split(None, 1)
        ch = ch_values[0]
        ch = ch.strip()
        ch = ch.replace(" ", "_")

        # add dictionary entry with channel as key and list of voltage values
        epochs[ch] = ch_values[1].split()

    data_file.close()

    return pd.DataFrame(epochs)


def make_labels_one_hot(labels, n_classes):
    """
    Takes a list of labels and converts them to one hot labels.
    :param labels: List of labels
    :param n_classes: Number of classes
    :return: Numpy array with one hot labels
    """
    labels_one_hot = []
    for label in labels:
        label_one_hot = np.zeros(n_classes)
        label_one_hot[label] = 1
        labels_one_hot.append(label_one_hot)
    labels = np.array(labels_one_hot)
    return labels

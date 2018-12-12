__author__ = 'Mike'

import os
import numpy as np
import random


class Container:

    def __init__(self, path, data_points, begin, n_classes=2):
        self.data = {}
        self.n_classes = n_classes
        dataset_folders = os.listdir(path)

        for dataset in dataset_folders:
            self.data[dataset] = read_eeg_data_from_folder(dataset, path, data_points, begin)


    def create_features_and_labels(self, channels, test_size=0.1, one_hot=True):
        merged_data = {}

        for participant in self.data:
            for trial in self.data[participant].keys():
                new_key = participant + '_' + str(trial)
                merged_data[new_key] = self.data[participant][trial]

        return self.prepare_features_and_labels(merged_data, channels, test_size, one_hot)


    def create_features_and_labels_for_dataset(self, dataset, channels, test_size=0.1, one_hot=True):
        return self.prepare_features_and_labels(self.data[dataset], channels, test_size, one_hot)


    def prepare_features_and_labels(self, data, channels, test_size, one_hot):
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


    def merge_data(self, identifier, data, variables):
        for dataset in self.data:
            for case in self.data[dataset]:
                for time_point in self.data[dataset][case]:
                    for key in data:
                        if self.data[dataset][case][time_point][identifier[0]] in data[key][identifier[1]]:
                            for variable in variables:
                                self.data[dataset][case][time_point][variable] = data[key][variable]


    def add_variable_to_dataset(self, dataset, name, value):
        for case in self.data[dataset]:
            for time_point in self.data[dataset][case]:
                self.data[dataset][case][time_point][name] = value


    def merge_datasets(self, a, b, name):
        self.data[name] = merge_two_dicts(self.data[a], self.data[b])
        del self.data[a]
        del self.data[b]


def read_eeg_data_from_folder(dataset, datasets_path, data_points, begin, verbose=False):
    """
    Reads ERP data epochs from folder that contains one or more files that are treated as one dataset. One dataset
    can comprise one or more files that each contain ERP data of e.g. different trial types of one participant (i.e. one
    dataset corresponds to one participant) or different participants of one experimental condition (i.e. one dataset
    corresponds to one experimental condition).
    :param dataset: String ID of dataset
    :param datasets_path: Folder path to datasets
    :param data_points: Number of data points that are included in one ERP data epoch
    :param begin: Number begin of ERP data stream in a data file
    :param verbose: Boolean defines the verbosity of data reading process
    :return: Dictionary with dataset IDs as keys and collected data from data files as values
    """
    dataset_path = os.path.abspath(os.path.join(datasets_path, dataset))

    dataset_files = os.listdir(dataset_path)

    current_trial_id = 0
    trials_data = {}

    if verbose:
        print("Dataset ID %s, record files:" % dataset)

    for file in dataset_files:
        if file.endswith(".dat"):

            participant_id = get_participant_id(file)
            target_class = get_target_class(file)
            congruency = get_congruency(file)

            file_path = os.path.abspath(os.path.join(dataset_path, file))
            channels_data = read_data(file_path, data_points, begin)
            trials_new_data = restructure_data(channels_data, participant_id, target_class, congruency, current_trial_id)

            current_trial_id = current_trial_id + len(trials_new_data)
            if verbose:
                print(file + ", " + str(len(trials_new_data)) + " trial(s)")

            trials_data.update(trials_new_data)
    if verbose:
        print("Total: %d" % current_trial_id)
    else:
        print("Dataset ID: %s, record files: %d" % (dataset, current_trial_id))

    return trials_data


def get_target_class(filename_in):
    """
    Gets target class of participant data file.
    :param filename_in: File name of generic data format file
    :return: Number code for target class (1 = correct, 0 = error)
    """
    if filename_in[-5] == "e":
        target_class = 0
    elif filename_in[-5] == "c":
        target_class = 1
    else:
        target_class = -99
    return target_class


def get_congruency(filename_in):
    """
    Gets congruency type from context specific stimulus code.
    :param filename_in: File name of generic data format file
    :return: Number code for type of congruency (1 = congruent, 0 = not congruent)
    """
    trial_type = filename_in[-9:-6]
    if trial_type == str(101):
        congruency = 1
    elif trial_type == str(102):
        congruency = 1
    elif trial_type == str(103):
        congruency = 0
    elif trial_type == str(104):
        congruency = 0
    else:
        congruency = -99
    return congruency


def get_participant_id(filename_in):
    """
    Gets participant id from context specific stimulus code.
    :param filename_in: File name of generic data format file
    :return: Participant ID
    """

    return filename_in[0:8]


def read_data(file_path, data_points, begin):
    """ Reads data from generic data format file (.dat) and creates a dictionary with channel name as key and
    a list of 150 scan points big chunks as value.
    :param file_name: File name of generic data format file
    :param path: Path to generic data format file
    :param begin:
    :return: Dictionary with channel name as key and list of 150 scan points big chunks which is the
    size of a trial as value
    """
    data_file = open(file_path, "r")
    channels = {}
    for line in data_file:
        ch = line[:begin]
        ch = ch.strip()
        ch = ch.replace(" ", "_")

        values = line[begin:].split()

        if not 'hEOG' in ch:
            channels[ch] = list(chunks(values, data_points))
    data_file.close()

    return channels


def restructure_data(trial_data_wide, dataset_id, target_class, congruency, current_trial_id):
    """
    Converts wide and generic data format read from generic data format file from brainvision analyzer to a long data format
    where each line represents a time point of a trial and columns are channel data.
    :param trial_data_wide: Dictionary with channel name as key and list of trials
    :param dataset_id: String ID of the dataset
    :param target_class: Number code for target class (1 = correct, 0 = error)
    :param congruency: Number code for type of congruency (1 = congruent, 0 = not congruent)
    :param current_trial_id: Number ID of the current trial of the dataset that can be used when dataset has more than one trial
    :return: Dictionary with trial number (as key) and dictionaries (as values) with time point (as key) and
    specific attributes such as trial number, time point, class, congruency types and channels (as values)
    """
    trial_data_long = {}
    for ch in trial_data_wide.keys():

        i = 1

        for trial in trial_data_wide[ch]:
            # check if there is already an entry for the next trial in trials_tmp_data
            if (current_trial_id + i) not in trial_data_long:
                trial_data_long[current_trial_id + i] = {}

            j = 1

            for time_point in trial:
                if j not in trial_data_long[current_trial_id + i]:
                    trial_data_long[current_trial_id + i][j] = {}
                    trial_data_long[current_trial_id + i][j]['ID'] = current_trial_id + i
                    trial_data_long[current_trial_id + i][j]['Time_point'] = j
                    trial_data_long[current_trial_id + i][j]['Class'] = target_class
                    trial_data_long[current_trial_id + i][j]['Congruency'] = congruency
                    trial_data_long[current_trial_id + i][j]['Participant_ID'] = dataset_id

                time_point = time_point.replace(",", ".")
                trial_data_long[current_trial_id + i][j][ch] = time_point

                j = j + 1

            i = i + 1

    return trial_data_long


def print_trials_data(trials_data):
    """
    Prints ERP data.
    :param trials_data: Dictionary of ERP data.
    """
    for trial in trials_data.keys():
        print(str(trial) + ": ")
        for time_point, ch in trials_data[trial].iteritems():
            print(time_point, ch)


def chunks(length, size):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(length), size):
        yield length[i:i + size]


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


def merge_two_dicts(x, y):
    """
    Merges two dictionaries to one.
    :param x: Dictionary 1
    :param y: Dictionary 2
    :return: Combined dictionary of dictionaries 1 and 2
    """
    z = x.copy()
    z.update(y)
    return z
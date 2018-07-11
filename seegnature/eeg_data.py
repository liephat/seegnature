__author__ = 'Mike'

import os
import numpy as np
import random


class Container:
    data = {}

    def __init__(self, path, data_points, n_classes=2):
        self.n_classes = n_classes
        dataset_folders = os.listdir(path)

        for dataset in dataset_folders:
            self.data[dataset] = read_eeg_data_from_folder(dataset, path, data_points)


    def create_features_and_labels(self, channels, test_size=0.1, one_hot=True):
        merged_data = {}

        for participant in self.data:
            for trial in self.data[participant].keys():
                new_key = participant + '_' + str(trial)
                merged_data[new_key] = self.data[participant][trial]

        return self.prepare_features_and_labels(merged_data, channels, test_size, one_hot)


    def create_features_and_labels_for_participant(self, participant, channels, test_size=0.1, one_hot=True):
        return self.prepare_features_and_labels(self.data[participant], channels, test_size, one_hot)


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

    def add_participant_data(self, subject_data, variables):
        for participant in self.data:
            for trial in self.data[participant]:
                for time_point in self.data[participant][trial]:
                    for key in subject_data:
                        if self.data[participant][trial][time_point]['Participant_ID'] in subject_data[key]['Subject_ID']:
                            for variable in variables:
                                self.data[participant][trial][time_point][variable] = subject_data[key][variable]


def read_eeg_data_from_folder(dataset, datasets_path, data_points):
    dataset_path = os.path.abspath(os.path.join(datasets_path, dataset))

    dataset_files = os.listdir(dataset_path)

    current_trial_no = 0
    trials_data = {}
    print("Dataset ID " + dataset + ", record files:")
    for file in dataset_files:
        if (file.endswith(".dat") and "S101" in file) or (
                    file.endswith(".dat") and "S102" in file) or (
                    file.endswith(".dat") and "S103" in file) or (
                    file.endswith(".dat") and "S104" in file):

            participant_id = get_participant_id(file)
            target_class = get_target_class(file)
            congruency = get_congruency(file)

            file_path = os.path.abspath(os.path.join(dataset_path, file))
            channels_data = read_data(file_path, data_points)
            trials_new_data = restructure_data(channels_data, participant_id, target_class, congruency, current_trial_no)

            current_trial_no = current_trial_no + len(trials_new_data)
            print(file + ", " + str(len(trials_new_data)) + " trial(s)")

            trials_data.update(trials_new_data)
    print("Total: " + str(current_trial_no))
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


def read_data(file_path, data_points):
    """ Reads data from generic data format file (.dat) and creates a dictionary with channel name as key and
    a list of 150 scan points big chunks as value.
    :param file_name: File name of generic data format file
    :param path: Path to generic data format file
    :return: Dictionary with channel name as key and list of 150 scan points big chunks which is the
    size of a trial as value
    """
    data_file = open(file_path, "r")
    channels = {}
    for line in data_file:
        ch = line[:6]
        ch = ch.strip()
        ch = ch.replace(" ", "_")

        values = line[6:].split()

        channels[ch] = list(chunks(values, data_points))
    data_file.close()

    return channels


def restructure_data(channels, participant_id, target_class, congruency, current_trial_no):
    """
    Converts wide data format read from generic data format file to a long data format where each line represents a
    time point of a trial and columns are channel data.
    :param channels: Dictionary with channel name as key and list of trials
    :param participant_id: String participant id
    :param target_class: Number code for target class (1 = correct, 0 = error)
    :param congruency: Number code for type of congruency (1 = congruent, 0 = not congruent)
    :param current_trial_no: Number of the current trial of the participant
    :return: Dictionary with trial number (as key) and dictionaries (as values) with time point (as key) and
    specific attributes such as trial number, time point, class, congruency types and channels (as values)
    """
    trials_tmp_data = {}
    for ch in channels.keys():

        i = 1

        for trial in channels[ch]:
            # check if there is already an entry for the next trial in trials_tmp_data
            if (current_trial_no + i) not in trials_tmp_data:
                trials_tmp_data[current_trial_no + i] = {}

            j = 1

            for time_point in trial:
                if j not in trials_tmp_data[current_trial_no + i]:
                    trials_tmp_data[current_trial_no + i][j] = {}
                    trials_tmp_data[current_trial_no + i][j]['Trial'] = current_trial_no + i
                    trials_tmp_data[current_trial_no + i][j]['Time_point'] = j
                    trials_tmp_data[current_trial_no + i][j]['Class'] = target_class
                    trials_tmp_data[current_trial_no + i][j]['Congruency'] = congruency
                    trials_tmp_data[current_trial_no + i][j]['Participant_ID'] = participant_id

                time_point = time_point.replace(",", ".")
                trials_tmp_data[current_trial_no + i][j][ch] = time_point

                j = j + 1

            i = i + 1

    return trials_tmp_data


def print_trials_data(trials_data):
    for trial in trials_data.keys():
        print(str(trial) + ": ")
        for time_point, ch in trials_data[trial].iteritems():
            print(time_point, ch)


def chunks(length, size):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(length), size):
        yield length[i:i + size]


def make_labels_one_hot(labels, n_classes):
    labels_one_hot = []
    for label in labels:
        label_one_hot = np.zeros(n_classes)
        label_one_hot[label] = 1
        labels_one_hot.append(label_one_hot)
    labels = np.array(labels_one_hot)
    return labels
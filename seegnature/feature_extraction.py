__author__ = "Mike"

from collections import OrderedDict
import os
import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class SeparabilityIndex:

    def __init__(self, name, raw_data, channels, target_variable, correlation_type, alpha=False):
        self.data = {}
        self.extracted_features = {}
        self.name = name
        self.raw_data = raw_data
        self.channels = channels
        self.target_variable = target_variable
        self.correlation_type = correlation_type
        SeparabilityIndex.create(self, raw_data, channels, target_variable, correlation_type, alpha)


    def create(self, raw_data, channels, target_variable, correlation_type, alpha):

        # NOTE: The following code is a complicated way to restructure the raw data which was read from brainvision files
        # and create a three-dimensional data structure (x: time, y: channel, z: trials) by help of dictionaries and arrays.
        # The data structure is used to calculate the separability measures for each time and channel combination.
        # Data handling would be much easier when building a three-dimensional data structure by help of one numpy array.
        # TODO: Convert data structure to a numpy array.

        # Initialize empty array for targets and dictionary for spatio temporal data matrix
        targets = np.array([])
        spatio_temporal_data = {}

        # initialize empty dictionaries for each time point with channels as keys and
        # initialiize empty arrays for each channel per time point of spatio temporal data matrix
        for time_point in raw_data[1].keys():
            spatio_temporal_data[time_point] = {}
            for channel in channels:
                spatio_temporal_data[time_point][channel] = np.array([])

        # fill targets array with target data and fill spatio temporal data structure with data from all trials
        for trial in raw_data.keys():
            targets = np.append(targets, raw_data[trial][time_point][target_variable])
            for time_point in raw_data[trial].keys():

                for channel in channels:
                    spatio_temporal_data[time_point][channel] = np.append(spatio_temporal_data[time_point][channel],
                                                                          raw_data[trial][time_point][channel])

        # calculate separability measures for all data points of spatio temporal data matrix
        for time_point in spatio_temporal_data:
            self.data[time_point] = {}

            for channel in spatio_temporal_data[time_point]:
                if correlation_type is 'pointbiserial':
                    r = stats.pointbiserialr(targets.astype(np.float), spatio_temporal_data[time_point][channel].astype(np.float))
                elif correlation_type is 'pearson':
                    r = stats.pearsonr(targets.astype(np.float), spatio_temporal_data[time_point][channel].astype(np.float))
                else:
                    raise ValueError('Not a valid correlation type')

                if alpha is not False:
                    if alpha is True:
                        if r[1] < 0.05:
                            self.data[time_point][channel] = r[0]
                        else:
                            self.data[time_point][channel] = 0
                    else:
                        if r[1] < alpha:
                            self.data[time_point][channel] = r[0]
                        else:
                            self.data[time_point][channel] = 0
                else:
                    self.data[time_point][channel] = r[0]


    def make_heatmap(self, cmap):
        time_points = self.data.keys()
        number_time_points = len(time_points)
        # helper array for pcolormesh
        channel_numbers = []
        i = 1
        for channel in self.channels:
            channel_numbers.append(i)
            i = i + 1

        correlations = []
        for time_point in self.data:
            correlations.append([self.data[time_point][channel] for channel in self.channels])

        # setup the 2D grid with Numpy
        time_points, channel_numbers = np.meshgrid(list(time_points), np.asarray(channel_numbers))

        # convert correlations (list of lists) to a numpy array for plotting
        correlations = np.array(correlations)

        # define channel labels
        channel_labels = []
        # channel_wanted = self.channels
        channel_wanted = ['Fz', 'FCz', 'Cz', 'Pz']
        for channel in self.channels:
            if channel in channel_wanted:
                channel_labels.append(channel)
            else:
                channel_labels.append('')

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.set_yticks(np.arange(1, len(self.channels), 1) + 0.5, minor=False)
        ax.set_yticklabels(channel_labels)
        ax.set_xticks(np.arange(0, number_time_points, 25) + 0.5, minor=False)
        ax.set_xticks(np.arange(0, number_time_points, 5) + 0.5, minor=True)
        ax.set_xticklabels(np.arange(-100, (number_time_points-25)*4, 100))
        ax.set_xlabel("[ms]")
        ax.set_ylabel("channel")
        ax.set_title("SI of " + self.name)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both', which='major', labelsize=12)

        # pick the desired colormap, sensible levels, and define a normalization
        # instance which takes data values and translates those into levels.

        mesh = ax.pcolormesh(time_points, channel_numbers, np.swapaxes(correlations, 0, 1), vmin=-0.45, vmax=0.45, cmap=cmap)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)

        fig.colorbar(mesh, cax=cax)  # need a colorbar to show the intensity scale

        self.heatmap = plt


    def show_heatmap(self, cmap='RdBu_r'):
        self.make_heatmap(cmap)
        self.heatmap.show()


    def save_heatmap(self, path, cmap='RdBu_r'):
        file = os.path.abspath(os.path.join(path, self.name + ".png"))
        self.make_heatmap(cmap)
        self.heatmap.savefig(file, dpi=300, bbox_inches='tight')
        print("Saved separability diagram to " + file)


    def pickle(self, path):
        file = os.path.abspath(os.path.join(path, self.name + '.pkl'))
        with open(file, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)


    def extract_features(self, time_periods, channels, target_variable):

        # transform time period from ms (which is more user-friendly) to scan points (which is the technical representation)
        time_periods = np.around(np.array(time_periods) / 4 + 25)

        for case in self.raw_data:

            self.extracted_features[case] = LastUpdatedOrderedDict()
            self.extracted_features[case]['ID'] = case
            self.extracted_features[case][target_variable] = self.raw_data[case][1][target_variable]

            for time_period in time_periods:
                begin = time_period[0]
                end = time_period[1]
                # get time points within range defined by begin and end
                time_points = np.arange(begin, end + 1, 1)

                for channel in channels:
                    sum = 0.0
                    for time_point in time_points:
                        sum = sum + float(self.raw_data[case][time_point][channel])

                    avg = sum / len(time_points)

                    variable_name = str(channel) + "_" + str(begin) + "-" + str(end)
                    self.extracted_features[case][variable_name] = avg

        # OrderedDict(sorted(d.items(), key=lambda t: t[0]))

        return self.extracted_features


    def get_features_and_labels(self, target_variable=None):

        features = self.extracted_features
        if target_variable is None:
            target_variable = self.target_variable

        if target_variable in features[1].keys():
            # stack labels into nparray
            y = np.hstack((features[case][target_variable] for case in features))
            y = y.astype(np.float64)
            y = y - 1

            # stack features into nparray
            X = np.vstack((np.hstack(features[case][feature] for feature in features[case].keys()) for case in features))
            X = np.delete(X, [0, 1], axis=1)
            X = X.astype(np.float64)

            return X, y
        else:
            raise ValueError('Not a valid target variable')


    def save_extracted_features_to_txt(self, path):

        features = self.extracted_features
        X = np.vstack((np.hstack(features[case][feature] for feature in features[case].keys()) for case in features))
        X = X.astype(np.float32)

        file = os.path.abspath(os.path.join(path, self.name + '.txt'))
        np.savetxt(file, X)


def load_separability_index(directory, name):
    file = os.path.abspath(os.path.join(directory, name + '.pkl'))

    with open(file, 'rb') as pickle_file:
        separability_index = pickle.load(pickle_file)

    print('Separability index %s loaded.' % name)
    return separability_index


class LastUpdatedOrderedDict(OrderedDict):
    'Store items in the order the keys were last added'

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        OrderedDict.__setitem__(self, key, value)

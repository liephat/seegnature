__author__ = "Mike"

from collections import OrderedDict
import os
import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

class SeparabilityIndex:

    def __init__(self, name, raw_data, channels, variable, correlation_type):
        self.data = {}
        self.name = name
        self.raw_data = raw_data
        self.channels = channels
        self.variable = variable
        self.correlation_type = correlation_type
        SeparabilityIndex.create(self, raw_data, channels, variable, correlation_type)


    def create(self, trial_data, channels, variable, correlation_type):
        target_variable = np.array([])
        spatio_temporal_data = {}

        for time_point in trial_data[1].keys():
            spatio_temporal_data[time_point] = {}
            for channel in channels:
                spatio_temporal_data[time_point][channel] = np.array([])

        for trial in trial_data.keys():
            target_variable = np.append(target_variable, trial_data[trial][time_point][variable])
            for time_point in trial_data[trial].keys():

                for channel in channels:
                    spatio_temporal_data[time_point][channel] = np.append(spatio_temporal_data[time_point][channel],
                                                                          trial_data[trial][time_point][channel])

        for time_point in spatio_temporal_data:
            self.data[time_point] = {}

            for channel in spatio_temporal_data[time_point]:
                if correlation_type is 'pointbiserial':
                    r = stats.pointbiserialr(target_variable, spatio_temporal_data[time_point][channel].astype(np.float))
                if correlation_type is 'pearson':
                    r = stats.pearsonr(target_variable.astype(np.float), spatio_temporal_data[time_point][channel].astype(np.float))
                else:
                    raise ValueError('Not a valid correlation type')
                self.data[time_point][channel] = r[0]


    def save_as_heatmap(self, path_out):
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
        # channel_wanted = channels
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
        cmap = plt.get_cmap('RdBu')

        mesh = ax.pcolormesh(time_points, channel_numbers, np.swapaxes(correlations, 0, 1), cmap=cmap)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)

        fig.colorbar(mesh, cax=cax)  # need a colorbar to show the intensity scale

        file_out = path_out + "\\" + self.name + ".png"
        plt.savefig(file_out, dpi=200)

        print("Saved separability diagram to " + file_out)


    def pickle(self, path):
        file = os.path.abspath(os.path.join(path, self.name + '.pkl'))
        with open(file, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)


def extract_features(trial_data, time_periods, channels):

    extracted_features = {}

    for trial in trial_data:

        extracted_features[trial] = LastUpdatedOrderedDict()
        extracted_features[trial]['Trial'] = trial
        extracted_features[trial]['Class'] = trial_data[trial][1]['Class']

        for time_period in time_periods:
            begin = time_period[0]
            end = time_period[1]
            # get time points within range defined by begin and end
            time_points = np.arange(begin, end + 1, 1)

            for channel in channels:
                sum = 0.0
                for time_point in time_points:
                    sum = sum + float(trial_data[trial][time_point][channel])

                avg = sum / len(time_points)

                variable_name = str(channel) + "_" + str(begin) + "-" + str(end)
                extracted_features[trial][variable_name] = avg

    # OrderedDict(sorted(d.items(), key=lambda t: t[0]))

    return extracted_features


class LastUpdatedOrderedDict(OrderedDict):
    'Store items in the order the keys were last added'

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        OrderedDict.__setitem__(self, key, value)

__author__ = "Mike"

from collections import OrderedDict
import os
import pickle
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def correlate(raw, chs, target, rtype, alpha):

    dim_x = len(raw.index.get_level_values(0).unique())
    dim_y = len(raw.index.get_level_values(1).unique())

    st = raw[chs].values.reshape((dim_x, dim_y, raw[chs].shape[1])).astype(np.float)
    targets = raw[[target]].values.reshape((dim_x, dim_y, 1))[:, 0, 0].astype(np.float)

    if rtype is 'pointbiserial':
        r_matrix = np.vstack((np.hstack(
            (stats.pointbiserialr(targets, st[:, t, ch])[0] for ch in range(st.shape[2]))) for t in range(st.shape[1])))
    elif rtype is 'pearson':
        r_matrix = np.vstack((np.hstack(
            (stats.pearsonr(targets, st[:, t, ch])[0] for ch in range(st.shape[2]))) for t in range(st.shape[1])))
    else:
        raise ValueError('Not a valid correlation type')

    return r_matrix


class SeparabilityIndex:

    def __init__(self, name, raw, chs, target, rtype='pearson', desc=None, alpha=None):
        self.name = name
        self.raw = raw
        self.chs = chs
        self.target = target
        self.rtype = rtype
        self.alpha = alpha

        if desc is None:
            self.desc = name
        else:
            self.desc = desc

        # initialize empty data and features dictionary
        self.features = {}
        self.r_matrix = correlate(raw, chs, target, rtype, alpha)
        self.heat_map = None

    def visualize(self, cmap=None, path=None, file_format='png'):

        # make default colormap
        if cmap is None:
            c = mcolors.ColorConverter().to_rgb
            cmap = make_colormap(
                [c('blue'), c('cyan'), 0.3, c('cyan'), c('white'), 0.45, c('white'), 0.55, c('white'), c('yellow'), 0.7,
                 c('yellow'), c('red')])

        t = range(1, self.r_matrix.shape[0] + 1)
        t_n = self.r_matrix.shape[0]
        # helper array for pcolormesh
        chs = range(1, len(self.chs) + 1)

        # setup the 2D grid
        t, chs = np.meshgrid(t, np.asarray(chs))

        # define channel labels
        ch_labels = []
        # channel_wanted = self.channels
        ch_wanted = ['Fz', 'FCz', 'Cz', 'Pz']
        for ch in self.chs:
            if ch in ch_wanted:
                ch_labels.append(ch)
            else:
                ch_labels.append('')

        fig, ax = plt.subplots(figsize=(15, 5))
        ax.set_yticks(np.arange(1, len(self.chs), 1) + 0.5, minor=False)
        ax.set_yticklabels(ch_labels)
        ax.set_xticks(np.arange(0, t_n, 25) + 0.5, minor=False)
        ax.set_xticks(np.arange(0, t_n, 5) + 0.5, minor=True)
        ax.set_xticklabels(np.arange(-100, (t_n - 25) * 4, 100))
        ax.set_xlabel("[ms]")
        ax.set_ylabel("channel")
        ax.set_title(self.desc)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(axis='both', which='major', labelsize=12)

        # pick the desired colormap, sensible levels, and define a normalization
        # instance which takes data values and translates those into levels.

        mesh = ax.pcolormesh(t, chs, np.swapaxes(self.r_matrix, 0, 1), vmin=-0.45, vmax=0.45, cmap=cmap)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.05)

        fig.colorbar(mesh, cax=cax)  # need a colorbar to show the intensity scale

        self.heat_map = plt

        if path is None:
            self.heat_map.show()
        else:
            file = os.path.abspath(os.path.join(path, self.name + '.' + file_format))
            self.heat_map.savefig(file, format=file_format, dpi=300, bbox_inches='tight')
            print("Saved separability diagram to " + file)

    def pickle(self, path):
        file = os.path.abspath(os.path.join(path, self.name + '.pkl'))
        with open(file, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def extract_features(self, time_periods, channels, target_variable):

        # transform time period from ms (which is more user-friendly) to scan points (which is the technical representation)
        time_periods = np.around(np.array(time_periods) / 4 + 25)

        for case in self.raw:

            self.features[case] = LastUpdatedOrderedDict()
            self.features[case]['ID'] = case
            self.features[case][target_variable] = self.raw[case][1][target_variable]

            for time_period in time_periods:
                begin = time_period[0]
                end = time_period[1]
                # get time points within range defined by begin and end
                time_points = np.arange(begin, end + 1, 1)

                for channel in channels:
                    sum = 0.0
                    for time_point in time_points:
                        sum = sum + float(self.raw[case][time_point][channel])

                    avg = sum / len(time_points)

                    variable_name = str(channel) + "_" + str(begin) + "-" + str(end)
                    self.features[case][variable_name] = avg

    def get_features_and_labels(self, target=None):

        features = self.features
        if target is None:
            target = self.target

        if target in features[1].keys():
            # stack labels into nparray
            y = np.hstack((features[case][target] for case in features))
            y = y.astype(np.float64)
            y = y - 1

            # stack features into nparray
            X = np.vstack(
                (np.hstack(features[case][feature] for feature in features[case].keys()) for case in features))
            X = np.delete(X, [0, 1], axis=1)
            X = X.astype(np.float64)

            return X, y
        else:
            raise ValueError('Not a valid target variable')

    def save_extracted_features(self, path):

        features = self.features
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


def make_colormap(seq):
    """
    Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


class LastUpdatedOrderedDict(OrderedDict):
    'Store items in the order the keys were last added'

    def __setitem__(self, key, value):
        if key in self:
            del self[key]
        OrderedDict.__setitem__(self, key, value)

__author__ = 'Mike'
import os
import csv


def write_separability_index_to_file(separability_index, filename_out, channels, path_out):
    """
    Writes separability indices to a csv file.
    :param separability_index: Dictionary with separability values
    :param filename_out: Name of output file
    :param path_out: Path to output file
    :return:
    """
    if not os.path.isdir(path_out):
        os.mkdir(path_out)

    file_out = path_out + "\\" + filename_out
    with open(file_out, "w") as csv_file:

        writer = csv.DictWriter(csv_file, fieldnames=channels, lineterminator='\n')

        writer.writeheader()
        for time_point in separability_index.keys():
            data = {key: value for key, value in separability_index[time_point].items()
                    if key in channels}
            writer.writerow(data)

    print("Saved separability data file to " + file_out)


def write_features_to_file(features, filename_out, path_out):
    """
    Writes features to a csv file.
    :param features:
    :param filename_out: Name of output file
    :param path_out: Path to output file
    :return:
    """
    if not os.path.isdir(path_out):
        os.mkdir(path_out)

    file_out = path_out + "\\" + filename_out
    with open(file_out, "w") as csv_file:

        fieldnames = features[1].keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, lineterminator='\n')

        writer.writeheader()
        for trial in features.keys():
            data = {key: value for key, value in features[trial].items()
                    if key in features[trial].keys()}
            writer.writerow(data)

    print("Saved feature data file to " + file_out)

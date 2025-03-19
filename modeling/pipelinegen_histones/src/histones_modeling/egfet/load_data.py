import pandas as pd
import numpy as np
import os
import sys
from typing import Dict
import re
from collections import defaultdict
from typing import List, Tuple

r1 = re.compile("s.\.xlsx$")  # ignore summary files

# get concentration from filename
r2 = re.compile("[0-9\.]?[0-9]+[nump]M")


def separate_by_cycle(
    data: pd.DataFrame, separate_key="Gate Voltage (V)"
) -> pd.DataFrame:
    """Given the data for a specific concentration and trial number, separate the data into individual cycles.

    A cycle is defined by the gate voltage returning to the initial value. This function will multi index by cycle number
    and return the multi-indexed dataframe.

    Args:
        data (pd.DataFrame): Data for a specific concentration and trial number

    Returns:
        List[pd.DataFrame]: Multi-indexed dataframe
    """
    # get the time elapsed
    separate_data = data[separate_key]
    # find when gate voltage drops at least 0.5V and is negative

    max_drop = np.min(np.diff(separate_data))
    cycle_indices = np.where(np.diff(separate_data) < 0.5 * max_drop)[0] + 1
    # get the indices for the cycles
    cycle_indices = list(cycle_indices)
    # add the last index
    cycle_indices.append(data.index[-1])
    # prepend the first index
    cycle_indices.insert(0, data.index[0])
    # get the cycles
    cycles = [
        data.iloc[cycle_indices[i] : cycle_indices[i + 1]]
        for i in range(len(cycle_indices) - 1)
    ]
    # multi index by cycle number
    for i in range(len(cycles)):
        cycles[i]["Cycle"] = i
    # combine the dataframes
    data = pd.concat(cycles)
    data = data.reset_index()
    return data


def flatten_data(data: pd.DataFrame) -> pd.DataFrame:
    """Normalizes cycles between zero and 1
    Does the following:
        1. Subtract the minimum value from each cycle
        2. Divide by the maximum value from each cycle


    Args:
        data (pd.DataFrame): dataframe to flatten
    Returns:
        pd.DataFrame: flattened dataframe
    """
    # get the maximum value for each cycle
    max_values = data.groupby(["Cycle"])["Drain Current (nA)"].max()
    # get the minimum value for each cycle
    min_values = data.groupby(["Cycle"])["Drain Current (nA)"].min()
    # get the difference between the max and min values
    diff = max_values - min_values
    # for each cycle subtract the minimum value and divide by the difference
    for i in range(len(max_values)):
        # cycle from the max_values dataframe
        cycle = max_values.index[i]
        # get the minimum value
        min_value = min_values[i]
        # get the difference
        difference = diff[i]
        # subtract the minimum value
        data.loc[data["Cycle"] == cycle, "Drain Current (nA)"] -= min_value
        # divide by the difference
        data.loc[data["Cycle"] == cycle, "Drain Current (nA)"] /= difference

    return data


def load_egfet_file(
    data_path,
    flatten: bool = False,
    downsample: int | None = None,
    verbose: int = 0,
) -> pd.DataFrame:
    """
    Load an individual EGFET EXCEL file from a csv with four columns
    (Drain Voltage, Drain Current, Time Elapsed, Gate Voltage)
    and return a pandas dataframe with the data separated by Cycle
    data_path: path to the data file
    """
    verbose_ = verbose > 0
    # Load the data
    df = pd.read_excel(data_path)
    df = separate_by_cycle(df)
    if verbose_:
        print(f"Loaded {data_path}")
        # print the number of cycles
        print(f"Number of cycles: {len(df['Cycle'].unique())}")
    if flatten:
        df = flatten_data(df)
    if downsample:
        df = df.iloc[::downsample]
    return df


def load_egfet_folder(
    data_path: str,
    exclude_concentration: List[float] | None = None,
    flatten: bool = False,
    downsample: int | None = None,
    verbose: int = 0,
) -> pd.DataFrame:
    """
    Load all the EGFET EXCEL files in a folder from a csv with four columns
    (Drain Voltage, Drain Current, Time Elapsed, Gate Voltage)
    and return a pandas dataframe with the data. Each of the files will have a concentration
    associated with it in the filename. Eg 20221214_HP_HBS_CTH_1uM.xlsx has a CTH concentration of 1uM and
    20221214_HP_HBS_CTH_10uM_1.xlsx has a CTH concentration of 10uM
    data_path: path to the data folder
    """
    verbose_ = verbose > 0
    data = pd.DataFrame()
    # Load the data
    loaded_concentrations = defaultdict(int)
    for filename in os.listdir(data_path):
        if verbose_:
            print(f"Loading {filename}")
        if not filename.endswith(".xlsx"):
            continue
        if re.search(r1, filename):
            continue
        # get the concentration from filename
        concentration = re.search(r2, filename)
        if not concentration:
            print("Unable to find concentration in filename: " + filename)
            continue
        concentration = concentration.group(0)  # should yield something like 6.25nM

        # separate units and value
        concentration_units = concentration[-2:]
        concentration_value = concentration[:-2]
        if verbose:
            print(f"Concentration: {concentration_value} {concentration_units}")
        # convert to nanomolar
        if concentration_units == "nM":
            concentration_value = float(concentration_value)
        elif concentration_units == "uM":
            concentration_value = float(concentration_value) * 1000
        elif concentration_units == "mM":
            concentration_value = float(concentration_value) * 1000000
        elif concentration_units == "pM":
            concentration_value = float(concentration_value) / 1000
        else:  # invalid units
            print("Invalid concentration units: " + concentration_units)
            continue
        if exclude_concentration and concentration_value in exclude_concentration:
            print(f"Excluding concentration: {concentration_value}")
            continue
        # load the data
        temp = load_egfet_file(
            data_path + filename, flatten, downsample, verbose=verbose - 1
        )
        # increment the trial number
        loaded_concentrations[concentration_value] += 1
        # add the trial number to the dataframe
        temp["Trial"] = loaded_concentrations[concentration_value]
        # add the concentration to the dataframe
        temp["Concentration"] = concentration_value
        # combine the dataframes
        data = pd.concat([data, temp], ignore_index=True)
    # multi index by concentration and Trial
    if verbose_:
        print(f"Loaded {data_path}")
        # print trials and concentrations
        print(f"Trials for each concentration:\n{loaded_concentrations}")
    data = data.reset_index()
    return data


def load_egfet_dataset(
    data_path: str,
    exclude_experiments: List[str] | None = None,
    exclude_concentration: List[float] | None = None,
    load_only: List[str] | None = None,
    flatten: bool = False,
    downsample: int | None = None,
    verbose: int = 0,
) -> pd.DataFrame:
    """Load the EGFET dataset folder by folder.
    Each folder contains data for a different analyte and aptamer. The folder name
    will be used to identify the experiment, and each data file in the folder will be loaded.
    Every folder starts with a date, followed by the analyte, then the aptamer, then the concentration.

    Args:
        data_path (str): Path to the dataset folder.
        exclude_experiments (List[str] | None): List of experiments to exclude from loading.
        exclude_concentration (List[float] | None): List of concentrations to exclude from loading.
        load_only (List[str] | None): List of experiments to load.
        flatten (bool): Whether to normalize the data cycles between 0 and 1.
        downsample (int | None): Factor by which to downsample the data.
        verbose (int): Verbosity level for logging.

    Returns:
        pd.DataFrame: Multi-indexed dataframe with the data organized by experiment, concentration, trial, and cycle.
    """
    # find valid folders
    verbose_ = verbose > 0
    folders = [
        f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))
    ]
    # make sure the folders start with numbers
    folders = [f for f in folders if f[0].isdigit()]
    if exclude_experiments:
        folders = [f for f in folders if f not in exclude_experiments]
    if load_only:
        folders = [f for f in folders if f in load_only]
    if verbose_:  # print the folders
        print("Folders:")
        print(folders)
    # load the data
    data = []
    for folder in folders:
        # load the data
        temp = load_egfet_folder(
            data_path + folder + "/",
            exclude_concentration,
            flatten,
            downsample=downsample,
            verbose=verbose - 1,
        )
        # check if empty
        if temp.empty:
            print("Unable to load folder: " + folder)
            continue
        # add the experiment to the dataframe
        temp["Experiment"] = folder
        # add the dataframe to the list
        data.append(temp)
    # combine the dataframes
    data = pd.concat(data, ignore_index=True)
    data = data.set_index(["Experiment", "Concentration", "Trial", "Cycle", "index"])
    if verbose_:
        print("Loaded dataset")
        # get indices
        indices = get_available_indices(data)
        # remove cycle and index and drop concentrations of 0
        indices = [i[:-1] for i in indices if i[1] != 0]
        # get unique indices
        indices = list(set(indices))
        # print the number of total cycles for this folder
        print(f"Number of total cycles: {len(indices)}")
        indices = [i[:-1] for i in indices]
        # print the number of available indices
        print(
            f"Number of loaded Experiments, Concentrations, and Trials: {len(indices)}"
        )

    return data


def get_available_indices(data: pd.DataFrame, depth="Cycle") -> List[Tuple[float, int]]:
    """Get the available indices from a dataframe

    Args:
        data (pd.DataFrame): dataframe to get the indices from

    Returns:
        List[Tuple[float, int]]: list of tuples with the concentration and trial number
    """
    # get the indices
    data = data.index.unique()
    if depth == "Trial":
        data = [i[:-2] for i in data]
    elif depth == "Concentration":
        data = [i[:-3] for i in data]
    elif depth == "Experiment":
        data = [i[:-4] for i in data]
    else:  # depth == "Cycle"
        data = [i[:-1] for i in data]
    return list(set(data))


def get_metadata_labels_groups(
    group, exclude, indices, label_map_override={}, verbose=0
):
    indices = list(indices)
    # remove any indices in the exclude list
    exclude_indices = []
    for e in exclude:
        exclude_indices += [i for i, j in enumerate(indices) if j[0] in e or e in j[0]]

    label_map = {
        "20221214_HP_HBS_CTH": 0,  # 0 corresponds to CTH
        "20230518_CTH_2kPEG": 0,
        "20240108_CTH_Hapt_2kPEG_24hr": 0,
        "20230929_1_BSA_Hapt_2kPEG": 0,  # 2 corresponds to BSA
        "20230929_2_BSA_Hapt_2kPEG": 0,
        "20240318_Histones_BSA": 1,
        "20240318_Histones_BSA_1": 1,
        "0240318_Histones_BSA_3": 1,
        "0240318_Histones_BSA_2": 1,
    }  # the others correspond to H4
    label_map.update(label_map_override)
    # use the Experiment index to get the labels
    labels = np.asarray([label_map.get(i[0], 1) for i in indices])
    # print the label distribution
    if verbose:
        print("Label distribution:")
        print(pd.Series(labels).value_counts())

    # generate groups by finding the unique values of the group indices matching the
    # groups to match
    group_cache = {}
    groups_store = []
    i = 0
    for experiment, concentration, trial, cycle in indices:
        key = ()
        if "Experiment" in group:
            key += (experiment,)
        if "Concentration" in group:
            key += (concentration,)
        if "Trial" in group:
            # key += (trial,)
            pass
        if "Cycle" in group:
            # key += (cycle,)
            pass
        if key not in group_cache:
            group_cache[key] = i
            i += 1
        groups_store.append(group_cache[key])
    groups = np.asarray(groups_store)
    return indices, exclude_indices, labels, groups


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    path = "/home/jalex/Data/aros/Histones/PavlidisGroup/EGFET Data/"
    data = load_egfet_dataset(path, flatten=True, verbose=2)
    print(data.keys())
    for key in data.keys():
        # assert that every dataframe is not empty
        assert not data[key].empty

    # get metadata
    indices, exclude, labels, groups = get_metadata_labels_groups(
        [
            "Experiment",
            "Concentration",
            "Trial",
        ],
        [],
        get_available_indices(data, depth="Cycle"),
        verbose=1,
        label_map_override={
            "20221214_HP_HBS_CTH": 0,  # 0 corresponds to CTH
            "20230518_CTH_2kPEG": 0,
            "20230929_1_BSA_Hapt_2kPEG": 2,  # 2 corresponds to BSA
            "20230929_2_BSA_Hapt_2kPEG": 2,
        },
    )
    print(f"Num Unique Trials {len(set(get_available_indices(data, depth='Trial')))}")

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Tuple, Dict, List, Union, Optional


_supported_datasets = {
    "2020": [
        "Calf Thymus Histone/Sensograms_CTH.txt",
        "Human Histones/Sensograms_HH.txt",
    ],
    "2023": ["Raw SPR Data 2023/All flow cells Kinetics raw data biaeval.txt"],
}


def find_phase_boundaries(df: pd.DataFrame) -> Tuple[float, float]:
    """Find boundaries for association and dissociation phases

    The association phase is marked by the largest increase in change of response. the dissociation phase is marked by the largest decrease in change of response (i.e. the largest negative change in response). It is expected that the regeneration phase is dropped from the dataframe before calling this function.
    """
    time = df["time"].values
    sensogram = df["sensogram"].values
    # calculate the change in response
    delta_sensogram = np.diff(sensogram)
    # calculate the change in time
    delta_time = np.diff(time)
    # calculate the change in response per second
    delta_sensogram_per_second = delta_sensogram / delta_time
    # find the index of the largest increase in response per second
    assoc_index = np.argmax(delta_sensogram_per_second)
    # find the index of the largest decrease in response per second
    disoc_index = np.argmin(delta_sensogram_per_second)
    # return the time at the index of the largest increase in response per second
    # and the time at the index of the largest decrease in response per second
    return time[assoc_index], time[disoc_index]


def load_spr_dataset(
    path: Path, find_phase_boundaries=True, dataset: Optional[str] = None
) -> pd.DataFrame:
    """Generates a dataframe from SPR data.

    Args:
        path (Path): Path to the SPR data.
        dataset (Optional[str], optional): Dataset to load. Defaults to None.

    Returns:
        pd.DataFrame: SPR data with columns: [time, flow_cell_x] and multiindex: [dataset, analyte, ligand, concentration]
    """
    if dataset is not None:
        raise NotImplementedError("Loading specific datasets is not yet supported")
    df = []
    for year, _ in _supported_datasets.items():
        if year == "2020":
            temp = load_2020_spr(
                path,
                find_boundaries=find_phase_boundaries,
                drop_regeneration=True,
            )
        elif year == "2023":
            temp = load_2023_spr(path)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        df.append(temp)
    df = pd.concat(df)
    return df


def load_2020_spr(
    path: Path, find_boundaries: bool = True, drop_regeneration: bool = True
) -> pd.DataFrame:
    """Generates a dataframe from SPR data.

    The data is in a tab separated csv file with the following format:

    Cycle={a}_Fc={b}_X  Cycle={a}_Fc={b}_Y

    where a represents the cycle number and b represents the flow cell number. X and Y are the raw data from the SPR,
    representing time and response respectively. Each cycle is associated with a unique concentration and an analyte, and each flow cell is
    associated with a ligand. The analyte and ligand are specified in the _cycle_legend and _flow_cell_legend variables.
    We ignore flow cell strings that are not in the _flow_cell_legend variable. We also ignore cycles that are not in
    the _cycle_legend variable.

    Concentrations are specified in nM. The analyte and ligand are specified in the _cycle_legend and _flow_cell_legend variables.

    Args:
        path (Path): Path to the SPR root spr data directory.
    Returns:
        pd.DataFrame: SPR data with columns: [time, sensogram] and multiindex: [dataset, analyte, ligand, concentration, cycle, phase]
    """
    data_files = _supported_datasets["2020"]

    _cycle_legend_1 = {
        4: (0, "CTH"),
        5: (1.3531, "CTH"),
        6: (2.7062, "CTH"),
        7: (5.4124, "CTH"),
        8: (10.825, "CTH"),
        9: (21.65, "CTH"),
        10: (43.3, "CTH"),
        11: (86.6, "CTH"),
        12: (173.2, "CTH"),
        13: (0, "BSA"),
        14: (1.5625, "BSA"),
        15: (3.125, "BSA"),
        16: (6.25, "BSA"),
        17: (12.5, "BSA"),
        18: (25, "BSA"),
        19: (50, "BSA"),
        20: (100, "BSA"),
        21: (200, "BSA"),
        22: (0, "CTH"),
        23: (43.3, "CTH"),
        24: (0, "BSA"),
        25: (50, "BSA"),
    }

    _cycle_legend_2 = {
        10: (0, "Histones 3.2"),
        11: (6.25, "Histones 3.2"),
        12: (12.5, "Histones 3.2"),
        13: (25, "Histones 3.2"),
        14: (50, "Histones 3.2"),
        15: (100, "Histones 3.2"),
        16: (200, "Histones 3.2"),
        17: (400, "Histones 3.2"),
        18: (50, "Histones 3.2"),
        19: (0, "Histones 4"),
        20: (0, "Histones 4"),
        21: (3.13, "Histones 4"),
        22: (6.25, "Histones 4"),
        23: (12.5, "Histones 4"),
        24: (25, "Histones 4"),
        25: (50, "Histones 4"),
        26: (100, "Histones 4"),
        27: (200, "Histones 4"),
        28: (25, "Histones 4"),
        29: (0, "Histones 4"),
        30: (0, "BSA"),
        31: (3.13, "BSA"),
        32: (6.25, "BSA"),
        33: (12.5, "BSA"),
        34: (25, "BSA"),
        35: (50, "BSA"),
        36: (100, "BSA"),
        37: (200, "BSA"),
        38: (400, "BSA"),
        39: (50, "BSA"),
        40: (0, "BSA"),
        41: (0, "CTH"),
        42: (25, "CTH"),
        43: (50, "CTH"),
        44: (100, "CTH"),
        45: (200, "CTH"),
    }

    _flow_cell_legend_1 = {
        3: "AU",
        4: "KU7(15Hr)",  # 1uM RNA injections for 15 hours
    }
    _flow_cell_legend_2 = {
        1: "AU1",  # These are the same, but I'm keeping them separate in case they change
        2: "AU2",
        3: "KU7(10Hr)",  # 1uM RNA injections for 10 hours
        4: "KU7(15Hr)",  # 1uM RNA injections for 15 hours
    }
    _assoc_time = 180
    _disoc_time = 600

    data_frames = []
    for data_file in data_files:
        if "Calf" in data_file:
            cycle_legend = _cycle_legend_1
            flow_cell_legend = _flow_cell_legend_1
        elif "Human" in data_file:
            cycle_legend = _cycle_legend_2
            flow_cell_legend = _flow_cell_legend_2
        else:
            raise ValueError(f"Unknown dataset: {data_file}")
        data_file_path = path / data_file
        data = pd.read_csv(data_file_path, sep="\t")
        # get every cycle and legend listed in the cycle and flow cell legends and make a dataframe
        # with the cycle number, flow cell number, and raw data
        for cycle, (concentration, analyte) in cycle_legend.items():
            for flow_cell, ligand in flow_cell_legend.items():
                cycle_str = f"Cycle={cycle}_Fc={flow_cell}"
                if f"{cycle_str}_X" in data.columns:
                    temp = data[[f"{cycle_str}_X", f"{cycle_str}_Y"]]
                    temp = temp.rename(
                        columns={
                            f"{cycle_str}_X": "time",
                            f"{cycle_str}_Y": "sensogram",
                        }
                    ).assign(
                        analyte=analyte,
                        ligand=ligand,
                        concentration=concentration,
                        dataset=data_file,
                        cycle=cycle,
                    )

                    # add experiment phase columns; any time before _assoc_time is association,
                    # any time before _assoc_time + _disoc_time and after_assoc_time is dissociation
                    # any time after _assoc_time + _disoc_time is regeneration
                    temp["phase"] = "association"
                    temp.loc[temp["time"] > _assoc_time, "phase"] = "dissociation"
                    if drop_regeneration:
                        temp = temp[temp["time"] < _assoc_time + _disoc_time]
                        if find_boundaries:
                            start_assoc_time, start_disoc_time = find_phase_boundaries(
                                temp
                            )
                            # drop data before start_assoc_time
                            temp = temp[temp["time"] > start_assoc_time]
                            temp.loc[temp["time"] < start_disoc_time, "phase"] = (
                                "association"
                            )
                            temp.loc[temp["time"] >= start_disoc_time, "phase"] = (
                                "dissociation"
                            )
                    else:
                        temp.loc[temp["time"] > _assoc_time + _disoc_time, "phase"] = (
                            "regeneration"
                        )
                    data_frames.append(temp)

    df = pd.concat(data_frames)
    df.dropna(inplace=True)  # drop rows with NaN values
    df = df.set_index(
        ["dataset", "analyte", "ligand", "concentration", "cycle", "phase"]
    )
    return df


def load_2023_spr(path: Path):
    """Loads the 2023 SPR data

    The data is in a tab separated csv file with the following format:

        Fc{a}_{ligand}_{analyte} [{concentration}]_X  Fc{a}_{ligand} [{concentration}]_Y

    where a represents the flow cell number, ligand represents the ligand, and
    concentration represents the concentration of the analyte in nM. X and Y are
    the raw data from the SPR, representing time and response respectively. Each flow
    cell is associated with a unique ligand.

    Args:
        path (Path): Path to the SPR root spr data directory.
    Returns:
        pd.DataFrame: SPR data with columns: [time, sensogram] and multiindex: [dataset, analyte, ligand, concentration, cycle, phase]
    """
    _data_file = _supported_datasets["2023"][0]

    # load the data
    data_file_path = path / _data_file
    data = pd.read_csv(data_file_path, sep="\t")

    # for each column, get the flow cell number, ligand, analyte, and concentration
    # and make a dataframe with the flow cell number, ligand, analyte, concentration, and raw data
    data_frames = []
    # get all column names and strip _X and _Y from the end
    # keep only unique columns
    columns = list(set(["_".join(col.split("_")[:-1]) for col in data.columns]))
    for cycle, col in enumerate(columns):
        # get the flow cell number
        flow_cell = int(col.split("_")[0].replace("Fc", ""))
        # get the ligand
        ligand = col.split("_")[1]
        # get the analyte
        analyte = col.split("_")[2].split(" ")[0]
        # get the concentration
        concentration = float(col.split(" ")[-1].replace("[", "").replace("]", ""))

        # make a dataframe with the flow cell number, ligand, analyte, concentration, and raw data
        temp = data[[f"{col}_X", f"{col}_Y"]]
        temp = temp.rename(
            columns={
                f"{col}_X": "time",
                f"{col}_Y": "sensogram",
            }
        ).assign(
            analyte=analyte,
            ligand=ligand,
            concentration=concentration,
            dataset="2023 SPR",
            cycle=cycle,
        )
        # drop regeneration phase
        temp = temp[temp["time"] < 800]
        start_assoc_time, start_disoc_time = find_phase_boundaries(temp)
        # drop data before start_assoc_time
        temp = temp[temp["time"] > start_assoc_time]
        temp.loc[temp["time"] < start_disoc_time, "phase"] = "association"
        temp.loc[temp["time"] >= start_disoc_time, "phase"] = "dissociation"
        data_frames.append(temp)
    data = pd.concat(data_frames)
    data.dropna(inplace=True)  # drop rows with NaN values
    data = data.set_index(
        ["dataset", "analyte", "ligand", "concentration", "cycle", "phase"]
    )
    return data


if __name__ == "__main__":
    path = Path("/home/jalex/Projects/aros/histones/Histones/data/SPR Data Files")
    df = load_spr_dataset(path)
    print(df)
    results_dir = Path("/home/jalex/Projects/aros/histones/Histones/data/results")
    results_dir.mkdir(exist_ok=True)
    results_dir = results_dir / "data_graphs"
    results_dir.mkdir(exist_ok=True)
    # make a subdirectory named data_graphs in the results directory
    # draw a graph for each cycle of each dataset
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    def make_plot(
        df,
        dataset,
        titles,
        analyte,
        ligand,
        concentration,
        cycle,
        save=True,
        display=False,
    ):
        fig = go.Figure()
        for phase, row in zip(["association", "dissociation"], [1, 2]):
            temp = df.loc[phase]
            fig.add_trace(
                go.Scatter(
                    x=temp["time"],
                    y=temp["sensogram"],
                    name=f"{analyte} {ligand} {concentration} nM {phase}",
                ),
            )
        fig.update_layout(
            title=f"Cycle {cycle}",
            xaxis_title="Time (s)",
            yaxis_title="Response (RU)",
        )
        if display:
            fig.show()
        if save:
            temp_dir = results_dir / dataset.replace(" ", "_").split("/")[0]
            temp_dir.mkdir(exist_ok=True)
            fig.write_image(
                temp_dir
                / f"{analyte}_ligand_{ligand}_concentration_{concentration}_cycle_{cycle}.png"
            )

    for dataset in df.index.get_level_values("dataset").unique():
        temp_dataset = df.loc[dataset]
        for analyte in temp_dataset.index.get_level_values("analyte").unique():
            temp_analyte = temp_dataset.loc[analyte]
            for ligand in temp_analyte.index.get_level_values("ligand").unique():
                temp_ligand = temp_analyte.loc[ligand]
                for concentration in temp_ligand.index.get_level_values(
                    "concentration"
                ).unique():
                    temp_concentration = temp_ligand.loc[concentration]
                    for cycle in temp_concentration.index.get_level_values(
                        "cycle"
                    ).unique():
                        temp_cycle = temp_concentration.loc[cycle]
                        titles = [
                            f"{dataset} Cycle {cycle} Association",
                            f"{dataset} Cycle {cycle} Dissociation",
                        ]
                        make_plot(
                            temp_cycle,
                            dataset,
                            titles,
                            analyte,
                            ligand,
                            concentration,
                            cycle,
                            save=True,
                            display=False,
                        )

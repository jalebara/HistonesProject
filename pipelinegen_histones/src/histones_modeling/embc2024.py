# system imports
from pathlib import Path
from typing import Dict, List, Union, Tuple, Any, Optional
import click
import warnings
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from collections import defaultdict

# only show warnings once
warnings.filterwarnings("once")

# plotting imports
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots

# feature extraction and machine learning imports
import scipy
import numpy as np
import pandas as pd
import tsfel
from tsfresh import extract_features, select_features
import pycatch22
from imblearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    SelectPercentile,
    f_classif,
    mutual_info_classif,
    SelectFromModel,
)
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    matthews_corrcoef,
    roc_curve,
    auc,
    make_scorer,
    precision_recall_curve,
)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn import set_config


# custom imports
from egfet.load_data import load_egfet_dataset, get_metadata_labels_groups

# sklearn output dataframe instead of numpy array
set_config(transform_output="pandas")
# seed the random number generator
np.random.seed(4)  # 4 is the best seed


# function for creating a feature importance dataframe
def imp_df(column_names, importances):
    df = (
        pd.DataFrame({"feature": column_names, "feature_importance": importances})
        .sort_values("feature_importance", ascending=False)
        .reset_index(drop=True)
    )
    return df


# plotting a feature importance dataframe (horizontal barchart)
def var_imp_plot(imp_df, title):
    plt.figure()
    imp_df.columns = ["feature", "feature_importance"]
    sns.barplot(
        x="feature_importance", y="feature", data=imp_df, orient="h", color="green"
    )


def draw_and_save_precision_recall_curve(ypred, ytrue, path: Path):
    """Draws and saves a precision recall curve.

    Args:
        ypred (np.ndarray): The predicted labels.
        ytrue (np.ndarray): The true labels.
    """
    precision, recall, _ = precision_recall_curve(ytrue, ypred)
    auc_score = auc(recall, precision)
    fig = go.Figure(
        data=[
            go.Scatter(
                x=recall,
                y=precision,
                mode="lines",
                name=f"Precision-Recall Curve (AUC = {auc_score:.2f})",
            )
        ]
    )
    fig.update_layout(
        title_text="Precision-Recall Curve",
        xaxis_title_text="Recall",
        yaxis_title_text="Precision",
        bargap=0.2,  # gap between bars of adjacent location coordinates.
        bargroupgap=0.1,  # gap between bars of the same location coordinate.
    )
    fig.write_image(path / "precision_recall_curve.png")


class SelectFeaturesRecord(BaseEstimator, TransformerMixin):
    def __init__(self, path: Path):
        self.path = path

    def fit(self, X, y):
        # expects a pandas dataframe and saves it to the path
        i = 0
        path = self.path
        while path.exists():
            i += 1
            # add a number before the suffix
            temp = "_".join(path.stem.split("_")[:-1])
            path = path.parent / f"{temp}_{i}{path.suffix}"

        self.path = path
        X.to_csv(self.path)
        return self

    def transform(self, X):
        # no need to transform
        return X


class RepeatedStratifiedGroupKFold(StratifiedGroupKFold):
    def __init__(self, n_splits=5, n_repeats=10, random_state=None):
        super().__init__(n_splits=n_splits)
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, X, y=None, groups=None):
        for _ in range(self.n_repeats):
            yield from super().split(X, y, groups)

    def get_n_splits(self, X, y=None, groups=None):
        return super().get_n_splits(X, y, groups) * self.n_repeats


def specificity_score(y_true, y_pred, **kwargs):
    pos_label = kwargs.pop("pos_label", 1)
    y_true = np.asarray(y_true) == pos_label
    y_pred = np.asarray(y_pred) == pos_label
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def clean_egfet_data(
    egfet: pd.DataFrame,
) -> pd.DataFrame:
    """Cleans the EGFET data.

    There are many instances where a particular sweep is empty or has too few points.
    We remove these instances from the data.
    Args:
        egfet (pd.DataFrame): The EGFET data to clean.

    Returns:
        pd.DataFrame, List[int]: The cleaned EGFET data and the distribution of cycle sizes.
    """

    # get all indices
    indices = egfet.index.unique()
    # indices to list
    indices = list(indices)
    # remove index entries in (Experiment, Concentration, Trial, Cycle, index) format
    indices = [i[:-1] for i in indices]
    indices = list(set(indices))
    # track the size of the cycles; used for debugging
    cycle_sizes = []
    # iterate over the indices
    for experiment, concentration, trial, cycle in indices:
        # get the data for each cycle
        temp = egfet.loc[(experiment, concentration, trial, cycle, slice(None))]
        # get the data as a numpy array
        temp = temp["Drain Current (nA)"].to_numpy()
        cycle_sizes.append(len(temp))
        # check if the data is empty
        if temp.size == 0:
            # remove the data
            egfet = egfet.drop((experiment, concentration, trial, cycle))
            continue
        # check if the data has too few points
        if temp.size < 60:  # determined by looking at cycle size histograms
            # remove the data
            view = egfet.loc[(experiment, concentration, trial, cycle, slice(None))]
            egfet = egfet.drop((experiment, concentration, trial, cycle))
            continue
        if temp.size >= 60:
            temp = temp[:60]
            temp = temp - np.min(temp) + 1e-6
            # view for debugging purposes
            view = egfet.loc[(experiment, concentration, trial, cycle, slice(None))]
            # found that these were just extra long cycles. No overlap between cycles

    return egfet


def get_tsfresh_features(data: pd.DataFrame) -> pd.DataFrame:
    """Gets the TSFRESH features from the data.

    We take the loaded dataframe and use the indices as the id column, disregarding the 'index' column.
    Effectively, we are converting to a long dataframe format specified by TSFresh.
    The features for each sample are computed and returned as a dataframe.

    Args:
        data (pd.DataFrame): The data to get the features from. The indices are a multi-index of the form (Experiment, Concentration, Trial, Cycle, index)

    Returns:
        pd.DataFrame: The TSFRESH features. Each column is a feature.
    """

    # get the indices and assign them as the id column
    # make a copy of the data so we don't modify the original
    data = data.copy()
    data = data.reset_index()
    data = data.drop(columns=["index"])
    if "level_0" in data.columns:
        data = data.drop(columns=["level_0"])
    # combine the Experiment, Concentration, Trial, and Cycle columns into one column
    # Concentration, Trial, and Cycle are integers and floats,
    # so we convert them to strings to combine them
    data["id"] = (
        data["Experiment"].astype(str)
        + "#"
        + data["Concentration"].astype(str)
        + "#"
        + data["Trial"].astype(str)
        + "#"
        + data["Cycle"].astype(str)
    )
    # drop the Experiment, Concentration, Trial, Cycle, and Gate Voltage (V) columns
    if "Gate Voltage (V)" in data.columns:
        data = data.drop(columns=["Gate Voltage (V)"])
    if "Drain Voltage (V)" in data.columns:
        data = data.drop(columns=["Drain Voltage (V)"])
    if "Experiment" in data.columns:
        data = data.drop(columns=["Experiment"])
    if "Concentration" in data.columns:
        data = data.drop(columns=["Concentration"])
    if "Trial" in data.columns:
        data = data.drop(columns=["Trial"])
    if "Cycle" in data.columns:
        data = data.drop(columns=["Cycle"])

    # compute the tsfresh features
    features = extract_features(data, column_id="id", column_sort="Time Elapsed (s)")
    # get the indices
    indices = features.index
    # indices to list
    indices = list(indices)
    # convert concentration, trial, and cycle to floats and integers
    indices = [i.split("#") for i in indices]
    indices = [(str(i[0]), float(i[1]), int(i[2]), int(i[3])) for i in indices]
    # make multi-index
    indices = pd.MultiIndex.from_tuples(indices)
    # set the indices
    features.index = indices
    return features


def get_tsfel_features(
    data: pd.DataFrame, config: List[str] | None = None, fs: float = 1
) -> pd.DataFrame:
    """Gets the TSFEL features from the data.

    Args:
        data (pd.DataFrame): The data to get the features from. Shape (n_samples, n_points)
        config (Dict): The configuration for the TSFEL features. Example:
            [
                "temporal", "statistical", "spectral"
            ]

    Returns:
        pd.DataFrame: The TSFEL features. Each column is a feature.
    """

    # get the indices
    indices = data.index.unique()
    # indices to list
    indices = list(indices)
    # remove index entries in (Experiment, Concentration, Trial, Cycle, index) format
    indices = [i[:-1] for i in indices]
    indices = list(set(indices))
    # get each cycle and store the data in a list
    dlist = []
    index_list = []
    for experiment, concentration, trial, cycle in indices:
        # get the data for each cycle
        temp = data.loc[(experiment, concentration, trial, cycle, slice(None))]
        # get the data as a numpy array
        temp = temp["Drain Current (nA)"].to_numpy()
        index_list.append((experiment, concentration, trial, cycle))
        dlist.append(temp)

    # Compute the features
    cfg = tsfel.get_features_by_domain()
    if config:
        cfg = {key: cfg[key] for key in config}
    features = tsfel.time_series_features_extractor(cfg, dlist, fs=fs, n_jobs=-1)
    # put the Experiment, Concentration, Trial, Cycle columns back
    index_list = pd.MultiIndex.from_tuples(index_list)
    # keep the old indices in the same Experiment, Concentration, Trial, Cycle format
    features.index = index_list
    return features


def catch22_features_worker(
    data: Tuple[Tuple, List[float]]
) -> Tuple[Tuple, List[float]]:
    """Worker function for multiprocessing.

    Args:
        index (tuple): The index of the data.
        data (List[float]): The data to compute the features from.

    Returns:
        Tuple(tuple, List[float]): The index and the features.
    """
    index = data[0]
    data = data[1]
    features = pycatch22.catch22_all(data)
    return (index, features)


def get_catch22_features(data: np.ndarray) -> pd.DataFrame:
    """Gets the catch22 features from the data.

    Args:
        data (pd.DataFrame): The data to get the features from.

    Returns:
        pd.DataFrame: The catch22 features. Each column is a feature
    """
    # get a list of the indices
    indices = data.index.unique()
    # indices to list
    indices = list(indices)
    # remove index entries in (Experiment, Concentration, Trial, Cycle, index) format
    indices = [i[:-1] for i in indices]
    indices = list(set(indices))
    # get each cycle and store the data in a list
    pool_data = []
    for experiment, concentration, trial, cycle in indices:
        # get the data for each cycle
        temp = data.loc[(experiment, concentration, trial, cycle, slice(None))]
        # get the data as a list
        temp = temp["Drain Current (nA)"].to_list()
        pool_data.append(((experiment, concentration, trial, cycle), temp))
    with Pool(cpu_count()) as pool:
        features = []
        with tqdm(total=len(pool_data), desc="Computing catch22 features") as pbar:
            for f in pool.imap_unordered(catch22_features_worker, pool_data):
                pbar.update()
                features.append(f)
    # convert the features to a dataframe
    idx, cols = zip(*features)
    # convert the index to a multi-index
    idx = pd.MultiIndex.from_tuples(idx)
    # each col entry is a dictionary with the  keys (names values)
    temp = defaultdict(list)
    for col in cols:
        names = col["names"]
        values = col["values"]
        for name, value in zip(names, values):
            temp[name].append(value)

    features = pd.DataFrame(temp)
    # set the index
    features.index = idx
    return features


def remove_collinear_features(x: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
    """
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model
        to generalize and improves the interpretability of the model.

    Inputs:
        x: features dataframe
        threshold: features with correlations greater than this value are removed

    Output:
        dataframe that contains only the non-highly-collinear features
    """

    # Calculate the correlation matrix
    # Find highly correlated features
    corr = x.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
    # Drop features
    x = x.drop(x[to_drop], axis=1)
    print(f"Removed {len(to_drop)} collinear features")
    return x


def prepare_egfet_data(
    path: Path,
    config: Dict[str, Any] | None = None,
    output: Path | None = None,
    exclude_concentration: List[float] | None = None,
    flatten: bool = False,
    downsample: int | None = None,
) -> pd.DataFrame:
    """Prepares the EGFET data for feature extraction.

    Args:
        path (Path): The path to the EGFET data.
        config (Dict[str,Any], optional): The configuration for the EGFET data. Defaults to None.
        output (Path, optional): The path to save the prepared data to. Defaults to None.

    Returns:
        pd.DataFrame: The prepared EGFET features for machine learning.

    """
    if output and output.exists():
        data = pd.read_csv(output)
        # drop the first column
        data = data.drop(columns=["Unnamed: 0"])
        # first 4 columns are Experiment, Concentration, Trial, Cycle -- rename them
        data.columns = ["Experiment", "Concentration", "Trial", "Cycle"] + list(
            data.columns[4:]
        )
        data = data.set_index(["Experiment", "Concentration", "Trial", "Cycle"])
        return data
    # load the data
    egfet, fs = prep_raw_egfet(path, exclude_concentration, flatten, downsample)

    # get number of experiments
    n_experiments = len(list(set([i[:-1] for i in list(egfet.index.unique())])))
    print(f"Number of curves: {n_experiments}")

    # get the gradient of the data
    indices = egfet.index.unique()
    # indices to list
    indices = list(indices)
    # remove index entries in (Experiment, Concentration, Trial, Cycle, index) format
    indices = [i[:-1] for i in indices]
    indices = list(set(indices))
    # get each cycle, compute the gradient, and store the data in a list
    dlist = []
    for experiment, concentration, trial, cycle in indices:
        # get the data for each cycle
        temp = egfet.loc[(experiment, concentration, trial, cycle, slice(None))]
        # get the data as a numpy array
        temp = temp["Drain Current (nA)"].to_numpy()
        # compute the gradient
        temp = np.gradient(temp)
        temp = pd.DataFrame(
            {
                "Drain Current (nA)": temp,
                "Time Elapsed (s)": egfet.loc[
                    (experiment, concentration, trial, cycle, slice(None))
                ]["Time Elapsed (s)"],
            }
        )
        temp["Concentration"] = concentration
        temp["Trial"] = trial
        temp["Cycle"] = cycle
        temp["Experiment"] = experiment
        temp["index"] = temp.index
        temp = temp.set_index(
            ["Experiment", "Concentration", "Trial", "Cycle", "index"]
        )
        dlist.append(temp)
    # convert the data to a dataframe
    dlist = pd.concat(dlist, axis=0)

    # compute the features
    catch22_features_grad = get_catch22_features(dlist)
    catch22_features_grad.columns = [i + "_grad" for i in catch22_features_grad.columns]
    tsfresh_features_grad = get_tsfresh_features(dlist)
    tsfresh_features_grad.columns = [i + "_grad" for i in tsfresh_features_grad.columns]
    # tsfel_features_grad = get_tsfel_features(dlist, fs=fs)
    # tsfel_features_grad.columns = [i + "_grad" for i in tsfel_features_grad.columns]

    catch22_features = get_catch22_features(egfet)
    tsfresh_features = get_tsfresh_features(egfet)
    # tsfel_features = get_tsfel_features(egfet, fs=fs)

    # combine the features. Every row is a sample, every column is a feature
    features = pd.concat(
        [
            # tsfel_features,
            tsfresh_features,
            catch22_features,
            # tsfel_features_grad,
            catch22_features_grad,
            tsfresh_features_grad,
        ],
        axis=1,
    )
    # remove columns with mostly nan values
    features = features.dropna(axis=1, thresh=0.9 * features.shape[0])
    # drop the rows with NaN values
    features = features.dropna(axis=0)

    features = remove_collinear_features(features)

    # Use variance thresholding to remove low variance features and keep column names
    selector = VarianceThreshold()
    features = selector.fit_transform(features)

    # Remove features with low correlation to target

    if output:
        features.reset_index().to_csv(output)
    return features


def prep_raw_egfet(path, exclude_concentration, flatten, downsample):
    egfet = load_egfet_dataset(
        path,
        exclude_concentration=exclude_concentration,
        flatten=flatten,
        downsample=downsample,
    )
    # clean the data
    egfet = clean_egfet_data(egfet)
    fs = 1 / downsample if downsample else 1
    return egfet, fs


def build_pca_extraction_model(
    results_path: Path,
    scaling: str = "robust",
    model: str = "random_forest",
):
    """Builds a classical model.


    Returns:
        sklearn.pipeline.Pipeline: The classical model.
    """
    components = []
    if scaling == "robust":
        components.append(RobustScaler())
    elif scaling == "standard":
        components.append(StandardScaler())
    elif scaling == "min_max":
        components.append(MinMaxScaler())
    else:
        raise ValueError(f"Invalid scaling method: {scaling}")
    pca = PCA(n_components=5)
    components.append(pca)

    if model == "random_forest":
        components.append(
            RandomForestClassifier(n_estimators=1000, max_depth=3, n_jobs=-1)
        )
    else:
        raise ValueError(f"Invalid model: {model}")


def build_classical_model(
    results_path: Path,
    scaling: str = "robust",
    feature_selection: str | None = None,
    features_selection_kwargs: Dict | None = None,
    model: str = "random_forest",
):
    """Builds a classical model.


    Returns:
        sklearn.pipeline.Pipeline: The classical model.
    """
    components = [SMOTE()]
    if scaling == "robust":
        components.append(RobustScaler())
    elif scaling == "standard":
        components.append(StandardScaler())
    elif scaling == "min_max":
        components.append(MinMaxScaler())
    else:
        raise ValueError(f"Invalid scaling method: {scaling}")

    if feature_selection == "select_k_best":
        components.append(SelectKBest(f_classif, 10))
    elif feature_selection == "select_percentile":
        components.append(SelectPercentile(f_classif, percentile=10))
    elif feature_selection == "select_model":
        components.append(
            SelectFromModel(
                RandomForestClassifier(n_estimators=100, max_depth=3, n_jobs=-1),
                max_features=50,
            )
        )
    results_path = results_path / "features"
    results_path.mkdir(parents=True, exist_ok=True)
    record = SelectFeaturesRecord(results_path / "selected_features_0.csv")
    components.append(record)
    if model == "random_forest":
        components.append(
            RandomForestClassifier(n_estimators=1000, max_depth=3, n_jobs=-1)
        )
    else:
        raise ValueError(f"Invalid model: {model}")

    pipeline = make_pipeline(*components)
    return pipeline


def draw_pairplot(data: pd.DataFrame, hue: str, output: Path):
    """Draws a pairplot of the data and saves it to the output path.

    Each column of the data is plotted against each other column.

    Args:
        data (pd.DataFrame): The data to plot.
        output (Path): The path to save the plot to.
    """
    feature_name_map = {
        "Drain Current (nA)__change_quantiles__f_agg_"
        "var"
        "__isabs_False__qh_1.0__ql_0.6_grad": r"$s$ Binned Quantile Change",
        "0_FFT mean coefficient_27": "FFT Mean Coefficient 27",
        "0_Absolute energy_grad": r"Absolute Energy in $s'$",
        "Drain Current (nA)__agg_linear_trend__attr_"
        "slope"
        "__chunk_len_50__f_agg_"
        "var"
        "_grad": r"$I_{DS}$ Linear Trend",
        'Drain Current (nA)__change_quantiles__f_agg_"var"__isabs_False__qh_1.0__ql_0.6_grad': r"$s'$ Binned Variance Quantile Change",
        'Drain Current (nA)__linear_trend__attr_"stderr"': r"$s$ Linear Trend Std. Error",
        "Drain Current (nA)__energy_ratio_by_chunks__num_segments_10__segment_focus_1": r"$s$ Energy Ratio",
        'Drain Current (nA)__agg_autocorrelation__f_agg_"median"__maxlag_40_grad': r"$s'$ Autocorrelation",
    }
    data = data.rename(columns=feature_name_map)
    sns.pairplot(data, hue=hue, kind="kde")
    plt.savefig(output)
    plt.close()


@click.command()
@click.option(
    "--data_path",
    type=str,
    default="/data/aros/Histones/PavlidisGroup/EGFET Data/",
    required=True,
    help="The path to the data. If not provided, then the features path must be provided and the file must exist.",
)
@click.option(
    "--exclude_concentration",
    type=click.Choice(["0nM", "1nM", "10nM", "10pM", "100nM", "100pM", "none"]),
    required=False,
    default=["0nM"],
    multiple=True,
    help="Whether to exclude the zero concentration data.",
)
@click.option(
    "--flatten",
    type=bool,
    required=False,
    is_flag=True,
    default=False,
    help="Whether to flatten the data cycle-wise between 0 and 1.",
)
@click.option(
    "--features_path",
    type=Path,
    required=False,
    default=Path("/data/aros/Histones/PavlidisGroup/EGFET\ Processed\ Data/"),
    help="The path to save the features to. If not provided, the features will not be saved. If the file already exists, the features will be loaded from the file.",
)
@click.option(
    "--results_path",
    type=Path,
    required=False,
    default=Path("/projects/aros/histones/Histones/results_prelim"),
    help="The path to save the results to.",
)
@click.option(
    "-g",
    "--group",
    type=click.Choice(["Experiment", "Concentration", "Trial", "Cycle"]),
    multiple=True,
    required=False,
    default=["Experiment", "Concentration", "Trial"],
    help="The group to stratify the data by. Can be used multiple times.",
)
@click.option(
    "-t",
    "--test_set",
    type=click.Choice(
        [
            # Other Analytes
            "20221214_HP_HBS_CTH",
            "20230929_2_BSA_Hapt_2kPEG",
            "20230518_CTH_2kPEG",
            "20240108_CTH_Hapt_2kPEG_24hr",
            "20230929_1_BSA_Hapt_2kPEG",
            # H4 Analytes
            "20230519_H2kPEG_H4_NoPolish",
            "20230525_Hapt_2kPeg_H4",
            "20230526_Hapt_24_H4",
            "0230518_H4_Rod",
            "20230518_H4_2kPEG",
            "20240318_Histones_BSA_Test_set",
            "20240318_Histones_BSA_1",
            "0240318_Histones_BSA_3",
            "0240318_Histones_BSA_2",
        ]
    ),
    required=False,
    multiple=True,
    default=[
        # Other Analytes
        "20221214_HP_HBS_CTH",
        "20230929_2_BSA_Hapt_2kPEG",
        # H4 Analytes
        # "20230519_H2kPEG_H4_NoPolish",
        # "20230525_Hapt_2kPeg_H4",
        # "20230526_Hapt_24_H4",
        "0230518_H4_Rod",
        # "20230518_H4_2kPEG"
    ],
)
@click.option(
    "--run_name",
    type=str,
    required=False,
    default="test",
    help="The name of the run. Used for saving results.",
)
@click.option(
    "-f",
    "--feature_selection",
    type=click.Choice(["select_k_best", "select_percentile", "select_model"]),
    required=False,
    default="select_model",
    help="The feature selection method to use.",
)
@click.option(
    "-e",
    "--exclude",
    type=click.Choice(
        [
            # H4 Analytes
            "20230519_H2kPEG_H4_NoPolish",
            "20230525_Hapt_2kPeg_H4",
            "20230526_Hapt_24_H4",
            "0230518_H4_Rod",
            "20230518_H4_2kPEG",
        ]
    ),
    required=False,
    multiple=True,
    default=[],
)
@click.option(
    "-d",
    "--downsample",
    type=int,
    required=False,
    default=1,
    help="The factor to downsample the data by.",
)
@click.option("-g", "--graphics-only", is_flag=True, required=False, default=False)
def main(
    data_path,
    exclude_concentration,
    flatten,
    features_path,
    results_path,
    group,
    test_set,
    run_name,
    feature_selection,
    exclude,
    downsample,
    graphics_only,
):
    """Main function for the script."""
    test_set = list(set(test_set))
    exclude = list(set(exclude))
    # assert there is no intersection between test and exclude
    assert len(set(test_set).intersection(set(exclude))) == 0
    # configure sklearn to output dataframe instead of numpy array'
    features_path.mkdir(parents=True, exist_ok=True)
    print(f"Experiment: {run_name}")
    name = "features.csv"
    if exclude_concentration:
        for e in exclude_concentration:
            name = e + "_" + name
        name = "exclude_concentrations" + "_" + name
    if flatten:
        name = "magnitude_controlled" + "_" + name

    if downsample:
        name = f"downsample_{downsample}" + "_" + name

    # convert excluded concentrations from strings to namomolar floats
    to_exclude = []
    for e in exclude_concentration:
        if e == "none":
            exclude_concentration = None
            break
        value = e[:-2]
        i = float(value)
        if "pM" in e:
            to_exclude.append(i / 1000)
        elif "uM" in e:
            to_exclude.append(i * 1000)
        elif "mM" in e:
            to_exclude.append(i * 1000000)
        elif "nM" in e:
            to_exclude.append(i)
        else:
            raise ValueError(f"Invalid concentration: {e}")

    if graphics_only:
        pass
    else:
        classical_pipeline(
            data_path,
            flatten,
            features_path,
            results_path,
            group,
            test_set,
            run_name,
            feature_selection,
            exclude,
            downsample,
            name,
            to_exclude,
        )


def pca_pipeline(
    data_path,
    flatten,
    features_path,
    results_path,
    group,
    test_set,
    run_name,
    feature_selection,
    exclude,
    downsample,
    name,
    to_exclude,
):
    egfet, fs = prep_raw_egfet(data_path, to_exclude, flatten, downsample)
    # get groups and labels
    indices = egfet.index.unique()
    # indices to list
    indices = list(indices)
    # remove index entries in (Experiment, Concentration, Trial, Cycle, index) format
    indices = [i[:-1] for i in indices]
    indices = list(set(indices))
    # get the labels
    label_map = {
        "20221214_HP_HBS_CTH": 0,  # 0 corresponds to CTH
        "20230518_CTH_2kPEG": 0,
        "20230929_1_BSA_Hapt_2kPEG": 0,  # 2 corresponds to BSA
        "20230929_2_BSA_Hapt_2kPEG": 0,
    }  # the others correspond to H4
    # use the Experiment index to get the labels
    labels = np.asarray([label_map.get(i[0], 1) for i in indices])
    # print the label distribution
    print("Label distribution:")
    print(pd.Series(labels).value_counts())


def classical_pipeline(
    data_path,
    flatten,
    features_path,
    results_path,
    group,
    test_set,
    run_name,
    feature_selection,
    exclude,
    downsample,
    name,
    to_exclude,
):
    features_path = features_path / name
    features = prepare_egfet_data(
        data_path,
        output=features_path,
        exclude_concentration=to_exclude,
        flatten=flatten,
        downsample=downsample,
    )
    print(features.shape)
    print(features.head())

    # get the indices
    indices = features.index.unique()
    # indices to list
    indices, exclude_indices, labels, groups = get_metadata_labels_groups(
        group, exclude, indices
    )
    # get the test indices
    test_indices = []
    for test in test_set:
        test_indices += [
            i
            for i, j in enumerate(indices)
            if j[0] in test or test in j[0] and i not in exclude_indices
        ]
    # get the train indices
    train_indices = [
        i
        for i in range(len(indices))
        if i not in test_indices and i not in exclude_indices
    ]
    # get the train and test data
    X_train, X_test = features.iloc[train_indices], features.iloc[test_indices]
    y_train, y_test = labels[train_indices], labels[test_indices]
    groups_train, groups_test = groups[train_indices], groups[test_indices]

    if not results_path.exists():
        results_path.mkdir(parents=True, exist_ok=True)
    results_path = results_path / run_name
    results_path.mkdir(parents=True, exist_ok=True)
    # report train and test set distributions
    print("Train set label distribution:")
    print(pd.Series(y_train).value_counts())
    print("Test set label distribution:")
    print(pd.Series(y_test).value_counts())
    # save the train and test set distributions

    # count the number of items in directory
    n = len(list(results_path.glob("*")))
    results_path = results_path / f"run_{n}"
    results_path.mkdir(parents=True, exist_ok=True)

    pd.Series(y_train).value_counts().to_csv(
        results_path / "train_set_label_distribution.csv"
    )
    pd.Series(y_test).value_counts().to_csv(
        results_path / "test_set_label_distribution.csv"
    )
    cv_results_path = results_path / "cv_results"
    cv_results_path.mkdir(parents=True, exist_ok=True)
    pipeline = build_classical_model(
        cv_results_path, feature_selection=feature_selection
    )
    scoring = {
        "accuracy": make_scorer(accuracy_score, greater_is_better=True),
        "balanced_accuracy": make_scorer(
            balanced_accuracy_score, greater_is_better=True
        ),
        "precision": make_scorer(
            precision_score, greater_is_better=True, average="macro"
        ),
        "recall": make_scorer(recall_score, greater_is_better=True, average="macro"),
        "sensitivity": make_scorer(
            recall_score, greater_is_better=True, average="macro", pos_label=1
        ),
        "specificity": make_scorer(specificity_score, greater_is_better=True),
        "f1": make_scorer(f1_score, greater_is_better=True, average="macro"),
    }
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        scoring=scoring,
        groups=groups_train,
        cv=RepeatedStratifiedGroupKFold(n_splits=5, n_repeats=100),
        n_jobs=-1,
        return_estimator=True,
    )
    scores.pop("fit_time")
    scores.pop("score_time")
    estimators = scores.pop("estimator")
    # save the feature importances for each estimator in a subfolder
    feature_importance_path = cv_results_path / "feature_importances"
    feature_importance_path.mkdir(parents=True, exist_ok=True)
    top_features = []
    for i, estimator in enumerate(estimators):
        # get the feature importances
        importances = estimator.steps[-1][1].feature_importances_
        # get the column names
        column_names = estimator.steps[-3][1].get_support()
        column_names = features.columns[column_names]
        # get top 3 features
        top_features.append(column_names[np.argsort(importances)[-3:]])

        # create a dataframe
        df = imp_df(column_names, importances)
        # save the dataframe
        df.to_csv(feature_importance_path / f"feature_importances_{i}.csv")
    # save the scores

    # compute the top 3 features by frequency
    top_features = np.concatenate(top_features)
    top_features = pd.Series(top_features)
    top_features = top_features.value_counts()
    top_features.to_csv(cv_results_path / "top_features.csv")
    # get the top 3
    top_features = top_features[:3]
    # get the feature values from the dataset
    top_features = features[top_features.index]
    # draw a pairplot of the top 3 features
    # add labels and set hue to labels
    top_features["labels"] = labels
    # change labels to strings 0 is Other, 1 is H4
    top_features["labels"] = top_features["labels"].astype(str)
    # change labels to strings 0 is Other, 1 is H4
    top_features["labels"] = top_features["labels"].replace("0", "Other")
    top_features["labels"] = top_features["labels"].replace("1", "H4")

    draw_pairplot(
        top_features, hue="labels", output=cv_results_path / "top_features_pairplot.png"
    )

    scores = pd.DataFrame(scores)
    scores.to_csv(cv_results_path / "scores.csv")
    # graph the score histograms
    fig = go.Figure()
    for key in scores.columns:
        fig.add_trace(go.Histogram(x=scores[key], name=key))
    fig.update_layout(
        title_text=f"Histogram of scores for {run_name}",
        xaxis_title_text="Score",
        yaxis_title_text="Number of scores",
        bargap=0.2,  # gap between bars of adjacent location coordinates.
        bargroupgap=0.1,  # gap between bars of the same location coordinate.
    )
    fig.write_image(cv_results_path / "scores_histogram.png")
    fig.write_json(cv_results_path / "scores_histogram.json")
    # save average scores as csv
    scores = scores.mean(axis=0)
    scores = pd.DataFrame(scores).T
    scores.to_csv(cv_results_path / "average_scores.csv")

    # get the test set scores
    test_results_path = results_path / "test_set_results"
    pipeline = build_classical_model(
        test_results_path, feature_selection=feature_selection
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)
    # get the classification report
    # use the scoring to get the classification report
    scores = {}
    for name, scorer in scoring.items():
        scores[name] = scorer(pipeline, X_test, y_test)
    scores = pd.DataFrame(scores, index=[0])
    scores.to_csv(test_results_path / "test_set_scores.csv")
    # get the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm = ConfusionMatrixDisplay(cm, display_labels=["Other", "H4"]).plot()
    plt.savefig(test_results_path / "confusion_matrix.png")
    # get the specificity
    sensitivity = recall_score(y_test, y_pred, average="macro", pos_label=1)
    print(f"Sensitivity: {sensitivity}")
    specificity = specificity_score(y_test, y_pred, pos_label=1)
    print(f"Specificity: {specificity}")
    # get the roc curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1], pos_label=1)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC curve (area = {roc_auc})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random classifier"))
    fig.update_layout(
        title_text=f"ROC curve for {run_name}",
        xaxis_title_text="False Positive Rate",
        yaxis_title_text="True Positive Rate",
    )
    fig.write_image(test_results_path / "roc_curve.png")
    draw_and_save_precision_recall_curve(y_pred, y_test, test_results_path)
    print(scores)
    # get the feature importances
    importances = pipeline.steps[-1][1].feature_importances_
    # get the column names
    column_names = pipeline.steps[-3][1].get_support()
    column_names = features.columns[column_names]

    # create a dataframe
    df = imp_df(column_names, importances)
    # save the dataframe
    df.to_csv(test_results_path / "feature_importances.csv")
    # plot the first 5 feature importances
    var_imp_plot(df[:5], "Feature importances")
    plt.savefig(test_results_path / "feature_importances.png")


if __name__ == "__main__":
    main()

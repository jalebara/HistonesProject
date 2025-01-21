import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict
import glob

RESULTS_PATH = Path("/home/jalex/Projects/aros/histones/Histones/results")
TEST = "single_test"


def main():
    # make a histogram of features by loading all the selected_features_{i}.csv files
    # collecting all the column names
    test_path = RESULTS_PATH / TEST
    # for every subdirectory in the results path
    for sub in test_path.iterdir():
        feature_names = defaultdict(int)
        # if it's a directory
        if sub.is_dir():
            # for every file in that directory
            for file in glob.glob(f"{sub}/**/selected_features_*.csv"):
                # read the csv
                df = pd.read_csv(file)
                # for every column in the df
                for col in df.columns:
                    # add it to the dictionary
                    feature_names[col] += 1
            # draw a histogram of the features
            fig = go.Figure(data=[go.Histogram(x=list(feature_names.values()))])
            fig.update_layout(
                title_text=f"Histogram of features selected for {sub.name}",
                xaxis_title_text="Number of times feature was selected",
                yaxis_title_text="Number of features",
                bargap=0.2,  # gap between bars of adjacent location coordinates.
                bargroupgap=0.1,  # gap between bars of the same location coordinate.
            )
            fig.show()
            fig.write_image(f"{sub}/histogram_features.png")


if __name__ == "__main__":
    main()

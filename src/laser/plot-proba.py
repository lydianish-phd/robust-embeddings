import os, argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import _name, _display_name

COLORS = plt.cm.tab10.colors

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", help="path to artificial scores csv file", type=str)
    args = parser.parse_args()

    output_dir , file_name = os.path.split(args.input_file)
    metric = file_name.split("_")[0].split("-")[-1]
    output_file = os.path.join(output_dir, f"{metric}_noise_proba_plot.pdf")
    scores = pd.read_csv(args.input_file)

    print("Creating plot...")

    g = sns.lineplot(data=scores, x="proba", y=_name(metric), hue="model", style="model", markers=True, dashes=False)
    g.set(ylabel=f"{_display_name(metric)} score", xlabel="Probability of artificial UGC", xticks=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    _, labels = g.get_legend_handles_labels()
    g.grid()
    plt.tight_layout()
    plt.savefig(output_file)





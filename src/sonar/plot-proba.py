import os, argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import COLORS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--artificial-scores-file", help="path to artificial scores csv file", type=str)
    parser.add_argument("--multilingual-scores-file", help="path to multilingual scores csv file", type=str)
    args = parser.parse_args()

    output_dir , file_name = os.path.split(args.artificial_scores_file)
    metric = file_name.split('_')[0]
    output_file = os.path.join(output_dir, f"{metric}_noise_proba_plot.pdf")

    artificial_scores = pd.read_csv(args.artificial_scores_file)
    multilingual_scores = pd.read_csv(args.multilingual_scores_file)

    artificial_scores['model'] = artificial_scores['model'].apply(lambda x: MODEL_NAMES[x])
    multilingual_scores['model'] = multilingual_scores['model'].apply(lambda x: MODEL_NAMES[x])

    print("Concatenating multingual scores to artificial ones...")
    
    seeds = artificial_scores['seed'].unique()

    arti_data = artificial_scores[['model', 'seed', 'proba', 'avg']]
    multi_data = pd.DataFrame(columns=['model', 'seed', 'proba', 'avg'])
    multi_data['model'] = np.repeat(multilingual_scores['model'], len(seeds))
    multi_data['seed'] = np.tile(seeds, len(multilingual_scores))
    multi_data['proba'] = np.repeat(0, len(multilingual_scores)*len(seeds))
    multi_data['avg'] = np.repeat(multilingual_scores['avg__rocsmt__norm.en.test'], len(seeds))
    plot_data = pd.concat([arti_data, multi_data], ignore_index=True)

    print("Creating plot...")

    multilingual_scores.set_index("model", inplace=True)

    g = sns.lineplot(data=plot_data, x="proba", y="avg", hue="model", style="model", markers=True, dashes=False)
    g.set(ylabel="COMET score", xlabel="Probability of artificial UGC", xticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5])
    _, labels = g.get_legend_handles_labels()
    for i, model in enumerate(labels):
        g.axhline(multilingual_scores["avg__rocsmt__raw.en.test"].loc[model], ls='dashdot', c=COLORS[i])
    g.grid()
    plt.tight_layout()
    plt.savefig(output_file)





import os, argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from utils import (
    AVERAGE_ROCSMT_NORM_COLUMN,
    AVERAGE_ROCSMT_RAW_COLUMN,
    METRIC_NAMES,
    LANG_NAMES
)

COLORS = plt.cm.tab10.colors

def plot_probas(
        artificial_scores, 
        multilingual_scores, 
        metric,
        models,
        seeds, 
        arti_score_column, 
        norm_score_column, 
        raw_score_column,
        ax
    ):
    columns = ["model", "seed", "proba", "score"]
    arti_data = artificial_scores[metric][columns[:3]]
    arti_data["score"] = artificial_scores[metric][arti_score_column]
    multi_data = pd.DataFrame(columns=columns)
    multi_data["model"] = np.repeat(models, len(seeds))
    multi_data["seed"] = np.tile(seeds, len(multilingual_scores[metric]))
    multi_data["proba"] = np.repeat(0, len(multilingual_scores[metric])*len(seeds))
    multi_data["score"] = np.repeat(multilingual_scores[metric][norm_score_column].values, len(seeds))
    plot_data = pd.concat([arti_data, multi_data], ignore_index=True)
    g = sns.lineplot(data=plot_data, x="proba", y="score", hue="model", style="model", markers=True, dashes=False, ax=ax)
    _, labels = g.get_legend_handles_labels()
    for k, model in enumerate(labels):
        g.axhline(multilingual_scores[metric][raw_score_column].loc[model], ls="dashdot", c=COLORS[k])
    g.grid()
    return g
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--scores-dir", help="path to scores directory", type=str)
    parser.add_argument("-m", "--metrics", help="list of metrics", type=str, nargs="+")
    args = parser.parse_args()

    artificial_scores = {}
    multilingual_scores = {}
    for metric in args.metrics:
        artificial_scores[metric] = pd.read_csv(f"{args.scores_dir}/{metric}_artificial.csv")
        multilingual_scores[metric] = pd.read_csv(f"{args.scores_dir}/{metric}_multilingual.csv")
    
    seeds = artificial_scores["comet"]["seed"].unique()
    probas = np.insert(artificial_scores["comet"]["proba"].unique(), 0, 0.0)
    models = artificial_scores["comet"]["model"].unique()
    target_langs = sorted(set([ col.split('__')[1].split('-')[1] for col in artificial_scores["comet"].columns if col.startswith("rocsmt")]))

    for metric in multilingual_scores.keys():
        multilingual_scores[metric].set_index("model", inplace=True)
    
    print("Plotting average scores...")
    for metric in args.metrics:
        output_file = f"{args.scores_dir}/{metric}_noise_proba_plot.pdf"
        plt.clf()
        fig, ax = plt.subplots()
        g = plot_probas(
            artificial_scores, 
            multilingual_scores, 
            metric,
            models,
            seeds,
            arti_score_column="avg", 
            norm_score_column=AVERAGE_ROCSMT_NORM_COLUMN, 
            raw_score_column=AVERAGE_ROCSMT_RAW_COLUMN, 
            ax=ax
        )
        ax.set_ylabel(f"{METRIC_NAMES[metric]} score", fontsize=16)
        ax.set_xlabel("Probability of artificial UGC", fontsize=16)
        ax.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(output_file)
    
    print("Plotting language scores...")
    output_file = f"{args.scores_dir}/noise_proba_plot_all.pdf"
    plt.clf()
    fig, axes = plt.subplots(nrows=len(args.metrics), ncols=len(target_langs), sharex=True, figsize=(15, 6))

    for i, metric in enumerate(args.metrics):
        axes[i, 0].set_ylabel(f"{METRIC_NAMES[metric]} score", fontsize=16)
        for j, lang in enumerate(target_langs):
            axes[0, j].set_title(r"English$\rightarrow$" + LANG_NAMES[lang], fontsize=16)
            _ = plot_probas(
                artificial_scores, 
                multilingual_scores, 
                metric,
                models,
                seeds,
                arti_score_column=f"rocsmt_artificial__eng_Latn-{lang}__norm.en_mix_all.test", 
                norm_score_column=f"rocsmt__eng_Latn-{lang}__norm.en.test", 
                raw_score_column=f"rocsmt__eng_Latn-{lang}__raw.en.test", 
                ax=axes[i, j]
            )

    fig.supxlabel("Probability of artificial UGC", fontsize=16)

    # Remove individual x-axis labels
    for ax in axes.flat:
        ax.set_xlabel('')
        ax.set_xticks(probas)
    
    # Remove individual y-axis labels (except for the leftmost ones)
    for ax in axes[:,1:].flat:
        ax.set_ylabel('')

    # Remove individual legends
    for ax in axes.flat:
        ax.legend_.remove()
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=14, loc="upper center", bbox_to_anchor=(0.5, 1.075), ncol=3)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')



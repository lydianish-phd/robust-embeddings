import os, argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot import checkpoint_display_name
import matplotlib.ticker as ticker

LOSSES = ["loss_std_gold", "loss_ugc_gold", "valid_distil_loss"]
x_formatter = ticker.ScalarFormatter(useMathText=True)
y_formatter = ticker.ScalarFormatter(useMathText=True)
y_formatter.set_powerlimits((-4, -4))  # Set the exponent range to (-4, -4)
x_formatter.set_powerlimits((5, 5))  # Set the exponent range to (5, 5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="path to directory to read validation scores", type=str)
    args = parser.parse_args()

    all_scores = pd.read_csv(os.path.join(args.input_dir, "all_scores.csv" )).sort_values(by="steps")
    #all_scores.drop(all_scores[all_scores['model'].endswith('init')].index, inplace=True)

    print("Plotting combined loss curves for all models...")
    for loss in LOSSES:
        plt.clf()
        g = sns.lineplot(all_scores, x='steps', y=loss, hue='model')
        g.set(ylim=(0, 0.001))
        g.set_xticklabels([checkpoint_display_name(str(int(c))) for c in g.get_xticks()])
        plt.savefig(os.path.join(args.input_dir, loss + ".png"))
        plt.savefig(os.path.join(args.input_dir, loss + ".pdf"))
    
    plt.clf()
    models = all_scores["model"].unique()
    n_rows = models.size//2
    n_cols = min(models.size, 2)
    fig, axs = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(4*n_cols, 3*n_rows), sharex=True)

    print("Plotting separate loss curves for each model...")
    for n, model in enumerate(models):
        i, j = n//2, n%2
        all_scores[all_scores["model"] == model].plot(x="steps", y=LOSSES, ax=axs[i,j], title=model)
        axs[i,j].xaxis.set_major_formatter(x_formatter)
        axs[i,j].yaxis.set_major_formatter(y_formatter)
    fig.tight_layout()
    plt.savefig(os.path.join(args.input_dir, "models_valid.png"))
    plt.savefig(os.path.join(args.input_dir, "models_valid.pdf"))

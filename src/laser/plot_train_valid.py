import os, argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot import checkpoint_display_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="path to directory to read validation scores", type=str)
    args = parser.parse_args()

    all_scores = pd.read_csv(os.path.join(args.input_dir, "all_scores" + ".csv" ))
    #all_scores.drop(all_scores[all_scores['model'].endswith('init')].index, inplace=True)

    def plot_loss(lossname):
        plt.clf()
        g = sns.lineplot(all_scores, x='steps', y=lossname, hue='model')
        g.set(ylim=(0, 0.005))
        g.set_xticklabels([checkpoint_display_name(str(int(c))) for c in g.get_xticks()])
        plt.savefig(os.path.join(args.input_dir, lossname + ".pdf"), format="pdf")

    for loss in ["loss_std_gold", "loss_ugc_gold", "valid_distil_loss"]:
        plot_loss(loss)



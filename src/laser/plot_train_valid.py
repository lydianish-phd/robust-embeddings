import os, configargparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from plot import checkpoint_display_name

input_dir = "/gpfsscratch/rech/ncm/udc54vm/experiments/robust-embeddings/laser/experiment_025_valid/scores"

all_scores = pd.read_csv(os.path.join(input_dir, "all_scores" + ".csv" ))
#all_scores.drop(all_scores[all_scores['model'] == 'roberta-init'].index, inplace=True)
#all_scores.drop(all_scores[all_scores['model'] == 'c-roberta-init'].index, inplace=True)

def plot_loss(lossname):
    plt.clf()
    g = sns.lineplot(all_scores[all_scores['steps'] < 700000], x='steps', y=lossname, hue='model')
    g.set_xticklabels([checkpoint_display_name(str(int(c))) for c in g.get_xticks()])
    plt.savefig(os.path.join(input_dir, lossname + ".pdf"), format="pdf")

for loss in ["loss_std_gold", "loss_ugc_gold", "valid_distil_loss"]:
    plot_loss(loss)



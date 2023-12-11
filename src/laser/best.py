import os, argparse
import pandas as pd
import numpy as np
from score import model_display_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="path to ugc directory with outputs of different seeds", type=str)
    args = parser.parse_args()

    models = [ f.name for f in os.scandir(args.input_dir) if f.is_dir() and f.name != "laser" ]

    all_scores = pd.DataFrame(columns=["model", "epoch", "steps", "loss_std_gold", "loss_ugc_gold", "valid_distil_loss" ])

    for model in models:
        scores = pd.read_csv(os.path.join(args.input_dir, model, "train_valid.csv"))
        scores["model"] = np.repeat(model_display_name(model), scores.shape[0])
        all_scores = pd.concat([all_scores, scores], ignore_index=True)

    idx_best_valid_distil_loss = all_scores.groupby("model")["valid_distil_loss"].idxmin()
    idx_best_loss_std_gold = all_scores.groupby("model")["loss_std_gold"].idxmin()
    idx_best_loss_ugc_gold = all_scores.groupby("model")["loss_ugc_gold"].idxmin()

    all_scores.loc[idx_best_valid_distil_loss].to_csv(os.path.join(args.input_dir, "idx_best_valid_distil_loss" + ".csv" ))
    all_scores.loc[idx_best_loss_std_gold].to_csv(os.path.join(args.input_dir, "idx_best_loss_std_gold" + ".csv" ))
    all_scores.loc[idx_best_loss_ugc_gold].to_csv(os.path.join(args.input_dir, "idx_best_loss_ugc_gold" + ".csv" ))
    all_scores.to_csv(os.path.join(args.input_dir, "all_scores" + ".csv" ))



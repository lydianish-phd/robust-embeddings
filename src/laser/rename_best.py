import os,argparse, shutil
import pandas as pd
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--basedir", help="path to experiment directory", type=str)
    parser.add_argument("-s", "--seed", help="validation seed", type=int)
    args = parser.parse_args()

    best_scores = pd.read_csv(os.path.join(f"{args.basedir}_valid", str(args.seed), "scores", "idx_best_valid_distil_loss.csv"))
    
    for _, row in best_scores.iterrows():
        if row['model'] == "roberta-maxpool":
            experiment_dir = args.basedir
        elif row['model'] == "roberta-maxpool-init":
            experiment_dir = f"{args.basedir}b"
        elif row['model'] == "roberta-meanpool":
            experiment_dir = f"{args.basedir}c"
        elif row['model'] == "roberta-meanpool-init":
            experiment_dir = f"{args.basedir}d"
        elif row['model'] == "c-roberta-maxpool":
            experiment_dir = f"{args.basedir}e"
        elif row['model'] == "c-roberta-maxpool-init":
            experiment_dir = f"{args.basedir}f"
        elif row['model'] == "c-roberta-meanpool":
            experiment_dir = f"{args.basedir}g"
        elif row['model'] == "c-roberta-meanpool-init":
            experiment_dir = f"{args.basedir}h"
        elif row['model'] == "roberta-cls":
            experiment_dir = f"{args.basedir}i"
        elif row['model'] == "roberta-cls-init":
            experiment_dir = f"{args.basedir}j"
        elif row['model'] == "c-roberta-cls":
            experiment_dir = f"{args.basedir}k"
        elif row['model'] == "c-roberta-cls-init":
            experiment_dir = f"{args.basedir}l"
        elif row['model'] == "roberta-maxpool-init-0.1":
            experiment_dir = args.basedir
        elif row['model'] == "roberta-maxpool-init-0.2":
            experiment_dir = f"{args.basedir}b"
        elif row['model'] == "roberta-maxpool-init-0.3":
            experiment_dir = f"{args.basedir}c"
        elif row['model'] == "roberta-maxpool-init-0.4":
            experiment_dir = f"{args.basedir}d"
        elif row['model'] == "roberta-maxpool-init-0.5":
            experiment_dir = f"{args.basedir}e"
        elif row['model'] == "roberta-maxpool-init-0":
            experiment_dir = f"{args.basedir}f"

        best_checkpoint = os.path.join(experiment_dir, "models", f"checkpoint_{row['epoch']}_{row['steps']}.pt")
        if not os.path.exists(best_checkpoint):
            best_checkpoint = os.path.join(experiment_dir, "models", f"checkpoint{row['epoch']}.pt")
        shutil.copy2(best_checkpoint, os.path.join(experiment_dir, "models", "checkpoint_best.pt"))

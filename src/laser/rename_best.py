import os,argparse, shutil
import pandas as pd

def get_experiment_dir(model_name, base_dir):
    
    experiment_dir_dict = {
        "roberta-maxpool": base_dir,
        "roberta-maxpool-init": f"{base_dir}b",
        "roberta-meanpool": f"{base_dir}c",
        "roberta-meanpool-init": f"{base_dir}d",
        "c-roberta-maxpool": f"{base_dir}e",
        "c-roberta-maxpool-init": f"{base_dir}f",
        "c-roberta-meanpool": f"{base_dir}g",
        "c-roberta-meanpool-init": f"{base_dir}h",
        "roberta-cls": f"{base_dir}i",
        "roberta-cls-init": f"{base_dir}j",
        "c-roberta-cls": f"{base_dir}k",
        "c-roberta-cls-init": f"{base_dir}l",
        "roberta-maxpool-init-0.1": base_dir,
        "roberta-maxpool-init-0.2": f"{base_dir}b",
        "roberta-maxpool-init-0.3": f"{base_dir}c",
        "roberta-maxpool-init-0.4": f"{base_dir}d",
        "roberta-maxpool-init-0.5": f"{base_dir}e",
        "roberta-maxpool-init-0": f"{base_dir}f",
        "roberta-maxpool-init-0.35": f"{base_dir}g",
        "roberta-maxpool-init-0.45": f"{base_dir}h",
        "roberta-maxpool-init-0.6": f"{base_dir}i",
        "RoLASER-0.1": base_dir,
        "RoLASER-0.2": f"{base_dir}b",
        "RoLASER-0.3": f"{base_dir}c",
    }
    
    return experiment_dir_dict[model_name]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--basedir", help="path to experiment directory", type=str)
    parser.add_argument("-s", "--seed", help="validation seed", type=int)
    args = parser.parse_args()

    best_scores = pd.read_csv(os.path.join(f"{args.basedir}_valid", str(args.seed), "scores", "idx_best_valid_distil_loss.csv"))
    
    for _, row in best_scores.iterrows():
        experiment_dir = get_experiment_dir(row['model'], args.basedir)
        best_checkpoint = os.path.join(experiment_dir, "models", f"checkpoint_{row['epoch']}_{row['steps']}.pt")
        if not os.path.exists(best_checkpoint):
            best_checkpoint = os.path.join(experiment_dir, "models", f"checkpoint{row['epoch']}.pt")
        shutil.copy2(best_checkpoint, os.path.join(experiment_dir, "models", "checkpoint_best.pt"))

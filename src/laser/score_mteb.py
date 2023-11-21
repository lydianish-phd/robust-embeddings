import os, json, argparse
import pandas as pd
from mteb_tasks import (
    TASK_LIST_S2S_ENGLISH, 
    TASK_LIST_PAIR_CLASSIFICATION, 
    TASK_LIST_CLASSIFICATION,
    TASK_LIST_STS
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LASER: Evaluate on MTEB benchmarks")
    parser.add_argument(
        "-i", "--input-dir", required=True, help="Directory to MTEB json score files"
    )
    args = parser.parse_args()
    models = [ f.name for f in os.scandir(args.input_dir) if f.is_dir() ]
    tasks_lists = {
        "pair_classification": TASK_LIST_PAIR_CLASSIFICATION,
        "classification": TASK_LIST_CLASSIFICATION,
        "sts": TASK_LIST_STS
    }

    print("Averaging scores across all models...")

    all_scores_df = pd.DataFrame(columns=["Model", "Average"] + TASK_LIST_S2S_ENGLISH)
    all_scores = {}
    for model in models:
        all_scores[model] = {}
        for task in tasks_lists:
            with open(os.path.join(args.input_dir, model, f"scores_{task}.json")) as f:
                task_scores = json.load(f)
            del task_scores["Average"]
            all_scores[model].update(task_scores)
    
    all_scores_df[TASK_LIST_S2S_ENGLISH] = pd.DataFrame.from_records(list(all_scores.values()))[TASK_LIST_S2S_ENGLISH]
    all_scores_df["Average"] = all_scores_df[TASK_LIST_S2S_ENGLISH].mean(axis=1)
    all_scores_df["Model"] = models
    all_scores_df.set_index(["Model"], inplace=True)
    all_scores_df.to_csv(os.path.join(args.input_dir, "scores_s2s_english.csv"), float_format='%.2f')

    print("Done...")
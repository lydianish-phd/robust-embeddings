import os, argparse, json
import pandas as pd
import numpy as np
from utils import (
    SCORE_FILE_SUFFIX,
    MODEL_NAMES,
    COLUMN_NAME_SEPARATOR,
    STAT_NAMES,
    STATS_FILE_PREFIX
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="path to experiment directory", type=str)
    parser.add_argument("-t", "--table-name", type=str)
    parser.add_argument("-c", "--corpora", type=str, nargs="+")
    parser.add_argument("-l", "--lang-pairs", type=str, nargs="+")
    parser.add_argument("-m", "--models", type=str, nargs="+")
    args = parser.parse_args()

    all_scores = { key: { "model": [ MODEL_NAMES[model] for model in args.models ] } for key in STATS }

    print(f"Aggregating {args.table_name} stats...")

    for corpus in args.corpora:
        for lang_pair in args.lang_pairs:
            for model in args.models:
                model_output_dir = os.path.join(args.input_dir, "outputs", model, corpus, lang_pair)
                if os.path.isdir(model_output_dir):
                    scores_files = [ f.path for f in os.scandir(model_output_dir) if (f.name.endswith(SCORE_FILE_SUFFIX) and f.name.startswith(STATS_FILE_PREFIX)) ]
                    for score_file in scores_files:
                        file_name = os.path.basename(score_file).removesuffix(SCORE_FILE_SUFFIX).removeprefix(STATS_FILE_PREFIX)
                        column_name = COLUMN_NAME_SEPARATOR.join([corpus, lang_pair, file_name])
                        with open(score_file) as f:
                            scores = json.load(f)
                        for stat, stat_name in STAT_NAMES.items():
                            if column_name in all_scores[stat]:
                                all_scores[stat][column_name].append(scores[stat_name])
                            else:
                                all_scores[stat][column_name] = [scores[stat_name]]

    print("Writing aggregated score files...")
    scores_dir = os.path.join(args.input_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)
    for stat in STAT_NAMES:
        score_file = os.path.join(scores_dir, f"{stat}_{args.table_name}.csv")
        scores_df = pd.DataFrame.from_dict(all_scores[stat]).set_index("model").T
        scores_df.to_csv(score_file, index=False)

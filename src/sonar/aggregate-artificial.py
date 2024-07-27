import os, argparse, json
import pandas as pd
import numpy as np
from utils import (
    SCORE_FILE_SUFFIX,
    MODEL_NAMES,
    COLUMN_NAME_SEPARATOR,
    BLEU_ROUND_DECIMALS,
    COMET_ROUND_DECIMALS
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="path to experiment directory", type=str)
    parser.add_argument("-t", "--table-name", type=str)
    parser.add_argument("-c", "--corpus", type=str)
    parser.add_argument("-l", "--lang-pairs", type=str, nargs="+")
    parser.add_argument("-m", "--models", type=str, nargs="+")
    parser.add_argument("-s", "--seeds", type=int, nargs="+")
    parser.add_argument("-p", "--probas", type=float, nargs="+")
    args = parser.parse_args()

    bleu_scores = {
        "model": [],
        "seed": [],
        "proba": []
    }
    comet_scores = {
        "model": [],
        "seed": [],
        "proba": []
    }

    total_files = len(args.seeds) * len(args.probas) * len(args.lang_pairs) * len(args.models)
    print("Total files:", total_files)

    print(f"Aggregating {args.table_name} scores...")
    for model in args.models:
        print("Model:", model)
        for seed in args.seeds:
            print("\t - seed:", seed)
            for proba in args.probas:
                print("\t\t - proba:", proba)

                bleu_scores["model"].append(MODEL_NAMES[model])
                bleu_scores["seed"].append(seed)
                bleu_scores["proba"].append(proba)
                comet_scores["model"].append(MODEL_NAMES[model])
                comet_scores["seed"].append(seed)
                comet_scores["proba"].append(proba)

                for lang_pair in args.lang_pairs:
                    print("\t\t\t - lang_pair:", lang_pair)
                    model_output_dir = os.path.join(args.input_dir, "outputs", model, args.corpus, lang_pair, str(seed), str(proba))
                    if os.path.isdir(model_output_dir):
                        scores_files = [ f.path for f in os.scandir(model_output_dir) if f.name.endswith(SCORE_FILE_SUFFIX)]
                        for score_file in scores_files:
                            file_name = os.path.basename(score_file).removesuffix(SCORE_FILE_SUFFIX)
                            column_name = COLUMN_NAME_SEPARATOR.join([args.corpus, lang_pair, file_name])
                            with open(score_file) as f:
                                scores = json.load(f)
                            if column_name in bleu_scores:
                                bleu_scores[column_name].append(scores["bleu"])
                            else:
                                bleu_scores[column_name] = [scores["bleu"]]
                            if column_name in comet_scores:
                                comet_scores[column_name].append(scores["comet"])
                            else:
                                comet_scores[column_name] = [scores["comet"]]
    
    print(f"Writing aggregated score files...")
    scores_dir = os.path.join(args.input_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)
    bleu_score_file = os.path.join(scores_dir, f"bleu_{args.table_name}.csv")
    comet_score_file = os.path.join(scores_dir, f"comet_{args.table_name}.csv")
    
    bleu_scores_df = pd.DataFrame.from_dict(bleu_scores)
    comet_scores_df = pd.DataFrame.from_dict(comet_scores)

    score_columns = [col for col in bleu_scores_df.columns if COLUMN_NAME_SEPARATOR in col]
    bleu_scores_df["avg"] = bleu_scores_df[np.array(score_columns)].mean(axis=1)
    comet_scores_df["avg"] = comet_scores_df[np.array(score_columns)].mean(axis=1)

    bleu_scores_df.round(BLEU_ROUND_DECIMALS).to_csv(bleu_score_file)
    comet_scores_df.round(COMET_ROUND_DECIMALS).to_csv(comet_score_file)
        

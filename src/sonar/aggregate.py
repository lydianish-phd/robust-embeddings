import os, argparse, json
import pandas as pd
from utils import SCORE_FILE_SUFFIX, MODEL_NAMES, COLUMN_NAME_SEPARATOR

def multilingual_average(scores):
    unique_files_names = set()
    for name in scores.columns:
        if COLUMN_NAME_SEPARATOR in name:
            corpus, _ , name = name.split(COLUMN_NAME_SEPARATOR)
            unique_files_names.add((corpus, name))
    for corpus, name in unique_files_names:
        column_names = [col for col in scores.columns if name in col]
        avg_column_name = COLUMN_NAME_SEPARATOR.join(["avg", corpus, name])
        scores[avg_column_name] = scores[column_names].mean(axis=1)
    return scores

def multilingual_delta(scores, lang_pairs, ugc_file_name="raw.en.test", std_file_name="norm.en.test"):
    column_names = [col for col in scores.columns if ugc_file_name in col]
    for ugc_col in column_names:
        col_name_prefix = ugc_col.removesuffix(ugc_file_name).strip(COLUMN_NAME_SEPARATOR)
        std_col = COLUMN_NAME_SEPARATOR.join([col_name_prefix, std_file_name])
        delta_column_name = COLUMN_NAME_SEPARATOR.join(["delta", col_name_prefix])
        scores[delta_column_name] = scores[ugc_col] - scores[std_col]
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="path to experiment directory", type=str)
    parser.add_argument("-t", "--table-name", type=str)
    parser.add_argument("-c", "--corpora", type=str, nargs="+")
    parser.add_argument("-l", "--lang-pairs", type=str, nargs="+")
    parser.add_argument("-m", "--models", type=str, nargs="+")
    args = parser.parse_args()

    bleu_scores = {
        "model": [ MODEL_NAMES[model] for model in args.models ]
    }
    comet_scores = {
        "model": [ MODEL_NAMES[model] for model in args.models ]
    }

    print(f"Aggregating {args.table_name} scores...")
    
    for corpus in args.corpora:
        for lang_pair in args.lang_pairs:
            for model in args.models:
                model_output_dir = os.path.join(args.input_dir, "outputs", model, corpus, lang_pair)
                if os.path.isdir(model_output_dir):
                    scores_files = [ f.path for f in os.scandir(model_output_dir) if f.name.endswith(SCORE_FILE_SUFFIX)]
                    for score_file in scores_files:
                        file_name = os.path.basename(score_file).removesuffix(SCORE_FILE_SUFFIX)
                        column_name = COLUMN_NAME_SEPARATOR.join([corpus, lang_pair, file_name])
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

    
    print("Writing aggregated score files...")
    scores_dir = os.path.join(args.input_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)
    bleu_score_file = os.path.join(scores_dir, f"bleu_{args.table_name}.csv")
    comet_score_file = os.path.join(scores_dir, f"comet_{args.table_name}.csv")

    bleu_scores_df = pd.DataFrame.from_dict(bleu_scores)
    comet_scores_df = pd.DataFrame.from_dict(comet_scores)

    if args.table_name == "multilingual":
        bleu_scores_df = multilingual_delta(multilingual_average(bleu_scores_df), args.lang_pairs)
        comet_scores_df = multilingual_delta(multilingual_average(comet_scores_df), args.lang_pairs)

    bleu_scores_df.to_csv(bleu_score_file)
    comet_scores_df.to_csv(comet_score_file)


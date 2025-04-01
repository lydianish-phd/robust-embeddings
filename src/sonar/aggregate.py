import os, argparse, json
import pandas as pd
import numpy as np
from scipy.stats import ttest_1samp
from utils import (
    SCORE_FILE_SUFFIX,
    GPT_NORM_FILE_PREFIX,
    MODEL_NAMES,
    METRIC_NAMES,
    COLUMN_NAME_SEPARATOR,
    ROUND_DECIMALS,
    ROCSMT_NORM_FILE_NAME,
    ROCSMT_RAW_FILE_NAME,
    FLORES_FILE_NAME,
    ROCSMT_CORPUS_NAME,
    FLORES_CORPUS_NAME,
    MULTILINGUAL_COLUMNS
)

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

def multilingual_delta(scores, ugc_file_name=ROCSMT_RAW_FILE_NAME, norm_file_name=ROCSMT_NORM_FILE_NAME, std_file_name=FLORES_FILE_NAME):
    ugc_column_names = [col for col in scores.columns if ugc_file_name in col]
    for ugc_col in ugc_column_names:
        col_name_prefix = ugc_col.removesuffix(ugc_file_name).strip(COLUMN_NAME_SEPARATOR)
        norm_col = COLUMN_NAME_SEPARATOR.join([col_name_prefix, norm_file_name])
        delta_column_name = COLUMN_NAME_SEPARATOR.join(["delta", col_name_prefix])
        scores[delta_column_name] = scores[ugc_col] - scores[norm_col]
    
    std_column_names = [col for col in scores.columns if std_file_name in col]
    for std_col in std_column_names:
        col_name_prefix = std_col.removesuffix(std_file_name).strip(COLUMN_NAME_SEPARATOR)
        norm_col = COLUMN_NAME_SEPARATOR.join([col_name_prefix.replace(FLORES_CORPUS_NAME, ROCSMT_CORPUS_NAME), norm_file_name])
        delta_column_name = COLUMN_NAME_SEPARATOR.join(["delta", col_name_prefix])
        scores[delta_column_name] = scores[norm_col] - scores[std_col]
    return scores

def statistical_significance(scores, column_name_prefixes, p_value_threshold=0.05):
    for column_name_prefix in column_name_prefixes:
        columns = scores.columns[scores.columns.str.startswith(column_name_prefix)]
        p_values = np.array([ ttest_1samp(scores[scores["model"] == model][columns].values.flatten(), 0)[1] for model in scores["model"] ])
        signif_column_name = column_name_prefix + COLUMN_NAME_SEPARATOR + "significant"
        scores[signif_column_name] = p_values < p_value_threshold
    return scores

def add_score(scores, column_name, score):
    if column_name in scores:
        scores[column_name].append(score)
    else:
        scores[column_name] = [score]

def get_multilingual_table(scores_df):
    return multilingual_delta(multilingual_average(scores_df).round(ROUND_DECIMALS))[MULTILINGUAL_COLUMNS]

def get_score_files(output_dir):
    return [ f.path for f in os.scandir(output_dir) if (f.name.endswith(SCORE_FILE_SUFFIX) and not f.name.startswith(GPT_NORM_FILE_PREFIX)) ]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="path to experiment directory", type=str)
    parser.add_argument("-t", "--table-name", type=str)
    parser.add_argument("-c", "--corpora", type=str, nargs="+")
    parser.add_argument("-l", "--lang-pairs", type=str, nargs="+")
    parser.add_argument("-m", "--models", type=str, nargs="+")
    parser.add_argument("--metrics", type=str, nargs="+", default=METRIC_NAMES.keys())
    args = parser.parse_args()

# TO DO: Remove redundant code by using a single object with the metrics as keys
    
    all_scores = { metric: { "model": [ MODEL_NAMES[model] for model in args.models ] } for metric in args.metrics }

    print(f"Aggregating {args.table_name} scores...")

    for corpus in args.corpora:
        for lang_pair in args.lang_pairs:
            for model in args.models:
                model_output_dir = os.path.join(args.input_dir, "outputs", model, corpus, lang_pair)
                if os.path.isdir(model_output_dir):
                    scores_files = get_score_files(model_output_dir)
                    for score_file in scores_files:
                        file_name = os.path.basename(score_file).removesuffix(SCORE_FILE_SUFFIX)
                        column_name = COLUMN_NAME_SEPARATOR.join([corpus, lang_pair, file_name])
                        with open(score_file) as f:
                            scores = json.load(f)
                        for metric in args.metrics:
                            if metric in scores:
                                add_score(all_scores[metric], column_name, scores[metric])

    
    print("Writing aggregated score files...")
    scores_dir = os.path.join(args.input_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)

    score_files = { metric: os.path.join(scores_dir, f"{metric}_{args.table_name}.csv") for metric in args.metrics }

    all_scores_df = { metric: pd.DataFrame.from_dict(all_scores[metric]) for metric in args.metrics }
 
    if "comet" in args.metrics:
        columns_to_multiply = all_scores_df["comet"].columns[all_scores_df["comet"].columns.str.contains(COLUMN_NAME_SEPARATOR)]
        all_scores_df["comet"][columns_to_multiply] *= 100
        
    
    if args.table_name == "multilingual":
        for metric in args.metrics:
            all_scores_df[metric] = get_multilingual_table(all_scores_df[metric])

    delta_column_prefixes = [ "delta" + COLUMN_NAME_SEPARATOR + corpus for corpus in args.corpora ]
    
    for metric in args.metrics:
        all_scores_df[metric] = statistical_significance(all_scores_df[metric], delta_column_prefixes, 0.05)
        all_scores_df[metric].round(ROUND_DECIMALS).to_csv(score_files[metric])

import os, argparse, json
import pandas as pd

SCORE_FILE_SUFFIX = ".out.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", help="path to experiment directory", type=str)
    parser.add_argument("-t", "--table-name", type=str)
    parser.add_argument("-c", "--corpora", type=str, nargs="+")
    parser.add_argument("-l", "--lang-pairs", type=str, nargs="+")
    parser.add_argument("-m", "--models", type=str, nargs="+")
    parser.add_argument("-s", "--seeds", type=int, nargs="+")
    parser.add_argument("-p", "--probas", type=float, nargs="+")
    args = parser.parse_args()

    bleu_scores = {
        "model": args.models,
        "seed": [],
        "proba": []
    }
    comet_scores = {
        "model": args.models,
        "seed": [],
        "proba": []
    }

    print(f"Aggregating {args.table_name} scores...")
    for seed in args.seeds:
        for proba in args.probas:
            for corpus in args.corpora:
                for lang_pair in args.lang_pairs:
                    for model in args.models:
                        model_output_dir = os.path.join(args.input_dir, "outputs", model, corpus, lang_pair, str(seed), str(proba))
                        if os.path.isdir(model_output_dir):
                            scores_files = [ f.path for f in os.scandir(model_output_dir) if f.name.endswith(SCORE_FILE_SUFFIX)]
                            for score_file in scores_files:
                                file_name = os.path.basename(score_file).removesuffix(SCORE_FILE_SUFFIX)
                                column_name = "__".join([corpus, lang_pair, file_name])
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
                                bleu_scores["seed"].append(seed)
                                comet_scores["seed"].append(seed)
                                bleu_scores["proba"].append(proba)
                                comet_scores["proba"].append(proba)
    
    print(f"Writing aggregated score files...")
    scores_dir = os.path.join(args.input_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)
    bleu_score_file = os.path.join(scores_dir, f"bleu_{args.table_name}.csv")
    comet_score_file = os.path.join(scores_dir, f"comet_{args.table_name}.csv")

    bleu_scores_df = pd.DataFrame.from_dict(bleu_scores).set_index("model")
    comet_scores_df = pd.DataFrame.from_dict(comet_scores).set_index("model")

    bleu_scores_df.round(2).to_csv(bleu_score_file)
    comet_scores_df.round(3).to_csv(comet_score_file)
        

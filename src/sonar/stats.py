import os, argparse, json
import sentencepiece as spm
import numpy as np
import re
from utils import STATS_FILE_SUFFIX, STATS_FILE_PREFIX
from lexical_diversity import lex_div as ld

def find_usernames_hashtags_urls(text):
    # Regular expression patterns
    url_pattern = r'https?://[^\s]+'              # Matches URLs starting with http or https
    username_pattern = r'@\w+'                    # Matches usernames starting with '@' and followed by word characters
    hashtag_pattern = r'#\w+'                     # Matches hashtags starting with '#' and followed by word characters

    return {
        "urls": re.findall(url_pattern, text), 
        "usernames": re.findall(username_pattern, text), 
        "hashtags": re.findall(hashtag_pattern, text)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-files", type=str, required=True, help="Path to the input files", nargs="+")
    parser.add_argument("-m", "--spm-model", type=str, required=True, help="Path to the sentencepiece model")
    parser.add_argument("-w", "--mattr-window-size", type=int, default=1000, help="Window size for Moving Average TTR calculation")
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    
    for input_file in args.input_files:
        output_dir, filename = os.path.split(input_file)
        output_file = os.path.join(output_dir, STATS_FILE_PREFIX + filename + STATS_FILE_SUFFIX)

        words = []
        types = set()
        tokens = []
        sentence_lengths = []
        urls = []
        usernames = []
        hashtags = []

        print(f"Calculating stats for {input_file}...")
        n_lines = 0
        with open(input_file, "r") as f:
            for line in f:
                words.extend(line.split())
                line_tokens = sp.encode(line)
                tokens.extend(line_tokens)
                types.update(line_tokens)
                sentence_lengths.append(len(line_tokens))
                results = find_usernames_hashtags_urls(line)
                urls.extend(results["urls"])
                usernames.extend(results["usernames"])
                hashtags.extend(results["hashtags"])
                n_lines += 1
        
        
        stats = {
            "lines": n_lines,
            "fertility": len(tokens) / len(words),
            "types": len(types),
            "tokens": len(tokens),
            "ttr": len(types) / len(tokens),
            "urls": len(urls),
            "usernames": len(usernames),
            "hashtags": len(hashtags),
            "urls_per_line": len(urls) / n_lines,
            "usernames_per_line": len(usernames) / n_lines,
            "hashtags_per_line": len(hashtags) / n_lines,
            "average_sentence_length": np.mean(sentence_lengths),
            "stddev_sentence_length": np.std(sentence_lengths),
            "mattr": ld.mattr(tokens, args.mattr_window_size),
            "hdd": ld.hdd(tokens),
        }
        print(stats)

        with open(output_file, "w") as f:
            json.dump(stats, f, indent=4)
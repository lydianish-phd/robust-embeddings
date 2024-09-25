import os, argparse, json
import sentencepiece as spm
import numpy as np
import re

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
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=args.spm_model)
    
    for input_file in args.input_files:
        output_dir, filename = os.path.split(input_file)
        output_file = os.path.join(output_dir, f"stats.{filename}.json")

        words = []
        types = set()
        tokens = []
        urls = []
        usernames = []
        hashtags = []

        print(f"Calculating stats for {input_file}...")

        with open(input_file, "r") as f:
            for line in f:
                words.extend(line.split())
                line_tokens = sp.encode(line)
                tokens.extend(line_tokens)
                types.update(line_tokens)
                results = find_usernames_hashtags_urls(line)
                urls.extend(results["urls"])
                usernames.extend(results["usernames"])
                hashtags.extend(results["hashtags"])
        
        
        stats = {
            "fertility": len(tokens) / len(words),
            "types": len(types),
            "tokens": len(tokens),
            "ttr": len(types) / len(tokens),
            "urls": len(urls),
            "usernames": len(usernames),
            "hashtags": len(hashtags)
        }
        print(stats)

        with open(output_file, "w") as f:
            json.dump(stats, f)
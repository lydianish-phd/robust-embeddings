import os, argparse
import re
from nltk.tokenize.treebank import TreebankWordDetokenizer

def detokenize_hashtags_usernames(sentence):
    hashtag_pattern = re.compile(r'#\s*\w+')
    username_pattern = re.compile(r'@\s*\w+')
    
    # Function to remove spaces from the matched tokenized string
    def detokenize(match):
        return match.group(0).replace(' ', '')
    
    sentence = hashtag_pattern.sub(detokenize, sentence)
    sentence = username_pattern.sub(detokenize, sentence)
    
    return sentence

def remove_extra_spaces(sentence):
    # Remove spaces before full stops
    sentence = re.sub(r'\s+\.', '.', sentence)
    # Remove spaces around all other slashes
    sentence = re.sub(r'\s*/\s*', '/', sentence)
    return sentence

def insert_missing_spaces(sentence):
    # Insert a space before hashtags and usernames that are stuck to a previous character
    sentence = re.sub(r'(?<!^)(?<!\s)(#\w+)', r' \1', sentence)
    sentence = re.sub(r'(?<!^)(?<!\s)(@\w+)', r' \1', sentence)
    # Fix emoji that was ruined by previous rule
    sentence = re.sub(r'@- @', '@-@ ', sentence)
    # Keep "w/" as abbreviation for "with"
    sentence = re.sub(r'\bw\s*/(?!o\b)', 'w/ ', sentence)

    return sentence

def detokenize_tweet(tweet):
    tweet = detokenize_hashtags_usernames(tweet)
    detokenizer = TreebankWordDetokenizer()
    detokenized_tweet = detokenizer.detokenize(tweet.split())
    detokenized_tweet = remove_extra_spaces(detokenized_tweet)
    detokenized_tweet = insert_missing_spaces(detokenized_tweet)
    return detokenized_tweet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-file", type=str, required=True, help="Input file containing tokenized tweets")
    args = parser.parse_args()

    base_dir, file_name = os.path.split(args.input_file)
    output_file = os.path.join(base_dir, f"detok.{file_name}")

    with open(args.input_file, 'r') as f, open(output_file, 'w') as out_f:
        for line in f:
            detokenized_tweet = detokenize_tweet(line.strip())
            out_f.write(detokenized_tweet + '\n')
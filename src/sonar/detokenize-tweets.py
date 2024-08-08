import os, argparse
import re
from nltk.tokenize.treebank import TreebankWordDetokenizer

def detokenize_urls_hashtags_usernames(sentence):
    # Define regex patterns for tokenized URLs, hashtags, and usernames
    url_pattern = re.compile(r'https?\s*:\s*/\s*/\s*t\.co\s*/\s*\w+')
    hashtag_pattern = re.compile(r'#\s*\w+')
    username_pattern = re.compile(r'@\s*\w+')
    
    # Function to remove spaces from the matched tokenized string
    def detokenize(match):
        return match.group(0).replace(' ', '')
    
    # Substitute all tokenized URLs, hashtags, and usernames in the sentence with their detokenized versions
    sentence = url_pattern.sub(detokenize, sentence)
    sentence = hashtag_pattern.sub(detokenize, sentence)
    sentence = username_pattern.sub(detokenize, sentence)
    
    return sentence

def detokenize_tweet(tweet):
    # Detokenize URLs, hashtags, and usernames
    tweet = detokenize_urls_hashtags_usernames(tweet)
    
    # Detokenize the tweet using the TreebankWordDetokenizer
    detokenizer = TreebankWordDetokenizer()
    detokenized_tweet = detokenizer.detokenize(tweet.split())
    
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
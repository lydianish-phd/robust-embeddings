import torch, os, argparse
import numpy as np
import pandas as pd
from embed import embed_sentences
from score import read_embeddings
from sklearn.decomposition import PCA
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import spearmanr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("")

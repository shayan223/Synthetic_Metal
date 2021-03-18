# Reference for implementing BLEU scores evaluations
# https://machinelearningmastery.com/calculate-bleu-score-for-text-python/

import json
import os
import re
import pandas as pd
from statistics import mean
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import sentence_bleu

# Loads the filtered songs to be used as the reference
def load_references (file_path):
    refs = []
    with open(file_path) as f:
        songs = json.load(f)['songs']
    for song in songs:
        refs.append(song.split())
    return refs

# Loads the baseline sample songs
def load_baseline (file_path):
    cands = []
    baseline_df = pd.read_csv(file_path)
    for ind in baseline_df.index:
        song = baseline_df.iloc[ind][1]
        cands.append(song.split()) 
    #print(baseline_df.iloc[1][1]) 
    return cands

# Loads the sample songs to be used as the candidates
def load_candidates (dir_path):
    cands = []
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        songs = load_cand(file_path)
        for song in songs:
            if len(song) > 15:
                cands.append(song)
    return cands

# Filters each song from the sample files
def load_cand (file_path):
    cands = []
    with open(file_path, 'r') as f:
        file_str = f.read()
        file_str = re.sub(pattern = '^.*.=\n', repl = '', string = file_str, count = 1)
        file_str = re.sub(pattern = '\<\|song\|\>', repl = '', string = file_str)
        songs = file_str.split('<|endoftext|>')
        for song in songs:
            cands.append(song.split())
    return cands 

# Calculates the bleu score using the reference and candidates
def calculate_bleu (refs, cands, size):
    print("Started calculating Bleu scores")
    bleu_scores = []
    print("# of songs:", len(cands))
    print("Only using the last", len(cands[-size:]), "songs")
    for i, cand in enumerate(cands[-size:]):
        print("On song:", i+1)
        score = corpus_bleu([refs], [cand], weights=(.10, .20, .70, .0))
        print("Score for song", i+1, "is:", score)
        bleu_scores.append(score)
    return bleu_scores

refs = load_references("data/filtered-songs.txt")
print("Loaded References")
#cands = load_candidates('data/small_1') # Uses small generated samples
#cands = load_baseline("data/baseline_sample_output.csv") # Uses baseline samples
cands = load_candidates('data/medium_samples') # Uses medium generated samples
print("Loaded Candidates")
bleu_scores = calculate_bleu(refs, cands, 250)
print("Avg BLEU score:", mean(bleu_scores))


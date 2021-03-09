import numpy as np
import pandas as pd
import nltk
import os
from os import path
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import json


DATA_DIR = '../data/metal_lyrics'
nltk.download('words')
nltk.download('punkt')


# data loaded from ../data/filtered-songs.txt
data = []

def filter_data():
    global data
    tokenizer = nltk.LineTokenizer()

    for subdir, dirs, files, in tqdm(os.walk(DATA_DIR)):
        # Skip the formatting directory.
        if (os.path.basename(subdir) == '.idea'):
            continue
        for file in files:
            filePath = subdir + os.sep + file
            try:
                data.append(open(filePath).read())
            except:
                continue
    # Get rid of another formatting string that gets read in.
    data.pop(0)

    # Tokenize data line by line.
    # Tokens contains at each index, a list of strings, each string being a line of a song.
    tokens = []

    # Stop-list of non-english tokens commonly seen in foreign language metal songs.
    filter_tokens = ['ich', 'en', 'se', 'z', 'Z', 'bir', 'ruh', 'ein', 'es', 'zur', 'te', 'y', 'di']
    filter_whole = ['Ã']

    # Builds the filtered song list.
    for song in tqdm(data):
        add_song = True
        tks = word_tokenize(song)
        if '[Instrumental]' in song or '[instrumental]' in song:
            add_song = False
        for f in filter_whole:
            if f in song:
                add_song = False
        if add_song:
            for f in filter_tokens:
                if f in tks:
                    add_song = False
            if add_song:
                if song != "\n":
                    tokens.append(song)

    # Saves off filtered song list to file.
    f_data = {'songs': tokens}
    with open('../data/filtered-songs.txt', 'w') as f_out:
        json.dump(f_data, f_out)
    return tokens

def load_data():
    global data
    fpath = "../data/filtered-songs.txt"

    # Load the filtered song list if it exists, read over corpus and create it.
    if path.exists(fpath):
        with open(fpath) as f:
            data = json.load(f)['songs']
    else:
        data = filter_data()
    print('Loaded/Pre-processed {} songs!'.format(len(data)))
    return data

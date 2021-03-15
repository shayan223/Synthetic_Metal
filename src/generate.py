from transformers import GPT2Tokenizer
import filter
import torch
from tqdm import tqdm
import os
from os import path
import json
from torch.utils.data import Dataset, DataLoader
import pickle

gpt2_small_chunks = 768
gpt2_medium_chunks = 1024
gpt2_large_chunks = 1280
gpt2_xl_chunks = 1600


# Returns the a tagged corpus of song data for use with the GPT2 transformer model
def get_tagged_data(data, control_code="song", gpt2_type="gpt2", model_size=gpt2_small_chunks,
                   load_from_file=True, save_to_file=True, remove_songs_over_limit=False):

    tpath = "../data/corpus_tagged.txt"
    tagged_corpus = ''
    songs = []
    loaded = False

    def load_tagged_data():
        # Load the filtered song list if it exists, read over corpus and create it.
        if path.exists(tpath):
            with open(tpath, 'r') as f:
                songs = f.read()
            #with open(tpath, 'rb') as f:
                #songs = pickle.load(f)
                return True, songs
        return False, []

    # Create the tagged data and save it to a file if no file currently exists or we aren't loading from file.
    loaded, songs  = load_tagged_data()
    if not loaded or not load_from_file:
        # Builds single corpus text with tags included
        print("\nUnable to load tagged corpus. Please wait while corpus is being built...")
        print("\nApplying tags...")
        for song in tqdm(data, position=0, leave=True):
            if len(song) > model_size and remove_songs_over_limit:
                continue
            else:
                songs.append('<|{}|>{}<|endoftext|>\n'.format(control_code, song))

        # Join all tagged songs into one large text string
        print("\nConsolidating songs...")
        tagged_corpus = ''.join(songs)
        #for song in tqdm(songs, position=0, leave=True):
        #    tagged_corpus += song

        # Load the filtered song list if it exists, read over corpus and create it.
        if save_to_file:
            print("\nSaving to file...")
            #with open(tpath, 'w') as f:
            #    f.write(tagged_corpus)
            #with open(tpath, 'wb') as f_out:
                #pickle.dump(tagged_corpus, f_out)
    else:
        tagged_corpus = songs
    print("\nSuccessfully Loaded tagged corpus!\n")
    return tpath


class lyrics_generator(Dataset):

    def __init__(self, data, control_code="song", gpt2_type="gpt2", max_length=gpt2_medium_chunks, load_from_file=True, save_to_file=True):
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_type)
        self.songs = []

        if load_from_file:
            if not self.load_encoded_data():
                # Append byte-tokenized songs to list in tensor form
                for song in tqdm(data):
                    if len(song) < max_length:
                        self.songs.append(torch.tensor(
                            self.tokenizer.encode("<|{}|>{}<|endoftext|>".format(control_code, song))
                        ))

                # Gets the number of songs loaded in
                self.song_count = len(self.songs)

                # Saves the byte-encoded GPT2-tokenized tensors to a file
                if save_to_file:
                    f_data = {'songs': self.songs}
                    with open('../data/song_tensors.pickle', 'wb') as f_out:
                        pickle.dump(f_data, f_out)


    def load_encoded_data(self):
        fpath = "../data/song_tensors.pickle"

        # Load the filtered song list if it exists, read over corpus and create it.
        if path.exists(fpath):
            with open(fpath, 'rb') as f:
                self.songs = pickle.load(f)['songs']
                return True
        return False



    def __len__(self):
        return self.song_count

    def __getitem__(self, item):
        return self.songs[item]
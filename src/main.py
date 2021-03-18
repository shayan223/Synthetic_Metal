import filter
from generate import get_tagged_data
from generate import lyrics_generator
import random
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2Tokenizer
from sklearn.model_selection import train_test_split
from gpt2_client import GPT2Client
from tqdm import tqdm
import gpt_2_simple as gpt2
import torch
import nvgpu
import os
from os import path
import tensorflow as tf


def main():
    data = filter.load_data()

    # Get the length of each song in characters
    song_lengths = []
    for song in data:
        song_lengths.append(len(song))

    # Plot the song length Histogram on X axis
    plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})
    x = np.random.normal(size = 10)
    plt.hist(song_lengths, bins = 150, range=(0,3000))
    plt.gca().set(title='Song Length Distribution (in chars)', ylabel = 'Frequency')
    # plt.show()


    # model_size is a customizable parameter used to fine tune the model size using the model_sizes dictionary
    model_size = 'small'
    model_names = {'small': '124M', 'medium': '355M', 'large': '774M', 'xl': '1558M'}
    model_dims  = {'small': 768,    'medium': 1024,   'large': 1280,   'xl': 1600}
    model_name  = model_names[model_size]
    model_root = '../models/'
    model_path = model_root + model_name
    check_point_root = '../checkpoint/'
    check_point_path = check_point_root + model_name

    # Name of the fine-tuned model checkpoint (different names store different checkpoints)
    run_name = 'small_1'

    # tagged_data = lyrics_generator(data=data, control_code="<|lyric|>", gpt2_type="gpt2")
    tpath = get_tagged_data(data=data, model_size=model_dims[model_size])

    # If CUDA isn't available make sure you run the following installations IN CONDA SPECIFICALLY for your environment:
    # conda install -c anaconda cudatoolkit
    # conda install -c anaconda cudnn
    # conda install -c pytorch pytorch
    # conda install -c pytorch torchvision
    # conda install -c pytorch pytorch-nightly
    # conda install -c anaconda tensorflow-gpu
    print("IS CUDA GPU AVAILABLE: ", torch.cuda.is_available())

    # Downloads the model if it hasn't already been downloaded.
    if not path.isdir(model_path):
        gpt2.download_gpt2(model_name=model_name, model_dir=model_root)
    # Resets the session for retraining
    # tf.reset_default_graph()
    # Starts the tensorflow session
    sess = gpt2.start_tf_sess()

    # Customizable parameter of whether you want to load a previously-fine-tuned model checkpoint.
    load_previous_model = True
    fine_tune_model     = True
    generate_lyrics     = False
    restore_method      = ['fresh', 'latest']

    # Whether a previous model's checkpoints should be loaded using 'run_name'.
    if load_previous_model:
        if path.isdir(check_point_path):
            gpt2.load_gpt2(sess, run_name=run_name, model_dir=model_path, checkpoint_dir=check_point_path, model_name=model_name)

    # Whether the model should be fine-tuned.
    if fine_tune_model:
        gpt2.finetune(sess,
                      dataset=tpath,
                      model_name=model_name,
                      model_dir=model_root,
                      steps=1000,
                      restore_from=restore_method[1],
                      run_name=run_name,
                      print_every=10,
                      sample_every=100,
                      save_every=500
                      )
        print('Model has completed its fine-tuning!')

    # Whether the model should be used to generate lyrics.
    if generate_lyrics:
        # Generate lyrics using fine-tuned GPT2 model
        # If prefix parameter is specified we can provide the input string used to generate the lyrics
        default_prefix = "<|song|>"
        truncate = '<|endoftext|>'
        prefix = default_prefix
        text = gpt2.generate(sess, run_name=run_name, prefix=prefix, include_prefix=False, truncate=truncate, return_as_list=True, temperature=0.85, batch_size=1)[0]
        print("Generated Lyrics: \n", prefix)


if __name__ == "__main__":
    main()

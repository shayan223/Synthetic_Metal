import filter
import generate
from generate import lyrics_generator
import random
import matplotlib.pyplot as plt
import numpy as np
from transformers import GPT2Tokenizer
from model import train


def main():
    data = filter.load_data()

    # Selects a random song to test output. Can be pruned later.
    # rselect = random.randint(1, len(data)-1)

    # Get the length of each song in characters
    song_lengths = []
    for song in data:
        song_lengths.append(len(song))

    # Plot the song length Histogram on X axis
    plt.rcParams.update({'figure.figsize': (7, 5), 'figure.dpi': 100})
    x = np.random.normal(size = 10)
    plt.hist(song_lengths, bins = 150, range=(0,3000))
    plt.gca().set(title='Song Length Distribution (in chars)', ylabel = 'Frequency')
    plt.show()

    dataset = lyrics_generator(data=data, control_code="<|lyric|>", gpt2_type="gpt2")

    #model = train(dataset, GPT2LMHeadModel.from_pretrained(gpt2_type),
    #                GPT2Tokenizer.from_pretrained(gpt2_type),
    #                batch_size=16,
    #                epochs=1,
    #                lr=3e-5,
    #                max_seq_len=140,
    #                warmup_steps=5000,
    #                gpt2_type=gpt2_type,
    #                device="cuda",
    #                output_dir="trained_models",
    #                output_prefix="twitter",
    #                save_model_on_epoch=True
    #)

    from transformers import GPT2Tokenizer, TFGPT2Model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = TFGPT2Model.from_pretrained('gpt2')
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='tf')
    output = model(encoded_input)
    print("Generated Text: {}".format(output))

if __name__ == "__main__":
    main()

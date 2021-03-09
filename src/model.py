import torch as torch
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel
from tqdm import tqdm, trange
import os
from os import path


def get_model(gpt2_type="gpt2"):
    return GPT2LMHeadModel.from_pretrained(gpt2_type)


# Packs a tensor so that it bundles in a bunch of songs at once to train
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None


def train(dataset, model, tokenizer, batch_size=16, epochs=4, lr=2e-5, max_seq_len=400, warmup_steps=5000,
          gpt2_type="gpt2", gpt2_xl_chunks = 1600, device="cuda", output_dir=".", output_prefix="wreckgar",
          test_mode=False, save_model_on_epoch=False,):

    acc_steps = 100
    model = model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):
        print(f"Training epoch {epoch}")
        for idx, entry in tqdm(enumerate(train_dataloader)):
            (input_tensor, carry_on, remainder) = pack_tensor(entry, input_tensor, gpt2_xl_chunks)

            if carry_on and idx != len(train_dataloader) - 1:
                continue
            input_tensor = input_tensor.to(device)
            outputs = model(input_tensor, labels=input_tensor)
            loss = outputs[0]
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
    return model


def generate(model, tokenizer, prompt, entry_count=10, entry_length=100, top_p=0.8, temperature=1.,):
    model.eval()
    generated_num = 0
    generated_list = []
    filter_value = -float("Inf")

    with torch.no_grad():
        for entry_idx in trange(entry_count):
            entry_finished = False
            generated = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

            # Using top-p (nucleus sampling): https://github.com/huggingface/transformers/blob/master/examples/run_generation.py

            for i in range(entry_length):
                outputs = model(generated, labels=generated)
                loss, logits = outputs[:2]
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
                generated = torch.cat((generated, next_token), dim=1)

                if next_token in tokenizer.encode("<|endoftext|>"):
                    entry_finished = True

                if entry_finished:
                    generated_num = generated_num + 1
                    output_list = list(generated.squeeze().numpy())
                    output_text = tokenizer.decode(output_list)
                    generated_list.append(output_text)
                    break

            if not entry_finished:
                output_list = list(generated.squeeze().numpy())
                output_text = f"{tokenizer.decode(output_list)}<|endoftext|>"
                generated_list.append(output_text)

    return generated_list
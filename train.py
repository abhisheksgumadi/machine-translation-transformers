import numpy as np
import torch
import torch.nn as nn
import transformers
from pathlib import Path
from einops import rearrange
from tqdm import tqdm
from transformers import RobertaTokenizer
from torch.optim import Adam
from Optim import ScheduledOptim
from torch.utils.data import DataLoader
from dataset import ParallelLanguageDataset
from model import LanguageTransformer


num_epochs = 20
batch_size = 8
max_seq_length = 96
d_model = 512
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
nhead = 8
pos_dropout = 0.1
trans_dropout = 0.1
n_warmup_steps = 4000
PRE_TRAINED_MODEL_NAME = "distilroberta-base"


def main():
    project_path = str(Path(__file__).resolve().parents[0])
    tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
    train_dataset = ParallelLanguageDataset(
        project_path + "/data/raw/en/train.txt",
        project_path + "/data/raw/fr/train.txt",
        tokenizer,
        max_seq_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    valid_dataset = ParallelLanguageDataset(
        project_path + "/data/raw/en/val.txt",
        project_path + "/data/raw/fr/val.txt",
        tokenizer,
        max_seq_length,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    model = LanguageTransformer(
        tokenizer.vocab_size,
        d_model,
        nhead,
        num_encoder_layers,
        num_decoder_layers,
        dim_feedforward,
        max_seq_length,
        pos_dropout,
        trans_dropout,
    ).to("cpu")
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal_(p)

    optim = ScheduledOptim(
        Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09), d_model, n_warmup_steps
    )

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    train_losses, val_losses = train(
        train_loader, valid_loader, model, optim, criterion, num_epochs
    )


def train(train_loader, valid_loader, model, optim, criterion, num_epochs):
    print_every = 5
    model.train()

    lowest_val = 1e9
    train_losses = []
    val_losses = []
    total_step = 0
    print("-" * 100)
    print("Starting Training")
    print("-" * 100)
    for epoch in range(num_epochs):
        pbar = tqdm(total=print_every, leave=False)
        total_loss = 0

        for step, data_dict in enumerate(iter(train_loader)):
            total_step += 1
            src, tgt, src_key_padding_mask, tgt_key_padding_mask = (
                data_dict["ids1"],
                data_dict["ids2"],
                data_dict["masks_sent1"],
                data_dict["masks_sent2"],
            )
            src, src_key_padding_mask = src.to("cpu"), src_key_padding_mask.to("cpu")
            tgt, tgt_key_padding_mask = tgt.to("cpu"), tgt_key_padding_mask.to("cpu")

            memory_key_padding_mask = src_key_padding_mask.clone()
            tgt_inp, tgt_out = tgt[:, :-1], tgt[:, 1:]
            tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to("cpu")

            optim.zero_grad()
            outputs = model(
                src,
                tgt_inp,
                src_key_padding_mask,
                tgt_key_padding_mask[:, :-1],
                memory_key_padding_mask,
                tgt_mask,
            )
            loss = criterion(
                rearrange(outputs, "b t v -> (b t) v"),
                rearrange(tgt_out, "b o -> (b o)"),
            )

            loss.backward()
            optim.step_and_update_lr()

            total_loss += loss.item()
            train_losses.append((step, loss.item()))
            pbar.update(1)
            if step % print_every == print_every - 1:
                pbar.close()
                print(
                    f"Epoch [{epoch + 1} / {num_epochs}] \t Step [{step + 1} / {len(train_loader)}] \t "
                    f"Train Loss: {total_loss / print_every}"
                )
                total_loss = 0
                pbar = tqdm(total=print_every, leave=False)
        pbar.close()
        val_loss = validate(valid_loader, model, criterion)
        val_losses.append((total_step, val_loss))
        if val_loss < lowest_val:
            lowest_val = val_loss
            torch.save(model, "output/transformer.pth")
        print(f"Val Loss: {val_loss}")
    return train_losses, val_losses


def validate(valid_loader, model, criterion):
    pbar = tqdm(total=len(iter(valid_loader)), leave=False)
    model.eval()

    total_loss = 0
    for step, data_dict in enumerate(iter(valid_loader)):
        with torch.no_grad():
            src, tgt, src_key_padding_mask, tgt_key_padding_mask = (
                data_dict["ids1"],
                data_dict["ids2"],
                data_dict["masks_sent1"],
                data_dict["masks_sent2"],
            )
            src, src_key_padding_mask = src.to("cpu"), src_key_padding_mask.to("cpu")
            tgt, tgt_key_padding_mask = tgt.to("cpu"), tgt_key_padding_mask.to("cpu")
            memory_key_padding_mask = src_key_padding_mask.clone()
            tgt_inp = tgt[:, :-1]
            tgt_out = tgt[:, 1:].contiguous()
            tgt_mask = gen_nopeek_mask(tgt_inp.shape[1]).to("cpu")
            outputs = model(
                src,
                tgt_inp,
                src_key_padding_mask,
                tgt_key_padding_mask[:, :-1],
                memory_key_padding_mask,
                tgt_mask,
            )
            loss = criterion(
                rearrange(outputs, "b t v -> (b t) v"),
                rearrange(tgt_out, "b o -> (b o)"),
            )
            total_loss += loss.item()
            pbar.update(1)

    pbar.close()
    model.train()
    return total_loss / len(valid_loader)


def gen_nopeek_mask(length):
    mask = rearrange(torch.triu(torch.ones(length, length)) == 1, "h w -> w h")
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


if __name__ == "__main__":
    # device = torch.device('cpu:0' if torch.cpu.is_available() else 'cpu')
    main()

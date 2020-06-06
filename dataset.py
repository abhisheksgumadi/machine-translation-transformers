import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset


class ParallelLanguageDataset(Dataset):
    def __init__(self, data_path_1, data_path_2, tokenizer, max_len):
        self.data_1, self.data_2 = self.load_data(data_path_1, data_path_2)
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_1)

    def __getitem__(self, item_idx):
        sent1 = str(self.data_1[item_idx])
        sent2 = str(self.data_2[item_idx])
        encoded_output_sent1 = self.tokenizer.encode_plus(
            sent1,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )
        encoded_output_sent2 = self.tokenizer.encode_plus(
            sent2,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )

        encoded_output_sent1["attention_mask"][
            encoded_output_sent1["attention_mask"] == 1
        ] = 2
        encoded_output_sent1["attention_mask"][
            encoded_output_sent1["attention_mask"] == 0
        ] = True
        encoded_output_sent1["attention_mask"][
            encoded_output_sent1["attention_mask"] == 2
        ] = False
        encoded_output_sent1["attention_mask"] = encoded_output_sent1[
            "attention_mask"
        ].type(torch.bool)

        encoded_output_sent2["attention_mask"][
            encoded_output_sent2["attention_mask"] == 1
        ] = 2
        encoded_output_sent2["attention_mask"][
            encoded_output_sent2["attention_mask"] == 0
        ] = True
        encoded_output_sent2["attention_mask"][
            encoded_output_sent2["attention_mask"] == 2
        ] = False
        encoded_output_sent2["attention_mask"] = encoded_output_sent2[
            "attention_mask"
        ].type(torch.bool)

        return_dict = {
            "ids1": encoded_output_sent1["input_ids"].flatten(),
            "ids2": encoded_output_sent2["input_ids"].flatten(),
            "masks_sent1": encoded_output_sent1["attention_mask"].flatten(),
            "masks_sent2": encoded_output_sent2["attention_mask"].flatten(),
        }
        return return_dict

    def load_data(self, data_path_1, data_path_2):
        with open(data_path_1, "r") as f:
            data_1 = f.read().splitlines()[1:200]
        with open(data_path_2, "r") as f:
            data_2 = f.read().splitlines()[1:200]
        return data_1, data_2

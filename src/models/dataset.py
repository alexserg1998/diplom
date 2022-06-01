import pandas as pd
from transformers import BertTokenizer
from torch.utils.data import Dataset
import torch


class NegativeSamplingDataset(Dataset):
    """
    ToxicCommentsDataset is created to create a custom dataset.
    later we wrap a lightning data module around it.
    """

    def __init__(self, data: pd.DataFrame, tokenizer: BertTokenizer, max_token_len: int):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        sent1 = data_row['sentence_a']
        sent2 = data_row['sentence_b']
        label = data_row['label'].flatten()

        encoding1 = self.tokenizer.encode_plus(

            sent1,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',

        )
        encoding2 = self.tokenizer.encode_plus(

            sent2,
            add_special_tokens=True,
            max_length=self.max_token_len,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',

        )

        return dict(
            input_ids1=encoding1["input_ids"].flatten(),
            attention_mask1=encoding1["attention_mask"].flatten(),
            input_ids2=encoding2["input_ids"].flatten(),
            attention_mask2=encoding2["attention_mask"].flatten(),
            labels=torch.tensor(label, dtype=torch.long))
import pandas as pd
import pytorch_lightning as pl
import torch

from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast as BertTokenizer

from typing import Optional


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
        label = data_row['label']

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
            labels=torch.Tensor(label))


class NegativeSamplingDataModule(pl.LightningDataModule):
    def __init__(self, config, train_df: pd.DataFrame, test_df: pd.DataFrame, tokenizer, batch_size: int):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.max_token_len = config.max_token_count
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset, self.test_dataset = None, None

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = NegativeSamplingDataset(self.train_df, self.tokenizer, self.max_token_len)
        self.test_dataset = NegativeSamplingDataset(self.test_df, self.tokenizer, self.max_token_len)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=2
        )
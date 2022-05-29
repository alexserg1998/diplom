from typing import List
import pytorch_lightning as pl
from torch import nn
import torch
import torchmetrics

from transformers import BertModel, AdamW, get_linear_schedule_with_warmup


class SBERTModel(pl.LightningModule):
    """
    creates a pytorch lightning model
    """

    def __init__(self, config,
                 n_warmup_steps: int = None,
                 n_training_steps: int = None,
                 n_classes: int = 1):
        super().__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(config.language_model_path, return_dict=True)
        self.fc = nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()

        self.save_hyperparameters()

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2, labels=None):
        """
        :param input_ids1:
        :param attention_mask1:
        :param input_ids2:
        :param attention_mask2:
        :param labels:
        :return:
        """
        output1 = self.bert(input_ids1, attention_mask=attention_mask1)
        output2 = self.bert(input_ids2, attention_mask=attention_mask2)
        output1 = self.fc(output1.pooler_output)
        output2 = self.fc(output2.pooler_output)
        subtract = torch.sub(output1, output2)
        subtracted_dense = self.classifier(subtract)
        # output = torch.sigmoid(subtracted_dense)
        output = subtracted_dense
        loss = 0
        if labels is not None:
            print('label_shape:', labels.shape)
            print('output_shape:', output.shape)
            print('label', labels)
            print('output', output)
            loss = self.criterion(output, labels)
        return loss, output

    def training_step(self, batch, batch_idx):
        """
        :param batch:
        :param batch_idx:
        :return:
        """
        input_ids1 = batch["input_ids1"]
        attention_mask1 = batch["attention_mask1"]
        input_ids2 = batch["input_ids2"]
        attention_mask2 = batch["attention_mask2"]
        labels = batch["labels"]
        loss, output = self(input_ids1, attention_mask1, input_ids2, attention_mask2, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return {"loss": loss, "predictions": output, "labels": labels}

    def validation_step(self, batch, batch_idx):
        """
        :param batch:
        :param batch_idx:
        :return:
        """
        input_ids1 = batch["input_ids1"]
        attention_mask1 = batch["attention_mask1"]
        input_ids2 = batch["input_ids2"]
        attention_mask2 = batch["attention_mask2"]
        labels = batch["labels"]
        loss, _ = self(input_ids1, attention_mask1, input_ids2, attention_mask2, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False)
        return loss

    def test_step(self, batch, batch_idx):
        """
        :param batch:
        :param batch_idx:
        :return:
        """
        input_ids1 = batch["input_ids1"]
        attention_mask1 = batch["attention_mask1"]
        input_ids2 = batch["input_ids2"]
        attention_mask2 = batch["attention_mask2"]
        labels = batch["labels"]
        loss, outputs = self(input_ids1, attention_mask1, input_ids2, attention_mask2, labels)
        binary_outputs = torch.as_tensor((outputs - 0.5) > 0,
                                         dtype=torch.int)
        self.log("test accuracy", self.accuracy(binary_outputs,
                                                torch.as_tensor(labels, dtype=torch.int)),
                 prog_bar=True, logger=True)
        self.log("test_loss", loss, prog_bar=True,
                 logger=True)
        return loss

    def configure_optimizers(self):
        """
        :return:
        """
        optimizer = AdamW(self.parameters(), lr=self.config.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.n_warmup_steps,
                                                    num_training_steps=self.n_training_steps)
        return dict(optimizer=optimizer, lr_scheduler=dict(scheduler=scheduler, interval="step"))
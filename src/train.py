import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from utils import calculate_warmup_steps
from configuration.config import BaseConfig

from models import NegativeSamplingDataModule, SBERTModel, build_checkpoint_callback

from sklearn.model_selection import train_test_split

from transformers import BertTokenizer
import pandas as pd
if __name__ == "__main__":
    torch.cuda.empty_cache()
    CONFIG_CLASS = BaseConfig()
    CONFIG = CONFIG_CLASS.get_config()
    random_seed = CONFIG.random_seed
    pl.seed_everything(random_seed)

    tokenizer = BertTokenizer.from_pretrained(CONFIG.language_model_tokenizer_path)
    print("tokenizer loaded .... ")
    print(CONFIG.train_data)
    data = pd.read_csv(CONFIG.train_data)
    train, val = train_test_split(data, test_size=0.05)

    data_module = NegativeSamplingDataModule(CONFIG, train, val, tokenizer,
                                             batch_size=CONFIG.batch_size,
                                             )

    total_training_steps, warmup_steps = calculate_warmup_steps(train, CONFIG.n_epochs,
                                                                CONFIG.batch_size)

    print(f"total_training_steps : {total_training_steps} , warmup_steps : {warmup_steps} ")
    model = SBERTModel(config=CONFIG, n_classes=1,
                       n_warmup_steps=warmup_steps,
                       n_training_steps=total_training_steps,
                       )

    checkpoint_callback = build_checkpoint_callback(CONFIG.save_top_k, CONFIG.saved_model_path)
    logger = TensorBoardLogger("lightning_logs", name="toxic-comments")

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=2)
    trainer = pl.Trainer(logger=logger,
                         callbacks=[checkpoint_callback, early_stopping_callback],
                         max_epochs=CONFIG.n_epochs, progress_bar_refresh_rate=30, gpus=1)
    trainer.fit(model, datamodule=data_module)
    trainer.test()
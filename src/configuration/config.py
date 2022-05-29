import argparse
from pathlib import Path


class BaseConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--model_name", type=str, default="siamese-bert")

        self.parser.add_argument("--language_model_path", type=str,
                                 default=Path(__file__).parents[1].__str__() + "/saved_model")
        self.parser.add_argument("--language_model_tokenizer_path", type=str,
                                 default=Path(__file__).parents[1].__str__() + "/saved_model")

        self.parser.add_argument("--save_top_k", type=int, default=1, help="...")

        self.parser.add_argument("--saved_model_path", type=str,
                                 default=Path(__file__).parents[1].__str__() + "/assets/saved_models/")

        self.parser.add_argument("--csv_logger_path", type=str,
                                 default=Path(__file__).parents[1].__str__() + "/assets")

        self.parser.add_argument("--train_data", type=str,
                                 default=Path(__file__).parents[1].__str__() + "/data/df_with_label.csv")

        self.parser.add_argument("--n_epochs", type=int,
                                 default=12,
                                 help="...")

        self.parser.add_argument("--batch_size", type=int,
                                 default=12,
                                 help="...")
        self.parser.add_argument("--max_token_count", type=int, default=512)

        self.parser.add_argument("--lr", default=7e-5,
                                 help="...")
        self.parser.add_argument("--random_seed", type=int, default=42)

    def get_config(self):
        return self.parser.parse_args()

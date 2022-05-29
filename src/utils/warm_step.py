import pandas as pd


def calculate_warmup_steps(train_df: pd.DataFrame, num_epochs: int, batch_size: int):
    steps_per_epoch = len(train_df) // batch_size
    total_training_steps = steps_per_epoch * num_epochs
    warmup_steps = total_training_steps // 5
    return total_training_steps, warmup_steps

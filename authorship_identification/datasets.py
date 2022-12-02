import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from typing import Tuple


def get_C50(
    orig_split, split, remove_entities, seed=42
) -> Tuple[pd.DataFrame, pd.DateOffset]:
    train_df = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            f'data/train{"_noents" if remove_entities else ""}.csv',
        )
    )
    test_df = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            f'data/test{"_noents" if remove_entities else ""}.csv',
        )
    )
    if not orig_split:
        train_df = pd.concat([train_df, test_df])
        test_df = train_df.sample(frac=1 - split, random_state=seed)
        train_df = train_df.drop(test_df.index)
    return train_df, test_df


def get_AllTheNews(
    orig_split, split, remove_entities, seed=42
) -> Tuple[pd.DataFrame, pd.DateOffset]:
    le = LabelEncoder()
    train_df = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__),
            f'data/all_the_news{"_noents" if remove_entities else ""}.csv',
        )
    )
    train_df = train_df[["content", "author"]]
    train_df["author"] = le.fit_transform(train_df["author"])
    train_df = train_df.rename(columns={"content": "text", "author": "label"})
    test_df = train_df.sample(frac=0.15 if orig_split else split, random_state=seed)
    train_df = train_df.drop(test_df.index).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, test_df

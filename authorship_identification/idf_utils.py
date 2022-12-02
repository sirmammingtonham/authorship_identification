import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Tuple

tfidf_default_args = {
    "ngram_range": (1, 2),
    "lowercase": False,
}  #'stop_words':'english'}


def idf_corpus(df, tfidf_args) -> pd.DataFrame:
    """
    Creates a dataframe of each word in the entire corpus and its corresponding idf value
    """
    vectorizer = TfidfVectorizer(**tfidf_args)
    vectorizer.fit(df["text"])
    output = pd.DataFrame(
        {"word": vectorizer.get_feature_names_out(), "idf": vectorizer.idf_}
    )
    return output.sort_values("idf", ascending=False)


def idf_author(
    train_df, test_df, author, tfidf_args, idf_threshold=None
) -> pd.DataFrame:
    """
    Creates a dataframe of each word in an author's corpus and its corresponding idf value
    """
    df = pd.concat([train_df, test_df])
    df = df[df["label"] == author]
    df = idf_corpus(df, tfidf_args)
    if idf_threshold:
        df = df[df["idf"] < idf_threshold]
    return df


def percentile_to_idf_threshold(train_idfdf, percentile) -> np.float32:
    """
    Calculates the idf threshold required to remove {percentile} feature types
    """
    return np.percentile(train_idfdf["idf"].to_numpy(), 100 - percentile)


def calc_feat_types_removed(train_idfdf, idf_threshold) -> np.float32:
    """
    Calculates the number of feature types removed at a certain idf threshold
    """
    return np.mean(train_idfdf["idf"].to_numpy() >= idf_threshold)


def calc_tokens_removed(
    train_df: pd.DataFrame, idf_df: pd.DataFrame, idf_threshold, tfidf_args
) -> np.float32:
    """
    Calculates the number of tokens removed per document on average at a certain idf threshold
    """
    full = TfidfVectorizer(**tfidf_args).build_analyzer()
    remove_words = idf_df[idf_df["idf"] >= idf_threshold]["word"].tolist()
    remove = TfidfVectorizer(
        **{**tfidf_args, "stop_words": remove_words}
    ).build_analyzer()
    full_len = train_df["text"].apply(lambda text: len(full(text))).to_numpy()
    remove_len = train_df["text"].apply(lambda text: len(remove(text))).to_numpy()
    return np.mean((full_len - remove_len) / full_len)


def df_to_tfidf(
    train_df, test_df, train_idfdf, tfidf_args, idf_threshold=None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorizes train and test data to tfidf, removes words above idf_threshold threshold if provided
    """
    if idf_threshold:
        tfidf_args["stop_words"] = train_idfdf[train_idfdf["idf"] >= idf_threshold][
            "word"
        ].tolist()
    vectorizer = TfidfVectorizer(**tfidf_args)
    X_train = vectorizer.fit_transform(train_df["text"])
    y_train = train_df["label"]
    X_test = vectorizer.transform(test_df["text"])
    y_test = test_df["label"]
    words = vectorizer.get_feature_names_out()

    return words, X_train, y_train, X_test, y_test

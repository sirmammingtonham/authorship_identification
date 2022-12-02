import sys
sys.path.append('./')
sys.path.append('../')

import streamlit as st

st.set_page_config(layout="wide")

import pandas as pd
import numpy as np
import plotly.express as px

from authorship_identification import datasets
from authorship_identification import idf_utils
from authorship_identification import linear_models
from text_annotator import text_annotator

# from dict_input import dict_input


dataset = st.radio("Select Dataset", ("C50", "All The News"))

orig_split = st.checkbox("Use original split?", value=True)
split = st.slider(
    "Custom train/test split fraction",
    min_value=0.0,
    max_value=1.0,
    step=0.1,
    value=0.5,
    disabled=orig_split,
)
remove_entities = st.checkbox("Remove entities?", value=True)

seed = 42


@st.cache
def split_df(dataset, orig_split, split, remove_entities):
    if dataset == "C50":
        return datasets.get_C50(orig_split, split, remove_entities, seed=seed)
    else:
        return datasets.get_AllTheNews(orig_split, split, remove_entities, seed=seed)


train_df, test_df = split_df(dataset, orig_split, split, remove_entities)

annotated_df = pd.read_csv("./authorship_identification/data/annotated_train.csv")
train_df_full = pd.read_csv(f"./authorship_identification/data/train.csv")

st.header("Annotate an example")
anno_idx = st.slider("Index to annotate", min_value=0, max_value=len(train_df_full) - 1)


def annotate(anno_idx):
    with st.form("annotate"):
        st.caption("Select all topic/non-style words")
        words = train_df_full["text"].iloc[anno_idx].split()
        prev_annotation = annotated_df.loc[anno_idx, "word_indexes"]
        selection = text_annotator(words, selected=prev_annotation)
        st.form_submit_button()

        if selection is not None:
            annotated_df.loc[anno_idx, "words"] = np.array(words)[selection].tolist()
            annotated_df.loc[anno_idx, "word_indexes"] = selection
            annotated_df.to_csv("./data/annotated_train.csv", index=None)


annotate(anno_idx)


tfidf_args = idf_utils.tfidf_default_args


@st.cache
def idf_corpus(df, tfidf_args):
    return idf_utils.idf_corpus(df, tfidf_args)


def idf_author(author, tfidf_args, idf_threshold=None):
    df = idf_utils.idf_author(train_df, test_df, author, tfidf_args, idf_threshold)
    st.dataframe(df)


dfcol1, dfcol2 = st.columns(2)
with dfcol1:
    st.header("Train set")
    st.dataframe(train_df)
    st.caption(f"Train IDF")
    train_idfdf = idf_corpus(train_df, tfidf_args)
    st.dataframe(train_idfdf)

with dfcol2:
    st.header("Test set")
    st.dataframe(test_df)
    st.caption(f"Test IDF")
    test_idfdf = idf_corpus(test_df, tfidf_args)
    st.dataframe(test_idfdf)


@st.cache
def calc_tokens_removed(train_df: pd.DataFrame, idf_threshold, tfidf_args):
    return idf_utils.calc_tokens_removed(
        train_df, train_idfdf, idf_threshold, tfidf_args
    )


cutoff_type = st.radio("IDF Cutoff type", ("Value", "Percentile"))
idf_threshold = st.number_input(
    "IDF cutoff value (remove if IDF >= value)",
    value=train_idfdf["idf"].max() + 0.01,
    disabled=cutoff_type != "Value",
    format="%.5f",
)
percentile = st.number_input(
    "Percent of IDF to remove",
    value=10.0,
    disabled=cutoff_type != "Percentile",
    format="%.5f",
)
if cutoff_type == "Value":
    st.write(
        f"Percentage feature types removed: {idf_utils.calc_feat_types_removed(train_idfdf, idf_threshold)}"
    )
if cutoff_type == "Percentile":
    idf_threshold = idf_utils.percentile_to_idf_threshold(train_idfdf, percentile)
    st.write(f"IDF Cutoff: {idf_threshold}")

st.write(
    f"Average percentage tokens removed: {calc_tokens_removed(train_df, idf_threshold, tfidf_args)}"
)


@st.cache
def train(train_df, test_df, train_idfdf, tfidf_args):
    words, X_train, y_train, X_test, y_test = idf_utils.df_to_tfidf(
        train_df, test_df, train_idfdf, tfidf_args, idf_threshold=idf_threshold
    )

    acc_nb, f1_nb, feat_nb = linear_models.train_eval_nb(
        X_train, y_train, X_test, y_test
    )
    acc_sgd, f1_sgd, feat_sgd = linear_models.train_eval_sgd(
        X_train, y_train, X_test, y_test
    )

    return words, feat_nb, feat_sgd, (acc_nb, f1_nb), (acc_sgd, f1_sgd)


st.header("Training")
words, feat_nb, feat_sgd, (acc_nb, f1_nb), (acc_sgd, f1_sgd) = train(
    train_df, test_df, train_idfdf, tfidf_args
)
st.success(f"Multinomial Naive Bayes Accuracy: {acc_nb}", icon="📈")
st.success(f"Multinomial Naive Bayes F1: {f1_nb}", icon="🎯")
st.success(f"Logistic Regression Accuracy: {acc_sgd}", icon="📈")
st.success(f"Logistic Regression F1: {f1_sgd}", icon="🎯")

st.header("Plot feature importance")


def impPlot(feat_nb, author, title, tfidf_df, top_k=10):
    features = feat_nb[author]
    top_k = np.argsort(-features)[:top_k][::-1]
    idfs = []
    for word in words[top_k]:
        idfs.append(tfidf_df[tfidf_df["word"] == word]["idf"])
    figure = px.bar(
        x=features[top_k],
        y=words[top_k],
        labels={"x": "Importance Value", "y": "Words", "index": "Columns"},
        hover_data={"idf": idfs},
        text=features[top_k],
        title=title + str(author),
        width=1000,
        height=600,
    )
    figure.update_layout(
        {
            "plot_bgcolor": "rgba(0, 0, 0, 0)",
            "paper_bgcolor": "rgba(0, 0, 0, 0)",
        }
    )
    st.plotly_chart(figure, use_container_width=True)


author = st.slider("Author", min_value=0, max_value=49, step=1)
top_k = st.slider("Top K", min_value=1, max_value=100, step=5, value=10)

imcol1, imcol2 = st.columns(2)
with imcol1:
    st.caption(f"Full IDF for Author #{author}")
    idf_author(author, tfidf_args)
    impPlot(
        feat_nb, author, "NB Feature Importance, Author #", train_idfdf, top_k=top_k
    )
with imcol2:
    st.caption(f"IDF for Author #{author} (after removal)")
    idf_author(author, tfidf_args, idf_threshold)
    impPlot(
        feat_sgd,
        author,
        "Logistic Regression Feature Importance, Author #",
        train_idfdf,
        top_k=top_k,
    )

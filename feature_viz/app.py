import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

orig_split = st.checkbox('Use original split?', value=True)
split = st.slider('Custom train/test split fraction', min_value=0., max_value=1., step=0.1, value=0.5, disabled=orig_split)
seed = 42

@st.cache
def split_df(orig_split, split):
    train_df = pd.read_csv('../data/train.csv')
    test_df = pd.read_csv('../data/test.csv')
    if not orig_split:
        train_df = pd.concat([train_df, test_df])
        test_df = train_df.sample(frac=1-split, random_state=seed)
        train_df = train_df.drop(test_df.index)
    return train_df, test_df

train_df, test_df = split_df(orig_split, split)

st.header('Train set')
st.dataframe(train_df)
st.header('Test set')
st.dataframe(test_df)

@st.cache
def train(train_df, test_df):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, stop_words='english')
    X_train = vectorizer.fit_transform(train_df['text'])
    y_train = train_df['label']
    X_test = vectorizer.transform(test_df['text'])
    y_test = test_df['label']
    words = vectorizer.get_feature_names_out()

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_nb = clf.predict(X_test)
    feat_nb = np.exp(clf.feature_log_prob_)
    acc_nb, f1_nb = accuracy_score(y_test, y_nb), f1_score(y_test, y_nb, average='micro')

    clf = SGDClassifier(loss='log', penalty='l2', alpha=0.000001, n_iter_no_change=5, shuffle=True, random_state=seed)
    clf.fit(X_train, y_train)
    y_sgd = clf.predict(X_test)
    feat_sgd = clf.coef_
    acc_sgd, f1_sgd = accuracy_score(y_test, y_sgd), f1_score(y_test, y_sgd, average='micro')

    return words, feat_nb, feat_sgd, (acc_nb, f1_nb), (acc_sgd, f1_sgd)

st.header('Training')
words, feat_nb, feat_sgd, (acc_nb, f1_nb), (acc_sgd, f1_sgd) = train(train_df, test_df)
st.success(f'Multinomial Naive Bayes Accuracy: {acc_nb}', icon="📈")
st.success(f'Multinomial Naive Bayes F1: {f1_nb}', icon="🎯")
st.success(f'SGD Accuracy: {acc_sgd}', icon="📈")
st.success(f'SGD F1: {f1_sgd}', icon="🎯")

st.header('Plot feature importance')
def impPlot(feat_nb, author, title, top_k=10):
    features = feat_nb[author]
    top_k = np.argsort(-features)[:top_k][::-1]
    # top_k = np.argpartition(features, -top_k)[-top_k:]
    figure = px.bar(x=features[top_k],
                    y=words[top_k], labels = {'x':'Importance Value', 'y':'Words', 'index':'Columns'},
                    text=features[top_k],
                    title=title + str(author),
                    width=1000, height=600)
    figure.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    st.plotly_chart(figure, use_container_width=True)

author = st.slider('Author', min_value=0, max_value=49, step=1)
top_k = st.slider('Top K', min_value=1, max_value=100, step=5, value=10)
impPlot(feat_nb, author, 'NB Feature Importance, Author #', top_k=top_k)
impPlot(feat_sgd, author, 'SGD Feature Importance, Author #', top_k=top_k)
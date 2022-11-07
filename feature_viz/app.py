import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
# import spacy
# import nltk as nltk

from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

from text_annotator import text_annotator
from dict_input import dict_input

try:
    st.set_page_config(layout="wide")
except:
    pass


# nlp = spacy.load('en_core_web_sm', disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])



orig_split = st.checkbox('Use original split?', value=True)
split = st.slider('Custom train/test split fraction', min_value=0., max_value=1., step=0.1, value=0.5, disabled=orig_split)
remove_entities = st.checkbox('Remove entities?', value=True)

seed = 42

@st.cache
def split_df(orig_split, split, remove_entities):
    train_df = pd.read_csv(f'./data/train{"_noents" if remove_entities else ""}.csv')
    test_df = pd.read_csv(f'./data/test{"_noents" if remove_entities else ""}.csv')
    if not orig_split:
        train_df = pd.concat([train_df, test_df])
        test_df = train_df.sample(frac=1-split, random_state=seed)
        train_df = train_df.drop(test_df.index)
    return train_df, test_df

train_df, test_df = split_df(orig_split, split, remove_entities)

# pd.DataFrame(list(zip([[]]*2500, [[]]*2500)), columns=['words', 'word_indexes']).to_csv('./data/annotated_train.csv', index=None)
annotated_df = pd.read_csv('./data/annotated_train.csv')
train_df_full = pd.read_csv(f'./data/train.csv')

st.header('Annotate an example')
anno_idx = st.slider('Index to annotate', min_value=0, max_value=len(train_df_full)-1)
def annotate(anno_idx):
    with st.form('annotate'):
        st.caption('Select all non-style words')
        words = train_df_full['text'].iloc[anno_idx].split()
        prev_annotation = annotated_df.loc[anno_idx, 'word_indexes']
        selection = text_annotator(words, selected=prev_annotation)
        st.form_submit_button()

        if selection is not None:
            annotated_df.loc[anno_idx, 'words'] = np.array(words)[selection].tolist()
            annotated_df.loc[anno_idx, 'word_indexes'] = selection
            annotated_df.to_csv('./data/annotated_train.csv', index=None)

annotate(anno_idx)


tfidf_args = {'ngram_range':(1, 2), 'lowercase':False,} #'stop_words':'english'}
# # tfidf_args = dict_input("TFIDF vectorizer args", tfidf_args_template)

# @st.cache
# def tfidf_corpus(df, tfidf_args):
#     bag_of_words = CountVectorizer(**tfidf_args)
#     counts = bag_of_words.fit_transform(df['text'])
#     tfidf_transformer = TfidfTransformer(use_idf=True, sublinear_tf=True)
#     tfidf = tfidf_transformer.fit_transform(counts)
#     tf = counts.toarray().mean(axis=0)
#     idf = tfidf_transformer.idf_
#     tfidf = tfidf.toarray().mean(axis=0)
#     output = pd.DataFrame({'word': bag_of_words.get_feature_names_out(), 'tf': tf, 'idf': idf, 'tf-idf': tfidf})
#     return output.sort_values('tf-idf', ascending=False)

@st.cache
def idf_corpus(df, tfidf_args):
    vectorizer = TfidfVectorizer(**tfidf_args)
    vectorizer.fit(df['text'])
    output = pd.DataFrame({'word': vectorizer.get_feature_names_out(),'idf': vectorizer.idf_})
    return output.sort_values('idf', ascending=False)


# def tfidf_author(author, tfidf_args):
#     df = pd.concat([train_df, test_df])
#     df = df[df['label'] == author]
#     st.dataframe(tfidf_corpus(df, tfidf_args))

def idf_author(author, tfidf_args):
    df = pd.concat([train_df, test_df])
    df = df[df['label'] == author]
    st.dataframe(idf_corpus(df, tfidf_args))

dfcol1, dfcol2 = st.columns(2)
with dfcol1:
    st.header('Train set')
    st.dataframe(train_df)
    st.caption(f'IDF')
    train_idfdf = idf_corpus(train_df, tfidf_args)
    st.dataframe(train_idfdf)
    # st.caption(f'TF-IDF')
    # train_tfidf = tfidf_corpus(train_df, tfidf_args)
    # st.dataframe(train_tfidf)

with dfcol2:
    st.header('Test set')
    st.dataframe(test_df)
    st.caption(f'IDF')
    test_idfdf = idf_corpus(test_df, tfidf_args)
    st.dataframe(test_idfdf)
    # st.caption(f'TF-IDF')
    # test_tfidf = tfidf_corpus(test_df, tfidf_args)
    # st.dataframe(test_tfidf)

idf_split = st.number_input('IDF cutoff value (remove if IDF >= value)', value=-1.)

# vectorizer = TfidfVectorizer(**tfidf_args)
# X_train = vectorizer.fit_transform(train_df['text'])
# y_train = train_df['label']
# X_test = vectorizer.transform(test_df['text'])
# y_test = test_df['label']
# words = vectorizer.get_feature_names_out()

@st.cache
def train(train_df, test_df, train_idfdf, tfidf_args):
    if idf_split > 0.0:
        tfidf_args['stop_words'] = train_idfdf[train_idfdf['idf'] >= idf_split]['word'].tolist()
    vectorizer = TfidfVectorizer(**tfidf_args)
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

    clf = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.000001, n_iter_no_change=5, shuffle=True, random_state=seed)
    clf.fit(X_train, y_train)
    y_sgd = clf.predict(X_test)
    feat_sgd = clf.coef_
    acc_sgd, f1_sgd = accuracy_score(y_test, y_sgd), f1_score(y_test, y_sgd, average='micro')

    return words, feat_nb, feat_sgd, (acc_nb, f1_nb), (acc_sgd, f1_sgd)

st.header('Training')
words, feat_nb, feat_sgd, (acc_nb, f1_nb), (acc_sgd, f1_sgd) = train(train_df, test_df, train_idfdf, tfidf_args)
st.success(f'Multinomial Naive Bayes Accuracy: {acc_nb}', icon="📈")
st.success(f'Multinomial Naive Bayes F1: {f1_nb}', icon="🎯")
st.success(f'SGD Accuracy: {acc_sgd}', icon="📈")
st.success(f'SGD F1: {f1_sgd}', icon="🎯")

st.header('Plot feature importance')
def impPlot(feat_nb, author, title, tfidf_df, top_k=10):
    features = feat_nb[author]
    top_k = np.argsort(-features)[:top_k][::-1]
    idfs = []
    for word in words[top_k]:
        idfs.append(tfidf_df[tfidf_df['word']==word]['idf'])
    # top_k = np.argpartition(features, -top_k)[-top_k:]
    figure = px.bar(x=features[top_k],
                    y=words[top_k], labels = {'x':'Importance Value', 'y':'Words', 'index':'Columns'},
                    hover_data={'idf': idfs},
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

st.caption(f'TF-IDF for Author #{author}')
idf_author(author, tfidf_args)

imcol1, imcol2 = st.columns(2)
with imcol1:
    impPlot(feat_nb, author, 'NB Feature Importance, Author #', train_idfdf, top_k=top_k)
with imcol2:
    impPlot(feat_sgd, author, 'SGD Feature Importance, Author #', train_idfdf, top_k=top_k)
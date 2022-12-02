import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score


def train_eval_nb(X_train, y_train, X_test, y_test):
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_nb = clf.predict(X_test)
    feat_nb = np.exp(clf.feature_log_prob_)
    acc_nb, f1_nb = accuracy_score(y_test, y_nb), f1_score(
        y_test, y_nb, average="micro"
    )

    return acc_nb, f1_nb, feat_nb


def train_eval_sgd(X_train, y_train, X_test, y_test, seed=42):
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.000001,
        n_iter_no_change=5,
        shuffle=True,
        random_state=seed,
    )
    clf.fit(X_train, y_train)
    y_sgd = clf.predict(X_test)
    feat_sgd = clf.coef_
    acc_sgd, f1_sgd = accuracy_score(y_test, y_sgd), f1_score(
        y_test, y_sgd, average="micro"
    )

    return acc_sgd, f1_sgd, feat_sgd

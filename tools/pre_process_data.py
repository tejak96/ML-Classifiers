# coding=utf-8
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from read_data import get_data
import re

def pre_proc():
    features_train, features_test, labels_train, labels_test = get_data()
    my_stop_words = text.ENGLISH_STOP_WORDS.union(["äôs", "äô"])

    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, preprocessor=lambda x: re.sub(r'(\d)+', 'NUM', x.lower()), stop_words=my_stop_words)
    features_train_transformed = vectorizer.fit_transform(features_train)
    features_test_transformed  = vectorizer.transform(features_test)

    selector = SelectPercentile(f_classif, percentile=10)
    selector.fit(features_train_transformed, labels_train)
    features_train_transformed = selector.transform(features_train_transformed).toarray()
    features_test_transformed  = selector.transform(features_test_transformed).toarray()

    return features_train_transformed, features_test_transformed, labels_train, labels_test
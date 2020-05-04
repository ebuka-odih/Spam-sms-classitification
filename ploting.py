import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder


def plot_learning_curve(name, estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    #if ylim is not None:
    #    plt.ylim(*ylim)
    plt.xlabel("Samples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Testing")

    plt.legend(loc="best")
    plt.savefig(name)
    return plt


data = pd.read_csv("disambiguate_spam.csv", encoding='latin-1')

data['length'] = data['message'].apply(len)
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})
x = data.message
y = data.label_num
vect = CountVectorizer()
le = LabelEncoder()
tags = le.fit_transform(y)
tok = keras.preprocessing.text.Tokenizer(num_words=500)
tok.fit_on_texts(x)
X = tok.texts_to_matrix(x, mode='count')

# converting features into numeric vector
#X = np.array(vect.fit_transform(x))
#print(X)
#print(type(X))

if __name__ == '__main__':
    #digits = load_digits()
    #X, y = digits.data, digits.target
    #print(X)
    #print(type(X))
    #exit(0)
    #title = "Learning Curves (Adaboost)"
    title = "Learning Curves (Logistic Regression)"
    # Cross validation with 100 iterations to get smoother mean test and train
    # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    #print(cv)

    #estimator = GaussianNB()
    estimator = LogisticRegression()
    #estimator = AdaBoostClassifier(n_estimators=23, random_state=11)
    plot_learning_curve("logistic.pdf", estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    # title = r"Learning Curves (Naive Bayes))"
    # title = r"Learning Curves (SVM, RBF Kernel)"
    title = r"Learning Curves (Random Forest)"
    # SVC is more expensive so we do a lower number of CV iterations:
    #cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
    estimator = RandomForestClassifier()
    #estimator = SVC(gamma=0.001)
    #estimator = MultinomialNB(alpha=0.2)
    plot_learning_curve("random.pdf", estimator, title, X, y, (0.7, 1.01), cv=cv, n_jobs=4)

    plt.show()

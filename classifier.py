import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import seaborn as sns

#from sklearn.feature_selection import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif

from sklearn.feature_selection import SelectKBest, chi2, f_regression, VarianceThreshold, SelectPercentile
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score



data = pd.read_csv("disambiguate_spam.csv", encoding='latin-1')
data.head()

data['length'] = data['message'].apply(len)
data['label_num'] = data.label.map({'ham': 0, 'spam': 1})
print(data.head())
x = data.message
y = data.label_num
vect = CountVectorizer()
# converting features into numeric vector
x = vect.fit_transform(x)
print(x)


# Feature selections
'''
Feature selection for SelectKBest
'''
selector_kbest = SelectKBest(score_func=chi2, k=20).fit(x, y)
x = selector_kbest.fit_transform(x, y)
#print(selector_kbest.pvalues_)

#print("shape:", selector_kbest.shape)
#print("pvalues_:", selector_kbest.pvalues_)
#print("scores_:", selector_kbest.scores_)


'''
This is the feature selection of SelectPercentile
'''

#X_new = SelectPercentile(chi2, percentile=10).fit(x, y)
#
#print("Pvalues_", X_new.pvalues_)
#print("Scores_", X_new.scores_)
#x = X_new.fit_transform(x, y)


'''
Feature Selection for VarianceThreshold
'''
#
#thresholder = VarianceThreshold(threshold=.5).fit(x, y)
#
## Conduct variance thresholding
#x = thresholder.fit_transform(x, y)
#
## View first five rows with features with variances above threshold
#print(x[0:5])

'''
Feature Selection using Mutual_info
'''

#sel_mutual = SelectKBest(mutual_info_classif, k=4).fit(x, y)
#x = sel_mutual.fit_transform(x, y)
##print(x)



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=11)




# Loading all classifier
svc = SVC(kernel='linear')
mnb = MultinomialNB(alpha=0.2)
gnb = GaussianNB()
lr = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=23, random_state=11)
abc = AdaBoostClassifier(n_estimators=23, random_state=11)
X_train = np.array(X_train.toarray(), dtype=np.float)


# defining functions for training and testing data
def training(clf, x_train, Y_train):
    clf.fit(x_train, Y_train)


# function for predicting labels
def predict(clf, X_test):
    return clf.predict(X_test)


classifier = {'SVM': svc, 'MultinomialNB': mnb, 'GaussianNB': gnb, 'logistic': lr, 'RandomForest': rfc, 'Adaboost': abc}

score = []

for n, c in classifier.items():
    training(c, X_train, y_train)
    pred = predict(c, X_test.toarray())
    score.append((n, [accuracy_score(y_test, pred, normalize=True)]))
    
    

score_df = pd.DataFrame.from_items(score, orient='index', columns=['Classification Score'])

# Adding accuracy column
score_df['Accuracy (%)'] = score_df['Classification Score'] * 100
print(score_df)


# precision score
pre_score = []

for n, c in classifier.items():
     pred = predict(c, X_test.toarray())
     pre_score.append((n, [precision_score(y_test, pred, average='weighted')]))
    
pre_score_df = pd.DataFrame.from_items(pre_score, orient='index', columns=['Precision Score'])    

pre_score_df['Accuracy (%)'] = pre_score_df['Precision Score'] * 100
print(pre_score_df)

#for plotting
plt.plot(pre_score_df, pre_score_df, label="Precision Score")
plt.ylabel("Score")
plt.legend()
plt.show()


# Recall Score
recal_score = []

for n, c in classifier.items():
     pred = predict(c, X_test.toarray())
     recal_score.append((n, [recall_score(y_test, pred, average='micro')]))
     
    
recal_score_df = pd.DataFrame.from_items(recal_score, orient='index', columns=['Recall Score'])    

recal_score_df['Accuracy (%)'] = recal_score_df['Recall Score'] * 100
print(recal_score_df)

# recall score plotting
plt.plot(recal_score_df, recal_score_df, label="Recall Score")
plt.legend()
plt.show()


##getting the precison data
#precision = precision_score(y_test, pred, average='weighted')
#print("Precision Score", precision)
#
#recall_score = recall_score(y_test, pred, average='weighted')
#print("Recall Score", recall_score)









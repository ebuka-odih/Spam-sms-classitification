#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 12:46:54 2020

@author: emma
"""

'''
Feature Selection for VarianceThreshold
'''
from sklearn.feature_selection import SelectKBest, chi2, f_regression, VarianceThreshold, SelectPercentile

thresholder = VarianceThreshold(threshold=.5).fit(x, y)

# Conduct variance thresholding
x = thresholder.fit_transform(x, y)

# View first five rows with features with variances above threshold
print(x[0:5])



'''
This is the feature selection of SelectPercentile
'''

X_new = SelectPercentile(chi2, percentile=10).fit(x, y)

print("Pvalues_", X_new.pvalues_)
print("Scores_", X_new.scores_)
x = X_new.fit_transform(x, y)


'''
Feature Selection using Mutual_info
'''
sel_mutual = SelectKBest(mutual_info_classif, k=4).fit(x, y)
x = sel_mutual.fit_transform(x, y)
#print(x)


'''
Feature selection for SelectKBest
'''
selector_kbest = SelectKBest(score_func=chi2, k=20).fit(x, y)
x = selector_kbest.fit_transform(x, y)
#print(selector_kbest.pvalues_)

#print("shape:", selector_kbest.shape)
#print("pvalues_:", selector_kbest.pvalues_)
#print("scores_:", selector_kbest.scores_)

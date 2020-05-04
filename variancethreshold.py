#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 12:46:54 2020

@author: emma
"""

'''
Feature Selection for VarianceThreshold
'''

thresholder = VarianceThreshold(threshold=.5).fit(x, y)

# Conduct variance thresholding
x = thresholder.fit_transform(x, y)

# View first five rows with features with variances above threshold
print(x[0:5])

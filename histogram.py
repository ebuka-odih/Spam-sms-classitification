import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

sns.set()
fig = plt.figure()
ax = fig.add_subplot(111)

from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
dataset = pd.read_csv('disambiguate_spam.csv', encoding='latin-1')

#print(dataset.groupby("labels").describe())
dataset['length'] = dataset['message'].apply(len)
ham = dataset[dataset["label"] == "ham"]
spam = dataset[dataset["label"] == "spam"]

#print(ham['length'].describe())
#print(spam['length'].describe())

num_bins = 100
x = np.array(ham['length'])
#x = np.array(spam['length'])
# the histogram of the data
n, bins, patches = ax.hist(x, num_bins, facecolor='purple')

# add a 'best fit' line
import matplotlib.mlab as mlab
#mean = 58.401820
#std = 73.713694
#sns.kdeplot(ham['length'])
#y = mlab.normpdf(bins, mean, std)
#plt.plot(bins, y, 'r--')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.title('Ham')

# Tweak spacing to prevent clipping of ylabel
plt.subplots_adjust(left=0.15)
plt.savefig("word_ham_distribution.pdf")
plt.show()
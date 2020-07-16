# -*- coding: utf-8 -*-
#
# Time-stamp: <Friday 2020-07-17 09:00:15 AEST Graham Williams>
#
# Copyright (c) Togaware Pty Ltd. All rights reserved.
# Licensed under the MIT License.
# Author: Graham.Williams@togaware.com
#
# ml demo azcv
#
# This demo is based on:
#
# https://github.com/boulaziz/Analysis-of-the-Iris-Data-Set

from mlhub.pkg import mlask, mlcat, mlpreview

mlcat("Python Machine Learning with the Iris Dataset", """\
Welcome to a demo of bulding machine learning models using Python.
The well known iris dataset is used. The dataset contains 150 observations of
iris flowers. Each observation records the petal and sepal width and length
in centimeters. Also recorded for each observation is the species of the flower,
being one of setosa, versicolor, and virginica.

The dataset is read in from a CSV (comma separated values) file, as
can be readily found on the Internet. This one comes from
https://access.togaware.com/iris.csv.
""")

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------

# Import the required libraries.

import pandas
import re

import matplotlib.pyplot as plt

from pandas.plotting  import scatter_matrix

from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Demonstrate loading a CSV file rather than sklearn.datasets.load_iris()
#
# Dataset from https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/639388c2cbc2120a14dcf466e85730eb8be498bb/iris.csv

ds = pandas.read_csv('iris.csv')

mlask(end=True)

mlcat("Review the Dataset", """\
We review the dataset from the first 5 observations, the last 5 and 
then a random sample of 5 observations. The numbers on the left are the offsets
of each observation, so they start at 0 for the first observation and end at 149
for the 150th observation. Python natively works with offsets rather than counts.
If you are familliar with R then you will know that R uses a count as the index
rather and an offset.
""")

print(ds.head())
print()
print(ds.tail())
print()
print(ds.sample(5))

mlask(True, True)

mlcat("Statisics of the Dataset", """\
A simple set of statistics gives some more insight into the data. Here
we see the count of the number of observations of each variable, the
mean value, standard deviation, and then the minimum and maximum together
with the three percentiles 25%, 50%, and 75%.
""")

print(ds.describe().round(1))

mlask(True, True)

mlcat("Identifying Missing Values", """\
It is always good to check for missing values and to then determine how they
should be handled. In this dataset there are no missing values but in most
datasets we will find many. For each variable we count the number of missing
values.""")

print(ds.isnull().sum().to_frame().rename(columns={0:''}))

mlask(True, True)

mlcat("Box and Wiskers", """\
A box and wiskers plot provides a quick visual insight into how the values of each
of the variables of the observations are distributed. The box contains 50% of the
values of the observations of the variable. 
The horizontal line within the box is the median value
and the notch represents a confidence interval around the mdeian.
The top and bottom bars show the minimum and maximum values, excluding outliers,
and extend to the other 25% up and 25% down of the observations. Any points outside of 
what is called the interquartile range is an outlier.

Close the graphic window using Ctrl-W.
""")

ds.plot(kind='box', notch=True, subplots=True, layout=(2,2), sharex=False, sharey=False)
plt.show()

mlask(end=True)

mlcat("Histograms", """\
A histogram provides a visual insight into the distribution of the values of each
of the observations of the variables. The x-axis corresponds to the values of the
variable whilst the y-axis is the count.

Close the graphic window using Ctrl-W.
""")

ds.hist()
plt.show()

mlask(end=True)

mlcat("Scatter Plot", """\
A scatter plot explores the distribution of the values 
of two variables, pair-waise over all variables. Do you see any patterns
in the data here?

Close the graphic window using Ctrl-W.
""")

scatter_matrix(ds)
plt.show()

mlask(end=True)

mlcat("Building Models", """\
It's time to build our machine learning models. The first task is to build training
and test datasets. As the names suggest, the training dataset is used to train (build
or fit) the model and the test dataset is used to evaluate the performance of the
model on otherwise unseen observations.
""")

split = 0.50

array = ds.values
X = array[:,0:4]
Y = array[:,4]

seed = 42

Xtr, Xte, Ytr, Yte = model_selection.train_test_split(X, Y,
                                                      test_size=split,
                                                      random_state=seed)

mlcat("", f"""\
The original dataset of {len(array)} observations is randomly partitioned
into {round(len(array)*(1-split))} observations for the training dataset and
{round(len(array)*split)} for the test dataset.

Typically, the training set will be 80% of the available data and the
testing dataset 20%.
""")

mlask(end=True)

models = []
models.append(('CART', DecisionTreeClassifier()))
#models.append(('LR',   LogisticRegression()))
models.append(('LDA',  LinearDiscriminantAnalysis()))
models.append(('KNN',  KNeighborsClassifier()))
models.append(('NB',   GaussianNB()))
models.append(('SVM',  SVC(gamma='scale')))

names = ""

for name, model in models:
    names = f"{names}, {name}"
names = re.sub(r", ([^,]*)$", ", and \\1", re.sub("^, ", "", names))

splits = 10
scoring = 'accuracy'

mlcat("Building Models", f"""\
We now build models that will take the training dataset of known species
and using the input variables (the flower dimensions) will fit a model to
identify the type of iris. 

We will use {len(models)} different algorithms to demonstrate,
including {names}.

Note that the training set here is tiny, with few observations of few variables,
and so the task is not particularly challenging.

The table below reports on the models that have been fit and their {scoring}. 
As is common the training dataset itself is split into {splits}
so-called folds, for a {splits}-fold cross-validation training paradigm. 
Typically, {splits-1} of the {splits} folds will be used to fit the model and the
{scoring} is calculated over the hold-out fold. This will be done {splits} times
to provide an average {scoring} and its standard deviation for each algorithm.
""")

results = []
names = []
print("Model: Mean  Std")
kfold = model_selection.KFold(n_splits=splits)
for name, model in models:
    cv = model_selection.cross_val_score(model, Xtr, Ytr, cv=kfold, scoring=scoring)
    results.append(cv)
    names.append(name)
    print(f"{name:5}: {round(cv.mean(), 2)} {round(cv.std(), 2)}")

mlask(True, True)

########################################################################
# KNN PERFORMANCE

mlcat("KNN Performance", f"""\
We can delve into the model performance a little more, considering the
accuracy of the model on the test dataset, revieing the so-called
confusion matrix, and then a general performance report. These
are generated by applying the model to the test dataset
that we held out earlier. 

Using this test dataset that has not otherwise been 
used in building the model provides an independent evaluation of the
performance of the model.
""")

knn = KNeighborsClassifier()
knn.fit(Xtr, Ytr)
Ypr = knn.predict(Xte)

acc = accuracy_score(Yte, Ypr)
cm  = confusion_matrix(Yte, Ypr)

mlcat("", f"""\
On the test dataset the KNN model's accuracy is
{acc.round(2)} or {100*acc:.0f}%.

The confusion matrix is listed below as a numeric array and displayed as a plot. 
The rows correspond to the true classes and the columns correspond 
to the predicted classes. This provides a quick visual evaluation of the
accuracy of the model with respect to the different classes.
""")

print(cm)
print()

mlcat("", """\
Close the graphic window using Ctrl-W.
""")

plot_confusion_matrix(knn, Xte, Yte)
plt.show()

mlask(end=True)

mlcat("Classification Report", """\
The following is then a summary report on the performance of the model on
the test data.

We often talk in terms of true/false positives/negatives, abbreviated
respectively TP, FP, TN, FN. TP is then the number of positive
observations (e.g., setosa) that are actually positive observations
(i.e., setosa). Similary for the others.

Precision is a measure of how accuracte the model is on predicting the
class or TP/(TP+FP).

Recall is a measure of how many actual observations in this class were
predicted to be in this class or TP/(TP+FN).

The F1-score is the proportion of the predictions for a specific class
that are actually correct or 2*Re*Pr/(Re+Pr).

Support is the number of occurences of each class.
""")

print(classification_report(Yte, Ypr))

mlask(end=True)

########################################################################
# DECISION TREE PERFORMANCE

mdt = DecisionTreeClassifier()
mdt.fit(Xtr, Ytr)
Ypr = mdt.predict(Xte)

acc = accuracy_score(Yte, Ypr)
cm  = confusion_matrix(Yte, Ypr)

mlcat("", f"""\
Similarly on the test dataset with the CART (decision tree) model the
accuracy is {acc.round(2)} or {100*acc:.0f}%.

The confusion matrix and plot are similar.
""")

print(cm)
print()

mlcat("", """\
Close the graphic window using Ctrl-W.
""")

plot_confusion_matrix(mdt, Xte, Yte)
plt.show()

mlask(end=True)

mlcat("Classification Report", end="")

print(classification_report(Yte, Ypr))

mlask(end=True)

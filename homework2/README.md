Naïve Bayes
===========
project description
--------------------
* Preprocess the dataset and remove low frequency words from the document.<br>
* Implement the Naive Bayes classifier and test its effect on 20Newsgroups.

DataSet
=======
[20Newsgroups](http://qwone.com/~jason/20Newsgroups/)

Requirements
---------------

* python==3.5<br>

File introduction
-------------------
* DataDict.py:<br>
Divide the data set, 80% for the training set and 20% for the data set. <br>
The document content is segmented, punctuation, lowercase, stemming, etc.<br>
Traverse the document statistics word frequency, remove less than 4 low frequency words, and obtain a dictionary.<br>
Clean the file according to the dictionary and remove the words that are not included in the dictionary.<br>

* Bayes.py:<br>
Get the number of occurrences of each word in each class and the total number of words in each class.<br>
Adopt multivariate distribution model to calculate conditional probability and prior probability.<br>
Get the probability of the test sample in each category and get the final calculation accuracy.<br>

Result
======
The accuracy is 84.6%.

The data set in the intermediate process is too large, so it is no longer uploaded.


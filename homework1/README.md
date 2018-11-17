VSM and KNN
===========
project description
--------------------
* Preprocess the text dataset to and get a VSM representation of each text<br>
* Implement the KNN classifier and test its effect on 20Newsgroups.

DataSet
------
[20Newsgroups](http://qwone.com/~jason/20Newsgroups/)

Requirements
---------------

* python==3.5<br>
* numpy<br>
* nltk and nltk data<br>
* Other required packages

File introduction
-------------------
* VSM.py:<br>
Divide the data set, 80% for the training set and 20% for the data set. <br>
Perform a 5-fold cross-validation on 80% of the data set, and divide it into 5 copies, and execute them in sequence.<br>
The document content is segmented, punctuation, lowercase, stemming, etc.<br>
Traverse the document statistics word frequency, removing less than 4 low frequency words, and obtaining a dictionary.<br>
Clean the file according to the dictionary and remove the words that are not included in the dictionary.<br>

* IFIDFcompute.py:<br>
Calculate IDF, then calculate TF-IDF.<br>

* KNN:<br>
Load the TFIDF of the training dictionary and process the TFIDF of the test data.<br>
Use cosine similarity to calculate the vector distance between the test document and all training samples.<br>
Find the class with the largest distance and calculate the correct rate of the test data set.<br>

Result
-------
Only running in K=10,20,30,40,50.After a 5-fold cross-validation, get the ErrorRate.<br>
K=10: 0.183474<br>
K=20: 0.1816788<br>
K=30: 0.1760158<br>
K=40: 0.1722186<br>
K=50: 0.1691542<br>
Select the appropriate K=50 and get the correct rate of 85%.

The data set in the intermediate process is too large, so it is no longer uploaded.


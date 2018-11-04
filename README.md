VSM and KNN
===========
project description
--------------------
Preprocess the text dataset to and get a VSM representation of each text<br>
Implement the KNN classifier and test its effect on 20Newsgroups.

Requirements
---------------

python==3.5<br>
numpy<br>
nltk and nltk data<br>
Other required packages

File introduction
-------------------
VSM.py:<br>
Divide the data set, 80% for the training set and 20% for the data set. <br>
Perform a 5-fold cross-validation on 80% of the data set, and divide it into 5 copies, and execute them in sequence.<br>
The document content is segmented, punctuation, lowercase, stemming, etc.<br>
Traverse the document statistics word frequency, removing less than 4 low frequency words, and obtaining a dictionary.<br>
Clean the file according to the dictionary and remove the words that are not included in the dictionary.<br>

IFIDFcompute.py:<br>
Calculate IDF, then calculate TF-IDF.<br>

KNN:<br>
Load the TFIDF of the training dictionary and process the TFIDF of the test data.<br>
Use cosine similarity to calculate the vector distance between the test document and all training samples.<br>
Find the class with the largest distance and calculate the correct rate of the test data set.<br>

Result
======
After a 5-fold cross-validation, select the appropriate K value and get the final correct rate of 85%.

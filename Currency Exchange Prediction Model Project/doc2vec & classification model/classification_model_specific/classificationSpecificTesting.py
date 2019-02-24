import csv
import pickle
import numpy as np
import xlrd

import logging
from gensim.models import doc2vec, Doc2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

#can alter between id and cn model
d2v_model= Doc2Vec.load("classification_model_specific/quarter_sample_d2v/d2v_cn.model")

loc = ("C:/Users/andre/Desktop/50.038/project/china_news.xlsx")
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)
nRows = sheet.nrows
# Classification Model Testing
print("CLASSIFICATION MODEL TESTING ...")
sentenceVecList = []
sentimentList = []
i = int(nRows*(6.0/100))
while (i < nRows*(15.0/100)):
    j=1
    # print("new row...: "+str(i))
    # j=1
    # print("new row...: "+str(i))
    rowVectorAvg = []
    # while j < (nCols):
    # print("new col...: "+str(j))
    # if (j == 1):
    sentimentList.append(str(sheet.cell_value(i, 20)))
    # else:
    sentence = str(sheet.cell_value(i, 4))
    sentenceVec = d2v_model.infer_vector(sentence)
    rowVectorAvg.append(sentenceVec)
    # j+=1
    rowVectorAvg = np.sum(rowVectorAvg, axis=0)
    sentenceVecList.append(rowVectorAvg)
    i += 1

testing_labels = np.array(sentimentList)
testing_vectors = np.array(sentenceVecList)

print("LOGISTIC REGRESSION")
model1 = pickle.load(open("C:/Users/andre/Desktop/50.038/project/tobesubmitted/doc2vec & classification model/classification_model_specific/quarter_sample_d2v/classification model (cn)/logistic_regression_model.sav", 'rb'))
testing_predictions = model1.predict(testing_vectors)
# print(np.unique(testing_predictions))
print(accuracy_score(testing_labels, testing_predictions))
print(f1_score(testing_labels, testing_predictions, average='weighted'))


# ---- Decision Tree -----------
from sklearn import tree
print("DECISION TREE")
model2 = pickle.load(open("C:/Users/andre/Desktop/50.038/project/tobesubmitted/doc2vec & classification model/classification_model_specific/quarter_sample_d2v/classification model (cn)/decision_tree_model.sav", 'rb'))
testing_predictions = model2.predict(testing_vectors)
# print(np.unique(testing_predictions))
print(accuracy_score(testing_labels, testing_predictions))
print(f1_score(testing_labels, testing_predictions, average='weighted'))

# ----- Random Forest ---------------
from sklearn.ensemble import RandomForestClassifier
print("RANDOM FOREST")
model3 = pickle.load(open("C:/Users/andre/Desktop/50.038/project/tobesubmitted/doc2vec & classification model/classification_model_specific/quarter_sample_d2v/classification model (cn)/random_forest_model.sav", 'rb'))
testing_predictions = model3.predict(testing_vectors)
# print(np.unique(testing_predictions))
print(accuracy_score(testing_labels, testing_predictions))
print(f1_score(testing_labels, testing_predictions, average='weighted'))

# ------ SVM Classifier ----------------
from sklearn.svm import SVC
print("SVM CLASSIFIER")
model4 = pickle.load(open("C:/Users/andre/Desktop/50.038/project/tobesubmitted/doc2vec & classification model/classification_model_specific/quarter_sample_d2v/classification model (cn)/svm_classifier_model.sav", 'rb'))
testing_predictions = model4.predict(testing_vectors)
# print(np.unique(testing_predictions))
print(accuracy_score(testing_labels, testing_predictions))
print(f1_score(testing_labels, testing_predictions, average='weighted'))

# -------- Nearest Neighbors ----------
from sklearn import neighbors
print("NEAREST NEIGHBORS")
model5 = pickle.load(open("C:/Users/andre/Desktop/50.038/project/tobesubmitted/doc2vec & classification model/classification_model_specific/quarter_sample_d2v/classification model (cn)/nearest_neighbors_model.sav", 'rb'))
testing_predictions = model5.predict(testing_vectors)
# print(np.unique(testing_predictions))
print(accuracy_score(testing_labels, testing_predictions))
print(f1_score(testing_labels, testing_predictions, average='weighted'))

# ---------- SGD Classifier -----------------
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
print("SGD CLASSIFIER")
model6 = pickle.load(open("C:/Users/andre/Desktop/50.038/project/tobesubmitted/doc2vec & classification model/classification_model_specific/quarter_sample_d2v/classification model (cn)/sgd_classifier_model.sav", 'rb'))
testing_predictions = model6.predict(testing_vectors)
# print(np.unique(testing_predictions))
print(accuracy_score(testing_labels, testing_predictions))
print(f1_score(testing_labels, testing_predictions, average='weighted'))

# --------- Gaussian Naive Bayes ---------
from sklearn.naive_bayes import GaussianNB
print("GAUSSIAN NAIVE BAYES")
model7 = pickle.load(open("C:/Users/andre/Desktop/50.038/project/tobesubmitted/doc2vec & classification model/classification_model_specific/quarter_sample_d2v/classification model (cn)/gaussian_naive_bayes_model.sav", 'rb'))
testing_predictions = model7.predict(testing_vectors)
# print(np.unique(testing_predictions))
print(accuracy_score(testing_labels, testing_predictions))
print(f1_score(testing_labels, testing_predictions, average='weighted'))

# ----------- Neural network - Multi-layer Perceptron  ------------
from sklearn.neural_network import MLPClassifier
print("NEURAL NETWORK")
model8 = pickle.load(open("C:/Users/andre/Desktop/50.038/project/tobesubmitted/doc2vec & classification model/classification_model_specific/quarter_sample_d2v/classification model (cn)/neural_network_multi_layer_perceptron_model.sav", 'rb'))
testing_predictions = model8.predict(testing_vectors)
# print(np.unique(testing_predictions))
print(accuracy_score(testing_labels, testing_predictions))
print(f1_score(testing_labels, testing_predictions, average='weighted'))
import pickle
import numpy as np
import xlrd
from gensim.models import doc2vec, Doc2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

d2v_model= Doc2Vec.load("classification_model_specific/quarter_sample_d2v/d2v_cn.model")

# Pre process data for classification model
# Read data from excel file

loc = ("C:/Users/andre/Desktop/50.038/project/china_news.xlsx")
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)
sentenceVecList = []
sentimentList = []
nRows = sheet.nrows
print(nRows)
i = 1


while i < (nRows*(6.0/100)):
    rowVectorAvg = []
    sentimentList.append(str(sheet.cell_value(i, 20)))
    sentence = str(sheet.cell_value(i,4))
    sentenceVec = d2v_model.infer_vector(sentence)
    rowVectorAvg.append(sentenceVec)
    rowVectorAvg = np.sum(rowVectorAvg,axis=0)
    sentenceVecList.append(rowVectorAvg)
    i+=1

training_labels = np.array(sentimentList)
training_vectors = np.array(sentenceVecList)

# Classification Model Training
print("CLASSIFICATION MODEL TRAINING ...")

model1 = LogisticRegression()
model1.fit(training_vectors, training_labels)
training_predictions = model1.predict(training_vectors)
# model1.
# print(np.unique(training_predictions))
filename = 'logistic_regression_model.sav'
pickle.dump(model1, open(filename, 'wb'))
print(accuracy_score(training_labels, training_predictions))
print(f1_score(training_labels, training_predictions, average='weighted'))

# ---- Decision Tree -----------
from sklearn import tree
print("DECISION TREE")
model2 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
model2.fit(training_vectors, training_labels)
training_predictions = model2.predict(training_vectors)
# model2.save("DT.model")
# print(np.unique(training_predictions))
filename = 'decision_tree_model.sav'
pickle.dump(model2, open(filename, 'wb'))
print(accuracy_score(training_labels, training_predictions))
print(f1_score(training_labels, training_predictions, average='weighted'))

# ----- Random Forest ---------------
from sklearn.ensemble import RandomForestClassifier
print("RANDOM FOREST")
model3 = RandomForestClassifier(n_estimators=10)
model3.fit(training_vectors, training_labels)
training_predictions = model3.predict(training_vectors)
# model3.save("RF.model")
# print(np.unique(training_predictions))
filename = 'random_forest_model.sav'
pickle.dump(model3, open(filename, 'wb'))
print(accuracy_score(training_labels, training_predictions))
print(f1_score(training_labels, training_predictions, average='weighted'))

# ------ SVM Classifier ----------------
from sklearn.svm import SVC
print("SVM CLASSIFIER")
model4 = SVC()
model4.fit(training_vectors, training_labels)
training_predictions = model4.predict(training_vectors)
# model4.save("LR.model")
# print(np.unique(training_predictions))
filename = 'svm_classifier_model.sav'
pickle.dump(model4, open(filename, 'wb'))
print(accuracy_score(training_labels, training_predictions))
print(f1_score(training_labels, training_predictions, average='weighted'))

# -------- Nearest Neighbors ----------
from sklearn import neighbors
print("NEAREST NEIGHBORS")
model5 = neighbors.KNeighborsClassifier()
model5.fit(training_vectors, training_labels)
training_predictions = model5.predict(training_vectors)
# model5.save("NN.model")
# print(np.unique(training_predictions))
filename = 'nearest_neighbors_model.sav'
pickle.dump(model5, open(filename, 'wb'))
print(accuracy_score(training_labels, training_predictions))
print(f1_score(training_labels, training_predictions, average='weighted'))

# ---------- SGD Classifier -----------------
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
print("SGD CLASSIFIER")
model6 = OneVsRestClassifier(SGDClassifier())
model6.fit(training_vectors, training_labels)
training_predictions = model6.predict(training_vectors)
# model6.save("SGDC.model")
# print(np.unique(training_predictions))
filename = 'sgd_classifier_model.sav'
pickle.dump(model6, open(filename, 'wb'))
print(accuracy_score(training_labels, training_predictions))
print(f1_score(training_labels, training_predictions, average='weighted'))

# --------- Gaussian Naive Bayes ---------
from sklearn.naive_bayes import GaussianNB
print("GAUSSIAN NAIVE BAYES")
model7 = GaussianNB()
model7.fit(training_vectors, training_labels)
training_predictions = model7.predict(training_vectors)
# model7.save("GNB.model")
# print(np.unique(training_predictions))
filename = 'gaussian_naive_bayes_model.sav'
pickle.dump(model7, open(filename, 'wb'))
print(accuracy_score(training_labels, training_predictions))
print(f1_score(training_labels, training_predictions, average='weighted'))

# ----------- Neural network - Multi-layer Perceptron  ------------
from sklearn.neural_network import MLPClassifier
print("NEURAL NETWORK - MULTI LAYER PERCEPTRON")
model8 = MLPClassifier()
model8.fit(training_vectors, training_labels)
training_predictions = model8.predict(training_vectors)
# model8.save("NN-MLP.model")
# print(np.unique(training_predictions))
filename = 'neural_network_multi_layer_perceptron_model.sav'
pickle.dump(model8, open(filename, 'wb'))
print(accuracy_score(training_labels, training_predictions))
print(f1_score(training_labels, training_predictions, average='weighted'))



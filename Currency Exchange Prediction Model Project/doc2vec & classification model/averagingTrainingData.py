import numpy as np
import pylab
import csv
import xlrd
from gensim.models import Doc2Vec
from sklearn.manifold import TSNE
from ast import literal_eval

model= Doc2Vec.load("d2v.model")

# Read data from excel file
loc = ("C:/Users/andre/Desktop/50.038/project/trainingset.xlsx")
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)

# file = open('trainingSet.txt','w+')
sentenceVecList = []
sentimentList = []

nRows = sheet.nrows
nCols = sheet.ncols
print(nRows)
print(nCols)
i = 1
j = 1


while i < (nRows):
    j=1
    print("new row...: "+str(i))
    rowVectorAvg = []
    while j < (nCols):
        print("new col...: "+str(j))
        if (j == 1):
            sentimentList.append(str(sheet.cell_value(i, j)))
        else:
            sentence = sheet.cell_value(i,j)
            sentenceVec = model.infer_vector(sentence)
            rowVectorAvg.append(sentenceVec)
        j+=1
    rowVectorAvg = np.sum(rowVectorAvg,axis=0)
    sentenceVecList.append(rowVectorAvg)
    i+=1

test1 = np.array(sentimentList)
test2 = np.array(sentenceVecList)
# csvData = []
#
# k  = 0
# while k <test1.size:
#     tempData = []
#     tempData.append(test1[k])
#     tempData.append(str(test2[k]))
#     csvData.append(tempData)
#     k+=1
# print(csvData)
# with open('averageTrainingData.csv', 'w') as csvFile:
#     writer = csv.writer(csvFile)
#     writer.writerows(csvData)
# csvFile.close()


# X = np.array(sentenceVecList)
# print(X)
# print(X.shape)
# X_embedded = TSNE(n_components=2).fit_transform(X)
# print(X_embedded.shape)
# pylab.scatter(X_embedded[:, 0], X_embedded[:, 1], 20)
# pylab.show()
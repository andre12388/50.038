import pickle
import numpy as np
import xlrd
import xlsxwriter as xlsxwriter
from gensim.models import doc2vec, Doc2Vec
from googletrans import Translator
translator = Translator()

loc = ("C:/Users/andre/Desktop/50.038/project/tobesubmitted/recent news/recent_indo_news.xlsx")
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)

d2v_model= Doc2Vec.load("quarter_sample_d2v/d2v_id.model")
nRows = sheet.nrows

# Classification Model Testing
print("CLASSIFICATION MODEL TESTING ...")
sentenceVecList = []
sentimentList = []
data = ([])

i = 1
while (i < nRows):
    rowVectorAvg = []
    sentence = str(sheet.cell_value(i, 5).encode("utf-8"))
    sentenceTranslated = translator.translate(sentence)
    sentenceTranslated = sentenceTranslated.text
    data.append([sentenceTranslated])
    sentenceVec = d2v_model.infer_vector(sentenceTranslated)
    rowVectorAvg.append(sentenceVec)
    rowVectorAvg = np.sum(rowVectorAvg, axis=0)
    sentenceVecList.append(rowVectorAvg)
    i += 1

print(data)

testing_vectors = np.array(sentenceVecList)

# ---------- SGD Classifier -----------------
print("SGD Classifier")
model = pickle.load(open("C:/Users/andre/Desktop/50.038/project/doc2vec tutorial/classification_model_specific/quarter_sample_d2v"
                          "/classification model (cn)/sgd_classifier_model.sav", 'rb'))
testing_predictions = model.predict(testing_vectors)
print(testing_predictions)

j = 0
for sentiment in testing_predictions:
    data[j].append(sentiment)
    j+=1

print(data)

workbook = xlsxwriter.Workbook('recent_indo_news_annotated.xlsx')
worksheet = workbook.add_worksheet()

# Start from the first cell. Rows and columns are zero indexed.
row = 0
col = 0

# Iterate over the data and write it out row by row.
for item, cost in (data):
    worksheet.write(row, col,     item)
    worksheet.write(row, col + 1, cost)
    row += 1

workbook.close()


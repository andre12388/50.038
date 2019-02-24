import xlrd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

# Remove stop words and tokenize sentences
def tokenizer_remove_stopwords(sentence):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(sentence)
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words and w not in punctuation:
            filtered_sentence.append(str(w))
    return filtered_sentence

# Extracting data from excel file
loc = ("C:/Users/andre/Desktop/50.038/project/indo_news.xlsx")
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)

trainingData = []

nRows = sheet.nrows
# nCols = sheet.ncols
print(nRows)
# print(nCols)
i = 1
# j = 2

while i < (nRows*(50.0/100)): #only use first 80% of dataset for training
    # j=2
    print("new row...: "+str(i))
    # while j < (nCols):
    #     print("new col...: "+str(j))
    # token = tokenizer_remove_stopwords(sheet.cell_value(i,j))
    sentence = sheet.cell_value(i,4)
    # print(sentence)
    sentence = ' '.join([word for word in sentence.split() if word not in stopwords.words("english") or word not in punctuation])
    trainingData.append(sentence)
    # print (tokenizeSentenceList)
    # j+=1
    i+=1

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(trainingData)]
max_epochs = 100
vec_size = 100
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v_id.model")
print("Model Saved")


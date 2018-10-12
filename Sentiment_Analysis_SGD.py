import pandas as pd
import numpy as np
import os, pickle, json
from nltk.corpus import stopwords as sw
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics import classification_report as clsr, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import TfidfVectorizer


base_dir = "/home/mayank/Downloads/"
path = "/home/mayank/Downloads/"
test_path = "/home/mayank/sgd/aclImdb/test/"


def preprocessing(inpath, name, outpath=base_dir, mix=False):
    stopwords = set(sw.words('english'))
    indices = []
    text = []
    rating = []
    j = 0
    pos_rev = os.listdir(inpath+"pos")
    for i in pos_rev[:30000]:
        with open(path+"pos/"+i, 'r') as f:
            data = json.load(f)
        if data['language'] == 'english':
            text.append(data['text'])
            rating.append('Positive')
            indices.append(j)
            j += 1

    print(len(indices))
    neg_rev = os.listdir(path+"neg")
    j = 0
    for i in neg_rev:
        with open(path+"neg/"+i, 'r') as f:
            data = json.load(f)
        if data['language'] == 'english':
            text.append(data['text'])
            rating.append('Negative')
            indices.append(j)
            j += 1

    Dataset = list(zip(indices, text, rating))

    if mix:
        np.random.shuffle(Dataset)
    print("Writing Data")
    df = pd.DataFrame(data=Dataset, columns=['row_Number', 'text', 'polarity'])
    df.to_csv(outpath+name, index=False, header=True)


def remove_stopwords(text, stopwords):
    textwords = text.split()
    resultwords = [word for word in textwords if word.lower() not in stopwords]
    result = ' '.join(resultwords)
    return result


def tfidf_process(data):
    vector = TfidfVectorizer()
    vector = vector.fit(data)
    return vector


def retrieve_data(name, path=base_dir):
    data = pd.read_csv(path+name, header=0, encoding="ISO-8859-1")
    x = data['text']
    y = data['polarity']
    return x, y


def stochastic_descent(xtrain, ytrain, xtest):
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=10, random_state=42, alpha=1e-3, tol=None)
    print("SGD Fitting")
    clf.fit(xtrain, ytrain)
    # Saving the model with pickle
    with open(base_dir+"Model", 'wb') as f:
        pickle.dump(clf, f)
    print("SGD Predicting")
    ytest = clf.predict(xtest)

    return ytest


def write_txt(data, name):
    data = ''.join(str(word) for word in data)
    file = open(base_dir+name, 'w')
    file.write(data)
    file.close()


#preprocessing(inpath=path, name="data.csv", mix=True)
#print("Preprocessing the Test Data")
#preprocessing(inpath=test_path, name="imdb_te.csv", mix=True)
[xtrain, ytrain] = retrieve_data("reviews.csv")
#[xtest, ytest] = retrieve_data("imdb_te.csv")
xtrain, xtest, ytrain, ytest = tts(xtrain, ytrain, test_size=0.5)
labels = LabelEncoder()
ytrain = labels.fit_transform(ytrain)
ytest = labels.fit_transform(ytest)
print("--------------Vectorizing  on Sample Data----------------")
tfidf_vector = tfidf_process(xtrain)
with open(base_dir + "vectorizer", 'wb') as f:
    pickle.dump(tfidf_vector, f)
xtrain_tf = tfidf_vector.transform(xtrain)
print("--------------Vectorizing  on Test Data----------------")
xtest_tf = tfidf_vector.transform(xtest)
ypred = stochastic_descent(xtrain_tf, ytrain, xtest_tf)
print(ypred, labels.classes_)
write_txt(ypred, name="output.txt")
print("\nAccuracy Score : ", accuracy_score(ytest, ypred))
print("\nConfusion : \n", confusion_matrix(ytest, ypred))
print("\nCLSR----------------------\n", clsr(ytest, ypred, target_names=labels.classes_))

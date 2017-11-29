from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf
from sklearn.metrics import f1_score, precision_score

ktf.set_session(tf.InteractiveSession())

y = open('./vec_code_labels100.txt', 'r').read().split('\n')
Y = []
for i in range(len(y)):
    if len(y[i].split()) == 0:
        continue
    Y.append(list(map(float, y[i].split())))
y = np.array(Y, np.float32)
print(y.shape)

data = open('./processed_docs.txt', 'r').read().split('\n')[:-1]
vectorizer = TfidfVectorizer(input='content', token_pattern=r'\S+', analyzer='word', min_df=0.0002)

X = vectorizer.fit_transform(data).todense()
print(X.shape)
num_tags = y.shape[1]
features = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

model = load_model('./model.h5')

def crange(st, ed, jump):
    i = st
    while i <= ed + 1e-9:
        yield i
        i += jump

for td in crange(0.1, 0.91, 0.1):
    pred = model.predict(X_test)
    for i in range(pred.shape[0]):
        threshold = td * np.max(pred[i])
        pred[i, np.where(pred[i] >= threshold)[0]] = 1.0
        pred[i, np.where(pred[i] < threshold)] = 0.0
    print("%f F1: %f" % (td, f1_score(y_test, pred, average='samples')))

# for numlabels in range(1, 6):
    # pred = model.predict(X_test)
    # for i in range(pred.shape[0]):
        # pred[i, pred[i].argsort()[-numlabels:][::-1]] = 1.0
        # pred[i, np.where(pred[i] <= 0.99)[0]] = 0.0

    # print("Acc:", np.sum(pred == y_test) / (y_test.shape[0] * y_test.shape[1]))

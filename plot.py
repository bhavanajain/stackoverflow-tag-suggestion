from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from keras.models import load_model
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

ktf.set_session(tf.InteractiveSession())

y = open('./vec_code_labels25.txt', 'r').read().split('\n')
Y = []
for i in range(len(y)):
    if len(y[i].split()) == 0:
        continue
    Y.append(list(map(float, y[i].split())))
y = np.array(Y, np.float32)
print(y.shape)

data = open('./processed_code_docs25.txt', 'r').read().split('\n')[:-1]
vectorizer = TfidfVectorizer(input='content', token_pattern=r'\S+', analyzer='word', min_df=0.02)
X = vectorizer.fit_transform(data).todense()
print(X.shape)
num_tags = y.shape[1]
features = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

model = load_model('model_25.h5')

print(model.evaluate(X_test, y_test, batch_size=1))
pred = model.predict(X_test)

for i in range(pred.shape[0]):
    numlabels = int(np.sum(y_test[i]))
    pred[i, pred[i].argsort()[-numlabels:][::-1]] = 1.0
    pred[i, np.where(pred[i] <= 0.99)[0]] = 0.0

print(np.sum(pred == y_test) / (y_test.shape[0] * y_test.shape[1]))

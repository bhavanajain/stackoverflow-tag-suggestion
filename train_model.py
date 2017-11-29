from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
from keras.layers.advanced_activations import LeakyReLU

import tensorflow as tf
import keras.backend.tensorflow_backend as ktf

ktf.set_session(tf.InteractiveSession())

y = open('./labels.txt', 'r').read().split('\n')
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

model = Sequential()
model.add(Dense(10500, activation='linear', input_dim=features))
model.add(LeakyReLU(alpha=.1))
model.add(Dense(2048, activation='linear'))
model.add(LeakyReLU(alpha=.1))
model.add(Dense(512, activation='linear'))
model.add(LeakyReLU(alpha=.1))
model.add(Dense(256, activation='linear'))
model.add(LeakyReLU(alpha=.1))
model.add(Dense(num_tags, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)

model.save('./model.h5')
print("Model trained and saved")

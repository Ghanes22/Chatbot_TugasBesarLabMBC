import json
from re import A
import random
import numpy as np
import pickle
import nltk 
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

data_file = open('intents.json').read()
intents = json.loads(data_file)

words = []
classes = []
docs = []
ignore_letters = ["!","?",".",","]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        docs.append((word_list, intent['tag']))
        if intent['tag'] not in classes :
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]

words = sorted(set(words))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0]*len(classes)

for doc in docs:
	bag = []
	pattern_words = doc[0]
	pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
	#print('Current Pattern Words: {}'.format(pattern_words))

	for w in words:
		bag.append(1) if w in pattern_words else bag.append(0)

	#print('Current Bag: {}'.format(bag))

	output_row = list(output_empty)
	output_row[classes.index(doc[1])] = 1
	#print('Current Output: {}'.format(output_row))

	training.append([bag, output_row])

#print('Training: {}'.format(training))
random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])
print('X: {}'.format(train_x))
print('Y: {}'.format(train_y))

model = Sequential()
model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# kompilasi model & tentukan fungsi pengoptimal
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])

mfit = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)
model.save('chatbot_model.h5', mfit)

print('Model Kamu Sudah Kelar !')

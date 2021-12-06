import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))
unknown = json.loads(open('unknown_answer.json').read())
unknown = unknown['unknown_answer']

def bow(sentence):
	sentence_words = nltk.word_tokenize(sentence)
	sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
	bag = [0]*len(words)
	for s in sentence_words:
		for i, w in enumerate(words):
			if w == s:
				bag[i]=1
	return (np.array(bag))

def predict_class(sentence):
	sentence_bag = bow(sentence)
	res = model.predict(np.array([sentence_bag]))[0]
	ERROR_THRESHOLD = 0.70
	results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
	#sort by probablity
	results.sort(key=lambda x: x[1], reverse=True)
	print(results)
	return_list = []
	for r in results:
		return_list.append({'intent':classes[r[0]], 'probablity':str(r[1])})
	return return_list
				

def getResponse(ints):
	if ints != []:
		tag = ints[0]['intent']
		list_of_intents = intents['intents']
		for i in list_of_intents:
			if(i['tag']==tag):
				result=random.choice(i['responses'])
				break
	else:
		result = random.choice(unknown)
	return result


def chatbot_response(msg):
	ints = predict_class(msg)
	res = getResponse(ints)
	return res
	
	
from flask import Flask, render_template, request
app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/post")
def input():
	userText = request.args.get('msg')
	return chatbot_response(userText)

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)

if __name__ == "__main__":
    app.run(debug=True,port=1010)
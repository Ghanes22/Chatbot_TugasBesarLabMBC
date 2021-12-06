import tkinter
from tkinter import *
from keras.models import load_model
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import random
import numpy as np
results = []
unknown_ans = ["Maf ", "asldkasdlkas"]

intents = json.loads(open('intents.json').read())
model = load_model('chatbot_model.h5')
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


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
		result = random.choice(unknown_ans)
	return result



def chatbot_response(msg):
	ints = predict_class(msg)
	res = getResponse(ints)
	return res


def send():
	msg = TextEntryBox.get("1.0", 'end-1c').strip()
	TextEntryBox.delete('1.0', 'end')

	if msg != '':
		ChatHistory.config(state=NORMAL)
		ChatHistory.insert('end', "You: " + msg + "\n\n")

		res = chatbot_response(msg)
		ChatHistory.insert('end', "Bot: " + res + "\n\n")
		ChatHistory.config(state=DISABLED)
		ChatHistory.yview('end')

base = Tk()
base.title("Bot MBC")
base.geometry("400x500")
base.resizable(width=False, height=False)

#chat history textview
ChatHistory = Text(base, bd=0, bg='white', font='Arial')
ChatHistory.config(state=DISABLED)

SendButton = Button(base, font=('Arial', 12, 'bold'), 
	text="Send", bg="#dfdfdf", activebackground="#3e3e3e", fg="#ffffff", command=send)

TextEntryBox = Text(base, bd=0, bg='white', font='Arial')

ChatHistory.place(x=6, y=6, height=386, width=386)
TextEntryBox.place(x=128, y=400, height=80, width=265)
SendButton.place(x=6, y=400, height=80, width=125)

base.mainloop()
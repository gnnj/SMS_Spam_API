import os
import io
import numpy as np
from PIL import Image
import base64

# %matplotlib inline
import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import pandas as pd
import sklearn
from sklearn.externals import joblib
import _pickle as pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.learning_curve import learning_curve

from flask import Flask, request, redirect, jsonify, render_template

app = Flask(__name__)

def split_into_tokens(sms):
    return TextBlob(sms).words

def split_into_lemmas(message):
    message = message.lower()
    words = TextBlob(message).words
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]

spam_detect = pickle.load(open('sms_spam_detector.pkl', 'rb'))

@app.route('/')
def run():
    return render_template('index.html')

@app.route('/send', methods=['post'])
def signup():
    sms_message = request.form['text']
    sms = []
    sms.append(sms_message)
    result = spam_detect.predict(sms)
    return (result[0])

@app.route('/sms="<sms_msg>"')
def sms_check(sms_msg):
	sms_message = sms_msg
	sms = []
	sms.append(sms_message)
	result = spam_detect.predict(sms)
	return jsonify(sms[0], result[0])

if __name__ == "__main__":

    app.run(debug=True)

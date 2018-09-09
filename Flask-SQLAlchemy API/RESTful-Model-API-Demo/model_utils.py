"""model_utils.py
first draft by BHamilton 8/6/2018
for NSV Alpha_Lantern

This file contains the functions necessary to make predictions on single
user inputs, such as those collected from a web interface via an API
(which is the eventual destination here)

REQUIREMENTS:
 - a folder ./model containing the w2v model: GoogleNews-vectors-negative300.bin
 ---> fyi this is quite large. 1.5G zipped.
 - gensim
 - nltk and nltk stopwords (via nltk.download(stop_words)) <-- may need an ini file
 - keras
 - model_config.py
 - the lstm model is not loaded in this script, it should be loaded in the api

"""

import pandas as pd
import numpy as np
import gensim
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
#import keras
#from keras.models import load_model ### <--- may not be necessary here

stop_words = stopwords.words('english')
exclude_chars = set(string.punctuation)
lemma = WordNetLemmatizer()

### CONFIGURATION:
from model_config import WORD2VEC_MODEL, INPUT_LENGTH_LIMIT, W2V_LENGTH, CLASSIFICATION_THRESHOLD

# note: importing WORD2VEC_MODEL will initialize the word2vec model,
# which can take several seconds as it is quite large.

####################################################################

def clean_user_input(user_input_string):
    """
        Cleans the User's Input
        - stop words, punctuation filter, lemmatize
    """

    #filter for stop words
    stop_filtered = [word for word in user_input_string.lower().split(' ') if word not in stop_words]
    #filter for punctuation
    punc_filtered = [word for word in stop_filtered if word not in exclude_chars]
    # lemmatize
    lemma_filtered = [lemma.lemmatize(word) for word in punc_filtered]

    return lemma_filtered

###################################################################


def input_volume(user_input_string, rnn_time_steps):
    """
        Generates an input volume for the model
        Using the user input
    """
    input_list = clean_user_input(user_input_string)

    # w2v volume has 3 components:
    # arg1: # of rows; here it is 1, in training it is number of training examples
    # arg2: # of words for the rnn_timesteps
    # arg3: # of elements in the w2v encoding, 300 for the NSV_Hackfest model
    w2v = np.zeros([1, rnn_time_steps, W2V_LENGTH])

    # x is a default w2v for a single word; all zeros
    x = np.zeros([W2V_LENGTH])

    w2v_idx = 0
    for word in input_list:

        try:
            x = WORD2VEC_MODEL[word]
        except KeyError:
            pass

        w2v[0][w2v_idx] = x
        w2v_idx += 1

    return w2v

####################################################################

def user_classification(user_input, model):
    """
        Classifies the user's input string according to the violence
        types built into the model

        Assumes model is a keras model and model.predict(input_vol)
        is sufficient for operation.

        Output is a string of 0's and 1's.
    """

    input_vol = input_volume(user_input, INPUT_LENGTH_LIMIT)

    y_pred = model.predict(input_vol)

    yp = []
    for label in y_pred:
        for x in label:
            val = x
        yp.append(val)
    print yp 
    return yp

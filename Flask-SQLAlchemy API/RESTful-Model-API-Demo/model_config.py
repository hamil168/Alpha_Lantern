"""model_config.py
initial draft BHamilton 8/6/2018
for NSV Alpha_Lantern

Configuration values required to perform predictions
on individual user inputs in a deployed model

Called by model_utils.py

NOTE: if the model input changes INPUT_LENGTH_LIMIT must change
OTHER NOTE: initializing WORD2VEC_MODEL can take several seconds

"""
import gensim

WORD2VEC_MODEL = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
INPUT_LENGTH_LIMIT = 6  # specific to NSV_Hackfest model
W2V_LENGTH = 300 # sepecific to this model
CLASSIFICATION_THRESHOLD = 0.8 # cutoff for positive classification

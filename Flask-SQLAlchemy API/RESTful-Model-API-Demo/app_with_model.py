# app.py for Web Demo 2
# Modified to load a model and to call the model prediction
# and incorporate it into the Student() class attributes

from flask import Flask, request, flash, url_for, redirect, render_template
from flask_sqlalchemy import SQLAlchemy
from keras.models import load_model
import keras
from keras import backend as K
import tensorflow as tf
import numpy as np

#K.clear_session()
input_behavior = []
stack = ["Tell me more!", "Tell me more!", "Thanks for the initial information. Now, tell me about Subject's recent behaviour?", "Thanks You! What is the Subject's Age?",
"Please provide Subject User-Id", "Great! Please help with some information here", 
"Would you like my help in understanding student behaviour and help them better?"]

#Initialize Flask object and Database info
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students.sqlite3'
app.config['SECRET_KEY'] = "random string"

#Initialize SQLAlchemy object
db = SQLAlchemy(app)

# Initialize the ML model
from model_utils import *

# this is needed to make sure the ML model's get_default_graph
# stays consistent across the boundaries of the flask object
global graph
graph = tf.get_default_graph()

# The class that contains all student information
# added v_classes for violence Classification
# which will tentatively be returned as a string of 0's and 1s i.e. '0100'
class students(db.Model):

   id = db.Column('student_id', db.Integer, primary_key = True)
   name = db.Column(db.String(100))
   city = db.Column(db.String(50))
   behavior = db.Column(db.String(200))
   pin = db.Column(db.String(10))
   v_classes = db.Column(db.String(20))


   def __init__(self, name, city, behavior, pin, v_class):
    self.name = name
    self.city = city
    self.behavior = behavior
    self.pin = pin
    self.v_classes = v_class




# Shows the web front end, refreshing to show all database entries
# The '/' means anytime the url ends in '/', instead of a specific html file
@app.route('/')
def chatbot():
   return render_template('new.html')

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    if len(stack) <= 2:
      input_behavior.append(userText)
    if len(stack) == 0:
      with graph.as_default():
        op1 = user_classification(input_behavior[0],ml_model)[0]
        op2 = user_classification(input_behavior[1],ml_model)[0]
        op3 = user_classification(input_behavior[2],ml_model)[0]
        return str(op1) + str(op2) + str(op3)
    return str(stack.pop())

# # Get info POST'd from the front end, check its validit
# # If valid, populate the object 'student' with the entered info
# @app.route('/new', methods = ['GET', 'POST'])
# def new():
#     if request.method == 'POST':

#         if not request.form['name'] or not request.form['city'] or not request.form['behavior']:
#             flash('Please enter all the fields', 'error')
#         else:
#             with graph.as_default():
#                 student = students(request.form['name'], request.form['city'],
#                 request.form['behavior'], request.form['pin'],
#                 user_classification(request.form['behavior'],ml_model)[0])

#                  # Add student info to the DB
#                 db.session.add(student)
#                 db.session.commit()

#                 flash('Record was successfully added')

#          # Go to the show_all page, refreshing the front end

#         return redirect(url_for('show_all'))

#     # if not POST, go to the 'new' page, which will run the new() function
#     return render_template('new.html')




if __name__ == '__main__':

   # Creates the little database and executes with debugging on
   db.create_all()

   with graph.as_default():
       ml_model = load_model('./ml_model/model.h5')

   app.run(debug = True)

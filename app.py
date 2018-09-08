# app.py for Web Demo 2
# Modified to load a model and to call the model prediction
# and incorporate it into the Student() class attributes

from flask import Flask, request, flash, url_for, redirect, render_template
from flask_sqlalchemy import SQLAlchemy
from keras.models import load_model
import keras
from keras import backend as K
import tensorflow as tf

#K.clear_session()

#Initialize Flask object and Database info
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students0.sqlite3'
app.config['SECRET_KEY'] = "random string"

#Initialize SQLAlchemy object
db = SQLAlchemy(app)

# Initialize the ML model
from model_utils import *

# this is needed to make sure the ML model's get_default_graph
# stays consistent across the boundaries of the flask object
global graph
graph = tf.get_default_graph()

# The class that contains all student info
# added v_classes for violence Classification
# which will tentatively be returned as a string of 0's and 1s i.e. '0100'
class students(db.Model):

   id = db.Column('student_id', db.Integer, primary_key = True)
   name = db.Column(db.String(100))
   city = db.Column(db.String(50))
   behavior = db.Column(db.String(200))
   pin = db.Column(db.String(10))
   v_classes = db.Column(db.String(20))
   class1p = db.Column(db.Float())
   class2p = db.Column(db.Float())
   class3p = db.Column(db.Float())
   class4p = db.Column(db.Float())
   class1bin = db.Column(db.Integer())
   class1bin = db.Column(db.Integer())
   class1bin = db.Column(db.Integer())
   class1bin = db.Column(db.Integer())

   def __init__(self, name, city, behavior, pin, v_class, class1p, class2p, class3p, class4p,
                class1bin, class2bin, class3bin, class4bin, threshold):
    self.name = name
    self.city = city
    self.behavior = behavior
    self.pin = pin
    self.v_classes = v_class
    self.class1p = class1p
    self.class2p = class2p
    self.class3p = class3p
    self.class4p = class4p
    self.class1bin = class1bin
    self.class2bin = class2bin
    self.class3bin = class3bin
    self.class4bin = class4bin
    self.threshold = threshold


# Shows the web front end, refreshing to show all database entries
# The '/' means anytime the url ends in '/', instead of a specific html file
@app.route('/')
def show_all():
   return render_template('show_all.html', students = students.query.all() )

# Get info POST'd from the front end, check its validit
# If valid, populate the object 'student' with the entered info
@app.route('/new', methods = ['GET', 'POST'])
def new():
    if request.method == 'POST':

        if not request.form['name'] or not request.form['city'] or not request.form['behavior']:
            flash('Please enter all the fields', 'error')
        else:
            with graph.as_default():
                yp, y_pred, y_class = user_classification(request.form['behavior'],ml_model)

                student = students(request.form['name'], request.form['city'],
                request.form['behavior'], request.form['pin'],
                yp[0], y_pred[0][0], y_pred[0][1], y_pred[0][2], y_pred[0][3],
                y_class[0][0], y_class[0][1], y_class[0][2], y_class[0][3],
                CLASSIFICATION_THRESHOLD)

                 # Add student info to the DB
                db.session.add(student)
                db.session.commit()

                flash('Record was successfully added')

         # Go to the show_all page, refreshing the front end

        return redirect(url_for('show_all'))

    # if not POST, go to the 'new' page, which will run the new() function
    return render_template('new.html')

#@app.route('/endpoints/query', methods = ['GET', 'POST'])
#@app.route('/endpoints/classify_behavior', methods = ['GET','POST'])


if __name__ == '__main__':

   # Creates the little database and executes with debugging on
   db.create_all()

   with graph.as_default():
       ml_model = load_model('./ml_model/model.h5')

   app.run(debug = True)

from flask import Flask, request, flash, url_for, redirect, render_template
from flask_sqlalchemy import SQLAlchemy

#Initialize Flask object and Database info
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///students.sqlite3'
app.config['SECRET_KEY'] = "random string"

#Initialize SQLAlchemy object
db = SQLAlchemy(app)

# The class that contains all student info
class students(db.Model):
   id = db.Column('student_id', db.Integer, primary_key = True)
   name = db.Column(db.String(100))
   city = db.Column(db.String(50))
   addr = db.Column(db.String(200))
   pin = db.Column(db.String(10))

   def __init__(self, name, city, addr, pin):
    self.name = name
    self.city = city
    self.addr = addr
    self.pin = pin

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

      if not request.form['name'] or not request.form['city'] or not request.form['addr']:
         flash('Please enter all the fields', 'error')
      else:
         student = students(request.form['name'], request.form['city'],
            request.form['addr'], request.form['pin'])

         # Add student info to the DB
         db.session.add(student)
         db.session.commit()
         flash('Record was successfully added')

         # Go to the show_all page, refreshing the front end
         return redirect(url_for('show_all'))

   # if not POST, go to the 'new' page, which will run the new() function
   return render_template('new.html')

if __name__ == '__main__':

   # Creates the little database and executes with debugging on
   db.create_all()
   app.run(debug = True)

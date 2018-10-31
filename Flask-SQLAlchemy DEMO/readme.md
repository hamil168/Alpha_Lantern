## Early-Stage PoC API ##
### Using Flask (for web interface) and Flask-SQLAlchemy (for database interface)
***They're two different but related things***

Example adapted from web examples.

How to Use:
- Install flask and flask-sqlalchemy on your environment
- Put the api python script in the folder of your choice
- Create a "Templates" folder in the folder with the script
- put the html templates in the Templates folder
- run the python script from the command line

### Resources for building a RESTful ML Model API using Flask

Tutorial/Blog:
https://www.toptal.com/python/python-machine-learning-flask-example

### In Development:
- model_config.py, model_utils.py provide functions to clean and predict user data
- the model itself needs to be included as a .h5 file and loaded as part of the api
- Based on work in Prep_Cleaning_For_Deployment.ipynb in the NSV_Hackfest repo
  - https://github.com/hamil168/NSV_Hackfest/blob/master/Prep_Cleaning_For_Deployment.ipynb
- model.h5 is an LSTM developed in NSV_Hackfest, 6 words maximum input

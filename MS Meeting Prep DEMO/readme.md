# Alpha_Lantern DEMO: Prep for 9/13 MS Meeting

Contents:

**app.py**: RESTful API scripted in Python, using flask and sqlalchemy. It connects front end, database, and the ml model.

**index.html**: Front end of the MS Meeting Prep Demo that needs to be running on its own.

*TO RUN:*
- index.html and app.py need to be changed from local host ...
- ex: search for var url = 'http://localhost:5000/predict.json'

**model_config.py**: configuration settings for the machine learning model.

**model_utils.py**: utility scripts for processing user input and model output based on hamil168/NSV_Hackfest/blob/master/Prep_Cleaning_For_Deployment.ipynb

**[templates]**: web templates that can be called from the API. They are vestiges of the original web tutorial (see other DEMO folder) that still function here. They are not, however, required for the MS Meeting Prep DEMO purpose ... only index.html is.

**[ml_model]**: folder containing the serialized LSTM machine learning model from the NoSchoolViolence/NSV_Hackfest repo.

**[w2v_model]**: empty folder that requires the language model. They can be very large, so it is here as a placeholder.

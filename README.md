# Alpha_Lantern
*Putting the pieces together.*

Here we are assembling the components for a basic web implementation of the LSTM Lantern application after the MS Hackfest in May 2018

**Folders:**

***Web-Model-DB DEMO***: Implementation of a basic front end that collects input features (id, location, behavior description), uses the ML model to classify according to violence types, and return the classifications along with the input data to the front end and to the database.

***MS Meeting Prep DEMO***: Implementation of a more attractive front end with a pseudo-chat-bot interface that collects behavior inputs, uses the ML model to calculate strengths of violence type association, and returns the values to a dynamic histogram that responds to successive user inputs.

***NOTES:***
- Both DEMOs are built on the same basic Flask and Flask-SQLAlchemy backbone. They can both be connected to a database or adapted with other RESTful API endpoints.
- The LANGUAGE MODEL (gensim + googlenews word2vec) is very slow to load for development purposes. A lighter model should be used until a bigger one is required for model accuracy. As of this writing, that model has not been chosen nor a substitute ML model trained. Those should be high priorities.

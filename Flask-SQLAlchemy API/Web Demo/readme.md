


Using the example here:

https://www.tutorialspoint.com/flask/flask_sqlalchemy.htm

- The ideas in the tutorial are very helpful
- The tutorial has a big indentation error at the end script
- Running the code copied directly from the web page causes an error
- Learned the hard way
- So the app.py shared here should work fine.

***What it does:***
- Creates a mini database (a single file in the folder with app.py)
- Starts listening to http://127.0.0.1:5000/ (or http://localhost:5000/)
- If you put that url into a web browser the interface pop up
  - content dictated by the html templates
- Add entries to the database. Tada

This should be a decent starting template for stage 1 proof of concept api.
Connecting to a remote DB, adding different fields, and so on are straightforward.

Switching from sqlite3 to mssql is linked below

**Other references:**
- Info: http://flask-sqlalchemy.pocoo.org/2.3/
- Declaring models: http://flask-sqlalchemy.pocoo.org/2.3/models/#models
- Queries: http://flask-sqlalchemy.pocoo.org/2.3/queries/
- Connections: http://flask-sqlalchemy.pocoo.org/2.3/config/#connection-uri-format
  - Also: http://docs.sqlalchemy.org/en/latest/core/connections.html
  - Engines: http://docs.sqlalchemy.org/en/latest/core/engines.html
  - Dialects: http://docs.sqlalchemy.org/en/latest/dialects/mssql.html

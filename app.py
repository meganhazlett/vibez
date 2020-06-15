import traceback
from flask import render_template, request, redirect, url_for
import logging.config
from flask import Flask
from src.configure_db import Hot_100_Labeled_Clusters
from  sqlalchemy.sql.expression import func, select
from flask_sqlalchemy import SQLAlchemy


# Initialize the Flask application
app = Flask(__name__, template_folder="app/templates")

# Configure flask app from flask_config.py
app.config.from_pyfile('config/flaskconfig.py')

# Define LOGGING_CONFIG in flask_config.py - path to config file for setting
# up the logger (e.g. config/logging/local.conf)
logging.config.fileConfig(app.config["LOGGING_CONFIG"])
logger = logging.getLogger(app.config["APP_NAME"])
logger.debug('Test log')

# Initialize the database
db = SQLAlchemy(app)


@app.route('/')
def index():
    """Main view that introduces users to app and prompts them to choose a broad beer type. Uses app/templates/index.html template.
    Returns: rendered html template
    """
    logger.info('At introductory app page.')
    return render_template('index.html')


@app.route("/getplaylist", methods=['POST'])
def get_playlist(): 
	""" Generates a playlist of 20 songs based on user specified cluster
	"""
	user_input = int(request.form['VibeNumber'])
	print(user_input)
	try: 
		playlist = db.session.query(Hot_100_Labeled_Clusters).filter_by(Cluster_Num=user_input).order_by(func.random()).limit(20)
		logger.debug("20 song vibe playlist genreated")
		return render_template('recommender.html', playlist=playlist)
	except: 
		traceback.print_exc()
		logger.warning("Not able to display playlist, error page returned")
		return render_template('error.html')



if __name__ == '__main__':
    app.run(debug=app.config["DEBUG"], port=app.config["PORT"], host=app.config["HOST"])

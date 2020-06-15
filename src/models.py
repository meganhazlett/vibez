# Packages 
import sys
import json
import warnings
warnings.filterwarnings('ignore')
import argparse 
import os
import urllib
import yaml
import pandas as pd 
import numpy as np
import logging.config
logger = logging.getLogger(__name__)
import sklearn
from sklearn import model_selection
from sklearn import linear_model
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import preprocessing 
# import statsmodels.api as sm
# from statsmodels.formula.api import ols
import seaborn as sns; sns.set()
import pickle
import random

def my_configurations(args): 
	''' Calls yaml file with configuration information 
	Args: 
		args {argparse.Namespace} : Path to config file 
	Returns: 
		config : configurations for model 
	'''
	if args.config is not None:
		with open(args.config, "r") as f:
			config = yaml.load(f)
		config = config['models']
	else:
		raise ValueError("Path to config yml file must be provided through --config")

	return config 


def load_data(args): 
	'''Loads Hot100 data from file and drops NAs 
	Args: 
		args {argparse.Namespace} : Path to config file 
	Returns: 
		data (data frame): : Pandas df of Hot100 data
	'''
	# Configure 
	config = my_configurations(args)

	# Read data
	try: 
		data = pd.read_excel(config['hot_100_data'])
		logger.info("Data loaded and ready to clean.")
	except: 
		logger.error("Data not loaded. Please check config file.")

	return data


def drop_rows(data):
	'''Drops rows with NaNs
	Args: 
		data (data frame): Produced in load_data
	Returns:
		data (data frame): Without any NaNs
	'''
	if type(data) == pd.core.frame.DataFrame:
		data = data.dropna()
		logger.info("Rows with NaN dropped")
	else:
		raise ValueError("You did not enter a pandas data frame") 
		logger.error("Rows with NaN NOT dropped")
	return data


def drop_columns(data):
	'''Drops columns not being used for analysis and drop rows with nas 
	Args:
		data (data frame) : Produced in drop_rows
	Returns: 
		data_cols (data frame) : Data frame with columns not in use removed 
	'''
	if type(data) == pd.core.frame.DataFrame:
		# Drop columns
		data_drop_cols = data.drop(columns=['spotify_genre','Performer', 'Song','spotify_track_id', 'spotify_track_preview_url', 'spotify_track_album', 'spotify_track_explicit', 'key', 'time_signature'])
		# Set SongID to index 
		data_drop_cols = data_drop_cols.set_index('SongID')

		logger.info("Irrelevant columns dropped")
	else:
		raise ValueError("You did not enter a pandas data frame") 
		logger.error("Irrelevant columns NOT dropped")

	return data_drop_cols

def standardize_data(data_drop_cols): 
	'''Stardarizes the freatures in preparation of K-mean modeling
	Args: 
		data_crop_cols (data frame) : Produced in drop_columns()
	Returns: 
		data_
	'''
	try: 
		# Standardize the variables being used 
		data_drop_cols['Zspotify_track_duration_ms'] = preprocessing.scale(data_drop_cols.spotify_track_duration_ms)
		data_drop_cols['Zspotify_track_popularity'] = preprocessing.scale(data_drop_cols.spotify_track_popularity)
		data_drop_cols['Zdanceability'] = preprocessing.scale(data_drop_cols.danceability)
		data_drop_cols['Zenergy'] = preprocessing.scale(data_drop_cols.energy)
		data_drop_cols['Zloudness'] = preprocessing.scale(data_drop_cols.loudness)
		data_drop_cols['Zspeechiness'] = preprocessing.scale(data_drop_cols.speechiness)
		data_drop_cols['Zacousticness'] = preprocessing.scale(data_drop_cols.acousticness)
		data_drop_cols['Zinstrumentalness'] = preprocessing.scale(data_drop_cols.instrumentalness)
		data_drop_cols['Zliveness'] = preprocessing.scale(data_drop_cols.liveness)
		data_drop_cols['Zvalence'] = preprocessing.scale(data_drop_cols.valence)
		data_drop_cols['Ztempo'] = preprocessing.scale(data_drop_cols.tempo)

		# Keep only relevant columns
		data_kmeans = (data_drop_cols[['Zspotify_track_duration_ms', 'Zspotify_track_popularity',
	       'Zdanceability', 'Zenergy', 'Zloudness', 'Zspeechiness',
	       'Zacousticness', 'Zinstrumentalness', 'Zliveness', 'Zvalence',
	       'Ztempo']])

		# Round 
		data_kmeans = data_kmeans.round(4)
		logger.info("Data is standardized")

	except: 
		raise ValueError("Cannot compute standardizations with strings")

	return data_kmeans 


def determine_number_clusters(data_kmeans, args):  
	''' Determine the idea number of clusters for the model. Saves an elbow plot to data folder. 
	Args: 
		data_kmeans (data frame): Generated from stadardize_data
		args {argparse.Namespace} : Path to config file 
	'''
	# Config 
	config = my_configurations(args)

	# In columns 
	in_columns = ['Zspotify_track_duration_ms','Zspotify_track_popularity','Zdanceability','Zenergy',
 	'Zloudness','Zspeechiness','Zacousticness','Zinstrumentalness','Zliveness','Zvalence','Ztempo']

	if list(data_kmeans.columns) != in_columns: 
		raise KeyError
	else: 
		# Determine clusters 
		wcss = []
		for i in range(2,50):
		    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
		    kmeans.fit(data_kmeans)
		    wcss.append(kmeans.inertia_)

		# Save to file
		plt.plot(range(2,50), wcss)
		plt.title('The elbow method')
		plt.xlabel('The number of clusters')
		plt.ylabel('Within Cluster Sum of Squares')
		try: 
			print(config['ideal_clusters_plot'])
			plt.savefig(config['ideal_clusters_plot'])
			logger.info("Ideal number of clusters plot saved to folder")
		except: 
			logger.error("Ideal number of clusters plot NOT saved. Check config file")

	return None



def cluster_solution(data_kmeans, args): 
	'''Executes the 13-cluster solution for Hot100 data
	Args: 
		data_kmeans (data frame): Generated from stadardize_data
		args {argparse.Namespace} : Path to config file
	Returns: 
		clus13 (model) : 13-cluster Kmeans solution

	'''
	# Config 
	config = my_configurations(args)
	# In columns 
	in_columns = ['Zspotify_track_duration_ms','Zspotify_track_popularity','Zdanceability','Zenergy',
 	'Zloudness','Zspeechiness','Zacousticness','Zinstrumentalness','Zliveness','Zvalence','Ztempo']

	if list(data_kmeans.columns) != in_columns: 
		raise KeyError
	else: 
		# Find 13-cluster solution
		clus13 = KMeans(n_clusters=13, init = 'k-means++', max_iter=300, n_init=100, random_state= 0)
		clus13.fit(data_kmeans)
		logger.info("13 cluster solution fit")

		with open(config['final_model_path'], "wb") as f:
				pickle.dump(clus13, f)
				logger.info("Trained model object saved to model_results folder")

	return clus13


def model_eval(clus13, data_kmeans, args):
	''' Evaluates the model generated in cluster_solution 
	Args: 
		clus13 (model) : 13 cluster K-means gereated in cluster_solution
		data_kmeans (data frame): Generated from stadardize_data
		args {argparse.Namespace} : Path to config fil
	Returns: 
		labels (numpy array) : List of which cluster number a song belongs to
	'''
	# Config 
	config = my_configurations(args)
	in_columns = ['Zspotify_track_duration_ms','Zspotify_track_popularity','Zdanceability','Zenergy',
 	'Zloudness','Zspeechiness','Zacousticness','Zinstrumentalness','Zliveness','Zvalence','Ztempo']

	if list(data_kmeans.columns) != in_columns: 
		raise KeyError
	else: 
		# Refit data 
		clus13.fit(data_kmeans)

		try: 
			# Cluster centers
			centers = clus13.cluster_centers_
			centers_df = pd.DataFrame(centers)
			centers_df.to_csv(config['cluster_centers'])

			# Cluster labels 
			labels = clus13.labels_
			labels_df = pd.DataFrame(labels)
			labels_df.to_csv(config['cluster_labels'])

			# Cluster within sum of squares 
			ssw = clus13.inertia_ 
			np.savetxt(config['cluster_ssw'],np.array([ssw]))

			# Cluster counts
			counts = pd.crosstab(index=clus13.labels_, columns="count")
			counts_df = pd.DataFrame(counts)
			counts_df.to_csv(config['cluster_counts'])

			logger.info("Model evaluation CSVs in file")

		except: 
			logger.error("Model evaluation CSVs NOT in file. Check config")

		# Cluster Attributes -- Used to name clusters
		try: 
			attributes = pd.DataFrame(data=centers, index=counts.index, columns=data_kmeans.columns)
			attributes.to_csv(config['cluster_attributes'])
			logger.info("Cluster attributes, needed for cluster naming, in file")
		except: 
			logger.error("Cluster attributes CSV NOT in file. Check config")

	return labels



def cluster_names(data_kmeans, labels, args):
	'''Add cluster names to the CSV
	Args:
		data_kmeans (data frame): Generated from stadardize_data
		args {argparse.Namespace} : Path to config fil
	''' 
	# Config 
	config = my_configurations(args)
	in_columns = ['Zspotify_track_duration_ms','Zspotify_track_popularity','Zdanceability','Zenergy',
 	'Zloudness','Zspeechiness','Zacousticness','Zinstrumentalness','Zliveness','Zvalence','Ztempo']

 	# Make sure that there are no labels > 13 and all columns are correct
	if any(i > 12 for i in labels) or list(data_kmeans.columns) != in_columns : 
		raise KeyError
	else: 
		# Name the clusters 
		cluster_names = {
	        'Cluster_Num':['0','1','2','3','4','5','6','7','8','9','10','11','12'],
	        'Cluster_Name' : ["Chill Kickback", "Day Drinking","Bro, I'm Sad","Sing Along", "Concert Pregame",
	                 "Who Tryin' to Drink?", "Let's Dance", "Long Talks and Beach Walks", 
	                 "Songs You Forgot About","In My Feels", "Time Machine", 
	                 "Nostalgia", "College Tailgate"]
		}
		cluster_names_df = pd.DataFrame(cluster_names)

		# Add group label to songs 
		group = pd.DataFrame(data_kmeans.index)
		group.insert(1, "Cluster_Num", labels, True) 
		group = group.astype({'Cluster_Num': 'str'})
		labeled_clusters_df = group.merge(cluster_names_df, how = 'left', left_on='Cluster_Num', right_on='Cluster_Num')
		try: 
			labeled_clusters_df.to_csv(config['labeled_clusters'], index=False)
			logger.info("Cluster labels added to data")
		except: 
			logger.error("Cluster labels NOT added to data. Check config")

	return labeled_clusters_df



# Main 
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(asctime)s - %(message)s')
    parser = argparse.ArgumentParser(description="Add config.yml in args")
    parser.add_argument('--config', default='src/config.yml')
    args = parser.parse_args()

    # Load data 
    data = load_data(args)
    # Drop rows 
    data = drop_rows(data)
    # Drop columns not used 
    data_drop_cols = drop_columns(data)
    # Standardize data
    data_kmeans = standardize_data(data_drop_cols)
    #Ideal number of clusters 
    determine_number_clusters(data_kmeans, args)
    # K-means
    clus13 = cluster_solution(data_kmeans, args)
    # Model evaluation 
    labels = model_eval(clus13, data_kmeans, args)
    # Add cluster names to data frame 
    cluster_names(data_kmeans, labels, args)

    







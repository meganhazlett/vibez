import pytest 
import src.models as mod
import pandas as pd 
import numpy as np
import argparse 
import yaml
from sklearn.cluster import KMeans
import sklearn
from numpy import genfromtxt

def test_drop_rows():
	'''Tests the drop_rows function'''
	happy_path = {'SongID':["Winter,StillImage", "Rex,Altars"], 
	'Performer':["StillImage", "Altars"], 'Song':["Winter", "Altars"], 'spotify_genre':["rock", "rock"], 
	'spotify_track_id':["1234","5678"],'spotify_track_preview_url':["http://mylink", np.nan], #no link
    'spotify_track_album':["Winter","Altars"],'spotify_track_explicit':[0.0, 0.0], 
    'spotify_track_duration_ms':[120.0, 140.36],'spotify_track_popularity':[91.0, 89.0], 
    'danceability':[100.0, 200.0], 'energy':[100.0,200.0], 'key':[1.0, 2.0], 'loudness':[-4, -5],
    'mode':[0.0,0.0], 'speechiness':[0.07, 0.08], 'acousticness':[0.017, 0.015], 
    'instrumentalness':[0.00001, 0.2], 'liveness':[0.126, 0.5],'valence':[0.6, 0.7],
    'tempo':[80.0, 20.5], 'time_signature':[4.0, 1.0]
	}
	happy_path_df = pd.DataFrame(happy_path)
	data = happy_path_df

	answers = {'SongID':["Winter,StillImage"], 'Performer':["StillImage"], 
		'Song':["Winter"], 'spotify_genre':["rock"], 
		'spotify_track_id':["1234"],'spotify_track_preview_url':["http://mylink"], #no link
    	'spotify_track_album':["Winter"],'spotify_track_explicit':[0.0], 
    	'spotify_track_duration_ms':[120.0],'spotify_track_popularity':[91.0], 
       'danceability':[100.0], 'energy':[100.0], 
       'key':[1.0], 'loudness':[-4],
       'mode':[0.0], 'speechiness':[0.07], 
       'acousticness':[0.017], 'instrumentalness':[0.00001], 
       'liveness':[0.126],'valence':[0.6],
       'tempo':[80.0], 'time_signature':[4.0]

	}

	output = pd.DataFrame(answers)
	assert isinstance(mod.drop_rows(data), pd.DataFrame)
	assert output.equals(mod.drop_rows(data))


def test_drop_rows_unhappy():
	'''Tests the drop_rows functon for unhappy path'''
	unhappy_path = {'SongID':["Winter,StillImage", "Rex,Altars"], 'Performer':["StillImage", "Altars"], 
		'Song':["Winter", "Altars"], 'spotify_genre':["rock", "rock"], 
		'spotify_track_id':["1234","5678"],'spotify_track_preview_url':["http://mylink", np.nan], #no link
    	'spotify_track_album':["Winter","Altars"],'spotify_track_explicit':[0.0, 0.0], 
    	'spotify_track_duration_ms':[120, 140.36],'spotify_track_popularity':[91, 89], 
       'danceability':[100, 200], 'energy':[100,200], 
       'key':[1.0, 2.0], 'loudness':[-4, -5],
       'mode':[0.0,0.0], 'speechiness':[0.07, 0.08], 
       'acousticness':[0.017, 0.015], 'instrumentalness':[0.00001, 0.2], 
       'liveness':[0.126, 0.5],'valence':[0.6, 0.7],
       'tempo':[80, 20.5], 'time_signature':[4.0, 1.0]
	}
	#Do not make a df 
	try: 
		mod.drop_rows(unhappy_path)
		assert False 
	except ValueError: 
		assert True 


def test_drop_columns():
	'''Tests the drop_columns function'''
	happy_path = {'SongID':["Winter,StillImage", "Rex,Altars"], 
		'Performer':["StillImage", "Altars"], 'Song':["Winter", "Altars"], 'spotify_genre':["rock", "rock"], 
		'spotify_track_id':["1234","5678"],'spotify_track_preview_url':["http://mylink", "http://mylink2"], 
    	'spotify_track_album':["Winter","Altars"],'spotify_track_explicit':[0.0, 0.0], 
    	'spotify_track_duration_ms':[120.0, 140.36],'spotify_track_popularity':[91.0, 89.0], 
    	'danceability':[100.0, 200.0], 'energy':[100.0,200.0], 'key':[1.0, 2.0], 
       'loudness':[-4, -5],'mode':[0.0,0.0], 'speechiness':[0.07, 0.08], 
       'acousticness':[0.017, 0.015], 'instrumentalness':[0.00001, 0.2], 
       'liveness':[0.126, 0.5],'valence':[0.6, 0.7],
       'tempo':[80.0, 20.5], 'time_signature':[4.0, 1.0]
	}

	happy_path_df = pd.DataFrame(happy_path)
	data = happy_path_df

	answers = {'SongID':["Winter,StillImage", "Rex,Altars"], 
    	'spotify_track_duration_ms':[120.0, 140.36],'spotify_track_popularity':[91.0, 89.0], 
    	'danceability':[100.0, 200.0], 'energy':[100.0,200.0], 
       'loudness':[-4, -5],'mode':[0.0,0.0], 'speechiness':[0.07, 0.08], 
       'acousticness':[0.017, 0.015], 'instrumentalness':[0.00001, 0.2], 
       'liveness':[0.126, 0.5],'valence':[0.6, 0.7],
       'tempo':[80.0, 20.5]
	}

	output = pd.DataFrame(answers)
	output = output.set_index('SongID')
	assert isinstance(mod.drop_columns(data), pd.DataFrame)
	assert output.equals(mod.drop_columns(data))


def test_drop_columns_unhappy(): 
	'''Tests drop_columns for unhappy path'''
	unhappy_path = {'SongID':["Winter,StillImage", "Rex,Altars"], 
		'Performer':["StillImage", "Altars"], 'Song':["Winter", "Altars"], 'spotify_genre':["rock", "rock"], 
		'spotify_track_id':["1234","5678"],'spotify_track_preview_url':["http://mylink", "http://mylink2"], 
    	'spotify_track_album':["Winter","Altars"],'spotify_track_explicit':[0.0, 0.0], 
    	'spotify_track_duration_ms':[120.0, 140.36],'spotify_track_popularity':[91.0, 89.0], 
    	'danceability':[100.0, 200.0], 'energy':[100.0,200.0], 'key':[1.0, 2.0], 
       'loudness':[-4, -5],'mode':[0.0,0.0], 'speechiness':[0.07, 0.08], 
       'acousticness':[0.017, 0.015], 'instrumentalness':[0.00001, 0.2], 
       'liveness':[0.126, 0.5],'valence':[0.6, 0.7],
       'tempo':[80.0, 20.5], 'time_signature':[4.0, 1.0]
	}
	# Do not make a df 
	try: 
		mod.drop_columns(unhappy_path)
		assert False 
	except ValueError: 
		assert True 


def test_standardize_data():
	'''Tests standardixe_data function'''
	happy_path = {'SongID':["Winter,StillImage", "Rex,Altars"], 
    	'spotify_track_duration_ms':[120.0, 140.36],'spotify_track_popularity':[91.0, 89.0], 
    	'danceability':[100.0, 200.0], 'energy':[100.0,200.0], 
       'loudness':[-4, -5],'mode':[0.0,0.0], 'speechiness':[0.07, 0.08], 
       'acousticness':[0.017, 0.015], 'instrumentalness':[0.00001, 0.2], 
       'liveness':[0.126, 0.5],'valence':[0.6, 0.7],
       'tempo':[80.0, 20.5]
	}
	happy_path_df = pd.DataFrame(happy_path)
	data_drop_cols = happy_path_df

	answers={'Zspotify_track_duration_ms':[-1.0,1.0], 
	'Zspotify_track_popularity':[1.0,-1.0],'Zdanceability':[-1.0,1.0], 
	'Zenergy':[-1.0, 1.0],'Zloudness':[1.0,-1.0], 'Zspeechiness':[-1.0, 1.0],'Zacousticness':[1.0,-1.0], 
	'Zinstrumentalness':[-1.0,1.0], 'Zliveness':[-1.0,1.0], 'Zvalence':[-1.0,1.0],'Ztempo':[1.0,-1.0]
	}

	output = pd.DataFrame(answers)
	output = output.round(4)
	assert isinstance(mod.standardize_data(data_drop_cols), pd.DataFrame)
	assert output.equals(mod.standardize_data(data_drop_cols))


def test_standardize_data_unhappy():
	'''Tests standardize_data for unhappy path'''
	# Replace some doubles with strings
	unhappy_path = {'SongID':["Winter,StillImage", "Rex,Altars"], 
    	'spotify_track_duration_ms':['120.0', 140.36],'spotify_track_popularity':[91.0, 89.0], 
    	'danceability':[100.0, 200.0], 'energy':[100.0,200.0], 
       'loudness':[-4, -5],'mode':['0.0',0.0], 'speechiness':[0.07, 0.08], 
       'acousticness':[0.017, 0.015], 'instrumentalness':['0.00001', 0.2], 
       'liveness':[0.126, 0.5],'valence':[0.6, 0.7],
       'tempo':[80.0, 20.5]
	}	
	unhappy_path_df = pd.DataFrame(unhappy_path)

	try: 
		mod.standardize_data(unhappy_path)
		assert False 
	except ValueError: 
		assert True 

def test_determine_number_clusters():
	# does not work 
	'''Tests determine_number_clusters function'''
	parser = argparse.ArgumentParser(description="Add config.yml in args")
	parser.add_argument('--config', default='src/test_config.yml')
	args = parser.parse_args()
	with open(args.config, "r") as f:
			config = yaml.load(f)
	config_test = config['models']

	happy_path_df = pd.read_csv(config_test['zscore_sample'])
	happy_path_df = happy_path_df.set_index("SongID")
	data_kmeans = happy_path_df

	output = None

	# assert isinstance(mod.determine_number_clusters(data_kmeans,args))
	assert output == (mod.determine_number_clusters(data_kmeans, args))


def test_determine_number_clusters_unhappy():
	'''Tests determine_number_clusters function with unhappy path'''
	parser = argparse.ArgumentParser(description="Add config.yml in args")
	parser.add_argument('--config', default='src/test_config.yml')
	args = parser.parse_args()
	with open(args.config, "r") as f:
			config = yaml.load(f)
	config_test = config['models']

	unhappy_path_df = pd.read_csv(config_test['zscore_sample_unhappy'])

	try: 
		mod.determine_number_clusters(unhappy_path_df,args)
		assert False 
	except KeyError: 
		assert True 


def test_cluster_solution(): 
	'''Tests cluster_solution function'''
	parser = argparse.ArgumentParser(description="Add config.yml in args")
	parser.add_argument('--config', default='src/test_config.yml')
	args = parser.parse_args()
	with open(args.config, "r") as f:
			config = yaml.load(f)
	config_test = config['models']

	happy_path_df = pd.read_csv(config_test['zscore_sample'])
	happy_path_df = happy_path_df.set_index("SongID")
	data_kmeans = happy_path_df

	output = KMeans(n_clusters=13, init = 'k-means++', max_iter=300, n_init=100, random_state= 0)

	assert isinstance(mod.cluster_solution(data_kmeans, args), sklearn.cluster.k_means_.KMeans)
	# assert output == (mod.cluster_solution(data_kmeans, args))


def test_cluster_solution_unhappy(): 
	'''Tests cluster_solution for unhappy path'''
	parser = argparse.ArgumentParser(description="Add config.yml in args")
	parser.add_argument('--config', default='src/test_config.yml')
	args = parser.parse_args()
	with open(args.config, "r") as f:
			config = yaml.load(f)
	config_test = config['models']

	unhappy_path_df = pd.read_csv(config_test['zscore_sample_unhappy'])
	
	try: 
		mod.cluster_solution(unhappy_path_df,args)
		assert False 
	except KeyError: 
		assert True 


def test_model_eval():
	'''Tests model_eval function'''
	parser = argparse.ArgumentParser(description="Add config.yml in args")
	parser.add_argument('--config', default='src/test_config.yml')
	args = parser.parse_args()
	with open(args.config, "r") as f:
			config = yaml.load(f)
	config_test = config['models']
	
	happy_path_df = pd.read_csv(config_test['zscore_sample'])
	happy_path_df = happy_path_df.set_index("SongID")
	data_kmeans = happy_path_df
	clus13 = KMeans(n_clusters=13, init = 'k-means++', max_iter=300, n_init=100, random_state= 0)

	# output = pd.read_csv(config_test['labels_test'])
	output = genfromtxt(config_test['labels_test'], delimiter=',')
	output = output.astype(int)

	assert isinstance(mod.model_eval(clus13, data_kmeans, args), np.ndarray)
	assert (output == mod.model_eval(clus13, data_kmeans, args)).all()


def test_model_eval_unhappy(): 
	'''Tests model_eval function for unhappy path'''
	parser = argparse.ArgumentParser(description="Add config.yml in args")
	parser.add_argument('--config', default='src/test_config.yml')
	args = parser.parse_args()
	with open(args.config, "r") as f:
			config = yaml.load(f)
	config_test = config['models']

	unhappy_path_df = pd.read_csv(config_test['zscore_sample_unhappy'])
	clus13 = KMeans(n_clusters=13, init = 'k-means++', max_iter=300, n_init=100, random_state= 0)
	
	try: 
		mod.model_eval(clus13, unhappy_path_df,args)
		assert False 
	except KeyError: 
		assert True 


def test_cluster_names(): 
	'''Test the cluster_names function'''
	parser = argparse.ArgumentParser(description="Add config.yml in args")
	parser.add_argument('--config', default='src/test_config.yml')
	args = parser.parse_args()
	with open(args.config, "r") as f:
			config = yaml.load(f)
	config_test = config['models']

	happy_path_df = pd.read_csv(config_test['zscore_sample'])
	happy_path_df = happy_path_df.set_index("SongID")
	data_kmeans = happy_path_df
	labels = genfromtxt(config_test['labels_test'], delimiter=',')
	labels = labels.astype(int)

	output = pd.read_csv(config_test["labeled_clusters_test"],index_col=0)
	# Output generated by function 
	mod.cluster_names(data_kmeans, labels, args)
	true_output = pd.read_csv(config_test["labeled_clusters"],index_col=0)

	assert isinstance(mod.cluster_names(data_kmeans, labels, args), pd.DataFrame)
	assert output.equals(true_output)


def test_cluster_names_unhappy(): 
	'''Tests the cluster_names function for unhappy path'''
	parser = argparse.ArgumentParser(description="Add config.yml in args")
	parser.add_argument('--config', default='src/test_config.yml')
	args = parser.parse_args()
	with open(args.config, "r") as f:
			config = yaml.load(f)
	config_test = config['models']

	unhappy_path_df = pd.read_csv(config_test['zscore_sample_unhappy'])
	labels_unhappy = genfromtxt(config_test['labels_test_unhappy'], delimiter=',')
	labels_unhappy = labels_unhappy.astype(int)

	try: 
		mod.cluster_names(unhappy_path_df, labels_unhappy,args)
		assert False 
	except KeyError: 
		assert True 
































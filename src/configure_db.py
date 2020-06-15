
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base 
from sqlalchemy import Column, Integer, String, Float, MetaData
import sqlalchemy as sql
import logging
import sys
import pandas as pd
import argparse
import yaml
import sqlite3
import os


Base = declarative_base() 

MYSQL_USER = os.environ.get("MYSQL_USER")
MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD")
MYSQL_HOST = os.environ.get("MYSQL_HOST")
MYSQL_PORT = os.environ.get("MYSQL_PORT")
MYSQL_DB = os.environ.get("MYSQL_DB")


class Hot_100_Labeled_Clusters(Base):
    """Schema for labeled clusters after k-means modeling"""
    __tablename__ = "hot_100_labeled_clusters"

    SongID = Column(String(200), primary_key=True)
    Cluster_Num = Column(Integer, unique=False)
    Cluster_Name = Column(String(200), unique=False)

    def __repr__(self): 
        hot_100_labeled_clusters_repr = "<Hot_100_Labeled_Clusters(SongID='%s', Cluster_Num='%d', Cluster_Name='%s')>"
        return hot_100_labeled_clusters_repr%(self.SongID, self.Cluster_Num, self.Cluster_Name)


def create_connection(rds=False, MYSQL_HOST=MYSQL_HOST, MYSQL_DB=MYSQL_DB, MYSQL_SQLTYPE="mysql+pymysql", MYSQL_PORT=3306,
                      MYSQL_USER=MYSQL_USER, MYSQL_PASSWORD=MYSQL_PASSWORD, SQLITELOCALENGINE="sqlite:///../data/hot100.db"):
    """Creates RDS/SQlite connection.
    Args:
        rds (bool) -- If true, creates databased in RDS. If false, creates locally in sqlite (default: {False})
    """
    print(MYSQL_USER)
    if rds == True:
        engine_string = "{}://{}:{}@{}:{}/{}".format(MYSQL_SQLTYPE, MYSQL_USER, MYSQL_PASSWORD, MYSQL_HOST, MYSQL_PORT, MYSQL_DB)
        print(engine_string)
        logger.info("Attempting to connect to RDS")
    else:
        engine_string = SQLITELOCALENGINE
        logger.info("Attempting to connect to SQLlite local")

    try:
        engine = sql.create_engine(engine_string)
        logger.info("Connection to database created")
    except: 
        logger.warning("Connection to database NOT created")
    return engine


def create_db(rds=False, MYSQL_HOST=MYSQL_HOST, MYSQL_DB=MYSQL_DB, MYSQL_SQLTYPE="mysql+pymysql", MYSQL_PORT=3306,
                      MYSQL_USER=MYSQL_USER, MYSQL_PASSWORD=MYSQL_PASSWORD, SQLITELOCALENGINE="sqlite:///../data/hot100.db"):
    """Creates a database with the data models inherited from `Base` (Tweet and TweetScore).
    Args:
        rds (boolean): Database is local or for RDS
    Returns:
        engine : A created connection to a sql engine 
    """
    try: 
        engine = create_connection(rds, MYSQL_HOST, MYSQL_DB, MYSQL_SQLTYPE, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, SQLITELOCALENGINE)
        Base.metadata.create_all(engine)
        logger.info("Datebase created")
    except: 
        logger.warning("Database NOT created")
    return engine


def get_session(engine):
    """ Creates SqLalchemy session
    Args:
        engine_string --SQLAlchemy connection string in the form of: "{sqltype}://{username}:{password}@{host}:{port}/{database}"
    Returns:
        SQLAlchemy session
    """
    try: 
        Session = sessionmaker(bind=engine)
        session = Session()
        logger.info("Session created")
    except:
        logger.info("Session not created. Please check engine string")
    return session


def persist_Hot_100_Labeled_Clusters(session, records):
    """Adds score records to tweet_score table in SQL database
    
    Args:
        session : SQL database session
        records (pandas data frame): list of dictionaries Hot_100_Audio_Features 
    """
    numrec = 0 #keeps track of number of records added to db
    for index, i in records.iterrows():
        song = Hot_100_Labeled_Clusters(SongID=i['SongID'],Cluster_Num=i["Cluster_Num"], Cluster_Name=i["Cluster_Name"]) 
        session.add(song) #add record
        numrec = numrec + 1 #keeps track of number of records added to db

    logger.info("Added {} records to SQL database".format(numrec))


def configure(args):
    """Runs script to run scoring
    
    Args:
        args {argparse.Namespace} -- Script arguments

    """

    if args.config is not None:
        with open(args.config, "r") as f:
            config = yaml.load(f,Loader=yaml.FullLoader)
        config = config['configure_db']
    else:
        raise ValueError("Path to yaml config file must be provided through --config")
    if args.rds:
        engine = create_db(rds=True, MYSQL_HOST=MYSQL_HOST, MYSQL_DB=MYSQL_DB, MYSQL_SQLTYPE="mysql+pymysql", MYSQL_PORT=3306,
                      MYSQL_USER=MYSQL_USER, MYSQL_PASSWORD=MYSQL_PASSWORD)
        logger.info("We will be creating an rds db")
    else:
        engine = create_db(rds=False, **config['sqlite'])
        logger.info("We will be creating a local db")
    session = get_session(engine)

    
    if args.input_hot100 is not None:
        hot100_data = pd.read_csv(args.input_hot100)
        hot100_data = hot100_data.drop_duplicates(keep='first') 
        logger.info("Data read in")
    else:
        raise ValueError("Path to CSV for input combination data must be provided through --input_combinations")
    
    # persist_Hot_100_Labeled_Clusters(session, hot100_data)
    try:
        persist_Hot_100_Labeled_Clusters(session, hot100_data)
        session.commit()
        logger.info("Persisted all records")
    except:
        logger.error("Records NOT persisted")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(asctime)s - %(message)s')
    logger = logging.getLogger(__file__)
    parser = argparse.ArgumentParser(description="Add config.yml in args")
    parser.add_argument('--config', default='src/config.yml', help='config.yml')
    parser.add_argument('--rds', default=False, help='path to yaml file with configurations')
    parser.add_argument('--input_hot100', default='data/Hot_100_Labeled_Clusters.csv', help='src/config.yml')
    args = parser.parse_args()
    
    MYSQL_USER = os.environ.get("MYSQL_USER")
    MYSQL_PASSWORD = os.environ.get("MYSQL_PASSWORD")
    MYSQL_HOST = os.environ.get("MYSQL_HOST")
    MYSQL_PORT = os.environ.get("MYSQL_PORT")
    MYSQL_DB = os.environ.get("MYSQL_DB")
    configure(args)

import boto3
import os
import sys
import argparse 
import yaml
import logging.config
logger = logging.getLogger(__name__)

def s3_to_local(file_name, bucket, localfilepath, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY):
    '''Brings data from S3 to local file
    Args: 
        file_name (str)
        bucket (str) : name of S3 bucket to put files in
        s3_path (str): path to S3 bucket
        AWS_ACCESS_KEY_ID (str): AWS access key of S3 bucket 
        AWS_SECRET_ACCESS_KEY (str): AWS secret access key of S3 bucket
    '''
    try:
        s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
        logger.info("Correct creds")
    except:
        logger.error('Incorrect AWS credentials')
    try:
        s3.download_file(bucket, file_name, localfilepath)
        logger.info('Successfully landed source data in configured path')
    except:
        logger.error('Incorrect AWS Bucket name or filepath name. Please re-enter it in config.yml')    



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(asctime)s - %(message)s')
    parser = argparse.ArgumentParser(description="Add config.yml in args")
    parser.add_argument('--config', default='src/config.yml')
    args = parser.parse_args()

    if args.config is not None:
    	with open(args.config, "r") as f:
    		config = yaml.load(f)
    	config = config['acquire_data']
    else:
    	raise ValueError("Path to config yml file must be provided through --config")

    # My variables 
    AWS_ACCESS_KEY_ID=os.environ.get("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY=os.environ.get("AWS_SECRET_ACCESS_KEY")

    file_name = config['file_name']
    bucket = config['bucket']
    localfilepath = config['localfilepath']
    s3_to_local(file_name, bucket, localfilepath, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)







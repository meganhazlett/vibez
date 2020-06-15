import boto3
import os
import sys
import argparse 
import yaml
import logging.config
logger = logging.getLogger(__name__)


def source_to_s3(file_name, bucket, s3_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY): 
    '''Uploads a file to S3 bucket 
    Args: 
    	file_name (str)
    	bucket (str) : name of S3 bucket to put files in
    	s3_path (str): path to S3 bucket
        AWS_ACCESS_KEY (str): AWS access key of S3 bucket 
        AWS_SECRET_ACCESS_KEY (str): AWS secret access key of S3 bucket
    '''

    try: 
        s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
        logger.info("Correct creds")
    except: 
        logger.error("Incorrect AWS credentials")

    try:
        s3_client.upload_file(file_name, bucket, s3_path)
        logger.info("Successfully loaded file to S3. Check your online bucket.")
    except:
    	traceback.print_exc()
    	logging.error("File was NOT uploaded to S3")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(asctime)s - %(message)s')
    parser = argparse.ArgumentParser(description="Add config.yml in args")
    parser.add_argument('--config', default='src/config.yml')
    args = parser.parse_args()

    if args.config is not None:
    	with open(args.config, "r") as f:
    		config = yaml.load(f)
    	config = config['data_to_s3']
    else:
    	raise ValueError("Path to config yml file must be provided through --config")

    # My variables 
    file_name=config['file_name']
    bucket=config['bucket']
    s3_path=config['s3_path']
    AWS_ACCESS_KEY_ID=os.environ.get("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY=os.environ.get("AWS_SECRET_ACCESS_KEY")
    source_to_s3(file_name, bucket, s3_path, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)

    
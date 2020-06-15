#!/usr/bin/env bash

# Acquire data
python3 src/acquire_data.py 

# Clean data, build kmeans model, evaluate model 
python3 src/models.py

# # Model testing 
py.test

# # Set up data base 
python3 src/configure_db.py 





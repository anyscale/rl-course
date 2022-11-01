#!/bin/sh

# Sync main exercise notebook
cp exercises/en/exercise.ipynb ../rl-course-exercises

# Copy utils and environments
cp notebooks/02-rllib/utils.py ../rl-course-exercises
cp notebooks/02-rllib/envs.py ../rl-course-exercises

# Copy models and data
cp -r notebooks/02-rllib/models/* ../rl-course-exercises/models/
cp -r notebooks/03-designing-environments/models/* ../rl-course-exercises/models/
cp -r notebooks/04-application-recommender/data/* ../rl-course-exercises/data/
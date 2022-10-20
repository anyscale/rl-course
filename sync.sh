#!/bin/sh

# Sync main exercise notebook
cp exercises/en/exercise.ipynb ../rl-course-exercises

cp notebooks/01-rl-components/utils_01.py ../rl-course-exercises

cp notebooks/02-rllib/utils_02.py ../rl-course-exercises
cp notebooks/02-rllib/envs_02.py ../rl-course-exercises
cp -r notebooks/02-rllib/models/* ../rl-course-exercises/models/

cp notebooks/03-designing-environments/utils_03.py ../rl-course-exercises
cp notebooks/03-designing-environments/envs_03.py ../rl-course-exercises
cp -r notebooks/03-designing-environments/models/* ../rl-course-exercises/models/

cp notebooks/04-application-recommender/utils_04.py ../rl-course-exercises
cp notebooks/04-application-recommender/envs_04.py ../rl-course-exercises
cp -r notebooks/04-application-recommender/data/* ../rl-course-exercises/data/

cp notebooks/05-under-the-hood/utils_05.py ../rl-course-exercises
cp notebooks/05-under-the-hood/envs_05.py ../rl-course-exercises
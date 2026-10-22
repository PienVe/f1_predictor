#!/bin/bash

python 02_prepare_dataset.py
python 03_train_model.py
python 04_predict_raceresults.py
python 05_create_upcoming_race.py
python 06_upcoming_raceresults.py
python 07_visualise_prediction.py
python 08_position_confidence.py
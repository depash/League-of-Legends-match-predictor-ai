import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models

df = pd.read_csv("backend/data/processed/match_training_data.csv")

x = df.drop(columns=['teamA_matchid',"teamA_teamid","teamA_id", "teamA_item1","teamA_item2","teamA_item3","teamA_item4","teamA_item5","teamA_item6", "teamA_trinket", "teamA_championid",
                     "teamB_matchid","teamB_teamid","teamB_id", "teamB_item1","teamB_item2","teamB_item3","teamB_item4","teamB_item5","teamB_item6", "teamB_trinket", "teamB_championid",
                     "matchid"])
y = df['teamA_win']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
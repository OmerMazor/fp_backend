from flask import Flask, jsonify, request
from flask_cors import CORS
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from selenium.webdriver.firefox.options import Options as FirefoxOptions
import pandas as pd
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import cross_val_predict
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.pipeline import Pipeline
from joblib import load
from get_data_from_site import teams_data


app = Flask(__name__)
CORS(app)  # מאפשר שליחת בקשות מ-React

# loaded = load("model_with_accuracy.joblib")
pipeline = load("pipeline.joblib")
mean_accuracy = load("accuracy.joblib")
# pipeline = loaded["model"]
# mean_accuracy = loaded["accuracy"]
results_df = pd.read_csv("results.csv")



@app.route("/api/accuracy", methods=["GET"])
def get_accuracy():
    return jsonify({"accuracy": round(mean_accuracy * 100, 2)})

@app.route("/api/teams", methods=["POST"])
def receive_teams():
    data = request.json
    home_team = data.get("homeTeam", "")
    away_team = data.get("awayTeam", "")
    home_market_value = data.get("homeMarketValue", "")
    away_market_value = data.get("awayMarketValue", "")

    home_wins = ""
    draws = ""
    away_wins = ""
    home_position = ""
    home_goals = ""
    home_goals_against = ""
    home_goalkeeping = ""
    home_shot_on_target = ""
    home_accuracy = ""
    away_position = ""
    away_goals = ""
    away_goals_against = ""
    away_goalkeeping = ""
    away_shot_on_target = ""

    home_wins, draws, away_wins, home_position, home_goals, home_goals_against, \
    home_goalkeeping, home_shot_on_target, home_accuracy, away_position, \
    away_goals, away_goals_against, away_goalkeeping, away_shot_on_target  = teams_data(home_team, away_team, home_market_value, away_market_value)

    try: 
        column_names = [
        'History_Team1_Win', 'History__Draw', 'History_Team2_Win', 'Points_Per_Game_Team1', 'Team1_Goals/Game', 'Team1_Absorbs/Game',
        'Team1_Goalkeaping', 'Team1Shot_on_Target/_Game', 'Team1_Accuracy',
        'Points_Per_Game_Team2', 'Team2_Goals/Game', 'Team2_Absorbs/Game', 'Team2_Goalkeaping',
        'Team2_Shot_on_Target/_Game', 'Team1_Market_Value', 'Team2_Market_Value'
        ]
        data = [[home_wins, draws, away_wins, home_position, home_goals, home_goals_against,
            home_goalkeeping, home_shot_on_target, home_accuracy,
            away_position, away_goals, away_goals_against, away_goalkeeping,
            away_shot_on_target, home_market_value, away_market_value]]
        print(data)
        new_df = pd.DataFrame(data, columns=column_names)
        prediction = pipeline.predict(new_df)[0]
        print("Predicted result:", prediction)
        if prediction == 1:
            result = home_team
        elif prediction == 2:
            result = away_team
        elif prediction == 3:
            result = "Draw"
    except:
        result = "Not enough data"

    response = {
        "result": f"Result prediction: {result}"
    }
    return jsonify(response)
    
@app.route("/api/predictions", methods=["GET"])
def get_predictions():
    return jsonify(results_df.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)

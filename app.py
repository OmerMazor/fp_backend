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
CORS(app) 

# loaded = load("model_with_accuracy.joblib")
# pipeline = load("pipeline.joblib")
# mean_accuracy = load("accuracy.joblib")
# pipeline = loaded["model"]
# mean_accuracy = loaded["accuracy"]
results_df = pd.read_csv("results.csv")


@app.route("/")
def home():
    return "✅ Flask Server is Running"

@app.route("/api/models", methods=["GET"])
def get_models():
    model_files = [f for f in os.listdir() if f.startswith("pipeline_") and f.endswith(".joblib")]
    models = []
    for file in model_files:
        name = file.replace("pipeline_", "").replace(".joblib", "")
        try:
            acc = load(f"accuracy_{name}.joblib")
            models.append({"name": name, "accuracy": round(acc * 100, 2)})
        except:
            continue
    return jsonify({"models": models})


@app.route("/api/teams", methods=["POST"])
def receive_teams():
    data = request.json
    home_team = data.get("homeTeam", "")
    away_team = data.get("awayTeam", "")
    home_market_value = data.get("homeMarketValue", "")
    away_market_value = data.get("awayMarketValue", "")
    model_name = data.get("modelName", "All")
    print(model_name)
    home_wins = ""
    draws = ""
    away_wins = ""
    home_position = ""
    home_goals = ""
    home_goals_against = ""
    home_goalkeeping = ""
    home_red_cards = ""
    home_shot_on_target = ""
    home_accuracy = ""
    away_position = ""
    away_goals = ""
    away_goals_against = ""
    away_goalkeeping = ""
    away_shot_on_target = ""
    away_accuracy = ""

    home_wins, draws, away_wins, home_position, home_goals, home_goals_against, \
    home_accuracy, home_goalkeeping, home_red_cards, home_shot_on_target, away_position, \
    away_goals, away_goals_against, away_shot_on_target, away_accuracy, away_goalkeeping, home_games, away_games  = teams_data(home_team, away_team, home_market_value, away_market_value)

    try: 
        history_total_games = str(int(home_wins) + int(draws) + int(away_wins))
        history_home_perc = str(int(home_wins) / int(history_total_games))
        history_draw_perc = str(int(draws) / int(history_total_games))
        history_away_perc = str(int(away_wins) / int(history_total_games))
        data = None
        if model_name == "All":
            column_names = [
            'Team1_Games', 'Points_Per_Game_Team1', 'Team1_Goals/Game',
            'Team1_Absorbs/Game', 'Team1Shot_on_Target/_Game', 'Team2_Games',
            'Points_Per_Game_Team2', 'Team2_Goals/Game', 'Team2_Absorbs/Game',
            'Team2_Shot_on_Target/_Game', 'Team2_Market_Value',
            'History_Total_Games', 'History_Team1_Win_Perc', 'History_Draw_Perc', 
            'History_Team2_Win_Perc'
            ]
            data = [[str(home_games), home_position, home_goals, 
                home_goals_against, home_shot_on_target, away_games,
                away_position, away_goals, away_goals_against,  
                away_shot_on_target, away_market_value,
                history_total_games, history_home_perc, history_draw_perc, 
                history_away_perc]]
        elif model_name == "America":
            column_names = [
            'Points_Per_Game_Team1', 'Team1_Goals/Game', 'Team1_Absorbs/Game',
            'Team1_Goalkeaping', 'Points_Per_Game_Team2', 'Team2_Goals/Game', 
            'Team2_Absorbs/Game', 'Team2_Accuracy', 'History_Total_Games', 
            'History_Team1_Win_Perc', 'History_Draw_Perc', 
            'History_Team2_Win_Perc'
            ]
            data = [[home_position, home_goals, home_goals_against,
                home_goalkeeping, away_position, away_goals, 
                away_goals_against, away_accuracy, history_total_games, 
                history_home_perc, history_draw_perc, 
                history_away_perc]]
        elif model_name == "Asia":
            column_names = [
            'Points_Per_Game_Team1', 'Team1_Goals/Game', 'Team1_Absorbs/Game',
            'Team1_Goalkeaping', 'Points_Per_Game_Team2', 'Team2_Goals/Game', 
            'Team2_Absorbs/Game', 'Team1_Market_Value', 'Team2_Market_Value', 
            'History_Team1_Win_Perc', 'History_Draw_Perc', 
            'History_Team2_Win_Perc'
            ]
            data = [[home_position, home_goals, home_goals_against,
                home_goalkeeping, away_position, away_goals, 
                away_goals_against, home_market_value, away_market_value, 
                history_home_perc, history_draw_perc, 
                history_away_perc]]
        elif model_name == "Europe":
            column_names = [
            'Points_Per_Game_Team1', 'Team1_Goals/Game', 'Team1_Absorbs/Game',
            'Team1Shot_on_Target/_Game', 'Points_Per_Game_Team2', 
            'Team2_Goals/Game', 'Team2_Absorbs/Game', 'Team2_Shot_on_Target/_Game',
            'History_Total_Games', 'History_Team1_Win_Perc', 'History_Draw_Perc', 
            'History_Team2_Win_Perc'
            ]
            data = [[home_position, home_goals, home_goals_against,
                home_shot_on_target, away_position, 
                away_goals, away_goals_against, away_shot_on_target, 
                history_total_games, history_home_perc, history_draw_perc, 
                history_away_perc]]
        elif model_name == "Men":
            column_names = [
            'Points_Per_Game_Team1', 'Team1_Goals/Game', 'Team1_Absorbs/Game',
            'Team1_Goalkeaping', 'Team1Shot_on_Target/_Game', 'Team1_Accuracy', 
            'Points_Per_Game_Team2', 'Team2_Goals/Game', 'Team2_Absorbs/Game',
            'Team2_Goalkeaping', 'Team2_Shot_on_Target/_Game', 'Team2_Accuracy', 
            'Team1_Market_Value', 'Team2_Market_Value', 'History_Total_Games', 
            'History_Team1_Win_Perc', 'History_Draw_Perc', 
            'History_Team2_Win_Perc'
            ]
            data = [[home_position, home_goals, home_goals_against,
                home_goalkeeping, home_shot_on_target, home_accuracy, 
                away_position, away_goals, away_goals_against, 
                away_goalkeeping, away_shot_on_target, away_accuracy, 
                home_market_value, away_market_value, history_total_games, 
                history_home_perc, history_draw_perc, 
                history_away_perc]]
        elif model_name == "Women_Only":
            column_names = [
            'Points_Per_Game_Team1', 'Team1_Absorbs/Game', 'Team1_Goalkeaping', 
            'Team1_Red', 'Points_Per_Game_Team2', 'Team2_Goals/Game', 
            'Team2_Shot_on_Target/_Game', 'History_Team1_Win_Perc', 
            'History_Draw_Perc', 'History_Team2_Win_Perc'
            ]
            data = [[home_position, home_goals_against, home_goalkeeping, 
                home_red_cards, away_position, away_goals, 
                away_shot_on_target, history_home_perc, history_draw_perc, 
                history_away_perc]]
            
        print(data)
        new_df = pd.DataFrame(data, columns=column_names)
        prediction = 0
        # try:
        pipeline = load(f"pipeline_{model_name}.joblib")
        print("📋 new_df.columns:", list(new_df.columns))
        print('\n')
        print("📋 expected columns:", pipeline.feature_names_in_)
        print(list(new_df.columns) == list(pipeline.feature_names_in_))
        prediction = pipeline.predict(new_df)[0]

        # except:
        #     return jsonify({"error": "Model not found"}), 400
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

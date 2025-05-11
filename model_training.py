import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from joblib import dump

model = None
scaler = None
results_df = None
mean_accuracy = None
pipeline = None
def label_result(result, teams):
    team1, team2 = teams.split("-vs-")
    if result == 1:
        return team1
    elif result == 2:
        return team2
    elif result == 3:
        return "Draw"

def train_model():
    global pipeline 
    global results_df
    global mean_accuracy
    train_file_path = "./train_new.xlsx"
    df = pd.read_excel(train_file_path)
    print(f"Original data shape: {df.shape}")
    df.columns = df.columns.str.strip()  
    df.columns = df.columns.str.replace("\n", " ", regex=True) 
    df.columns = df.columns.str.replace(" ", "_")  
    df = df.loc[~df.drop(columns=['Date']).isnull().all(axis=1)]
    df.drop_duplicates(inplace=True)
    print(df.isnull().sum() / len(df) * 100)
    df = df[(df["Team1_Games"] >= 1) & (df["Team2_Games"] >= 1)]

    def convert_market_value(value):
        if isinstance(value, str):  
            value = value.replace("€", "").strip() 
            multiplier = 1  
            if value.endswith("m"):  
                multiplier = 1_000_000
                value = value.replace("m", "")
            elif value.endswith("k"): 
                multiplier = 1_000
                value = value.replace("k", "")
            elif value.endswith("bn"):  
                multiplier = 1_000_000_000
                value = value.replace("bn", "")
            try:
                return float(value) * multiplier
            except ValueError:
                return value 
        return value

    df["Team1_Market_Value"] = df["Team1_Market_Value"].apply(convert_market_value)
    df["Team2_Market_Value"] = df["Team2_Market_Value"].apply(convert_market_value)
    df = df.drop(['Team1_Hosting', 'Team2_Hosting'], axis=1)
    df = df.drop( ['Team1_Defence', 'Team1_Corners', 'Team2_Defence', 'Team2_Corners'] , axis=1)
    df = df.drop(['Team1_Games', 'Team2_Games'], axis=1)
    df = df.drop( ['Team2_Red', 'Team1_Red', 'Team1_Yellow', 'Team2_Yellow', 'Team2_Accuracy'] , axis=1)
    df = df[df["Result"] != "NA"]





    print(df.columns)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    print(df.isnull().sum() / len(df) * 100)
    print(df.shape)
    df = df[df['Result'].isin([1, 2, 3])]
    df['Result'] = df['Result'].astype(int) 
    target_column = "Result"
    dates = df['Date']
    teams = df['Teams']
    df = df.drop( ['Date', 'Teams'] , axis=1)
    X = df.drop(columns=[target_column])
    # X.dropna(inplace=True)
    X.replace('-', np.nan, inplace=True)
    # y = df[target_column] 
    X = X.apply(pd.to_numeric, errors='coerce')
    valid_rows = X.notna().all(axis=1)
    X = X[valid_rows]
    y = df.loc[valid_rows, target_column]




    scaler = MinMaxScaler()
    baseline = y.value_counts(normalize=True).max() * 100
    print(f"Baseline Accuracy: {baseline:.2f}%")


    stacking_model = StackingClassifier(
    estimators=[
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, min_samples_split=10, max_features="sqrt", random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.05)),
        ('svm', SVC(probability=True))
    ],
    final_estimator=LogisticRegression()  # המודל הסופי שמחליט איך לשלב את התוצאות
    )

    correlations = df.corr(numeric_only=True)
    result_corr = correlations["Result"].sort_values(ascending=False)
    print(result_corr)

    dates = dates.loc[X.index]
    teams = teams.loc[X.index]
    pipeline = Pipeline([
    ("scaler", MinMaxScaler()),
    ("model", stacking_model)
    ])
    scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"🔹 Cross-Validation Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    mean_accuracy = np.mean(scores)
    pipeline.fit(X, y)
    ## dump({"model": pipeline, "accuracy": mean_accuracy}, "model_with_accuracy.joblib")
    dump(pipeline, "pipeline.joblib")
    dump(mean_accuracy, "accuracy.joblib")
    predictions = pipeline.predict(X)
    team_labels = teams.loc[X.index].values



    results_df = pd.DataFrame({
        'Date': dates.values,
        'Teams': teams.values,
        'Prediction': predictions,
        'Actual': y,
        'Correct': predictions == y
    })

    results_df["Predicted_Label"] = results_df.apply(lambda row: label_result(row["Prediction"], row["Teams"]), axis=1)
    results_df["Actual_Label"] = results_df.apply(lambda row: label_result(row["Actual"], row["Teams"]), axis=1)
    results_df["Correct"] = results_df["Predicted_Label"] == results_df["Actual_Label"]
    results_df.to_csv("results.csv", index=False)


    rf_model = stacking_model.named_estimators_['rf']
    importances = rf_model.feature_importances_
    importance_df = (
    pd.DataFrame({"Feature": X.columns, "Importance": importances})
    .sort_values(by="Importance", ascending=False)
    )
    print(importance_df.head(30))
    return stacking_model, scaler


train_model()
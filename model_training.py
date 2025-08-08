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
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import leagues
from scipy.stats import zscore
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV


def load_and_prepare_data():
    path = "./train_new.xlsx"
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip().str.replace("\n", " ", regex=True).str.replace(" ", "_")
    df = df.loc[~df.drop(columns=['Date']).isnull().all(axis=1)]
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
    df = df.dropna(subset=['Team1_Market_Value', 'Team2_Market_Value'], how='any')
    df = df.drop(['Team1_Defence', 'Team1_Corners', 'Team2_Defence', 'Team2_Corners'], axis=1)
    # df = df.drop(['Team2_Red', 'Team2_Accuracy', 'Team2_Yellow',
    #             'Team1_Red', 'Team1_Accuracy', 'Team1_Yellow',
    #             'Team2_Goalkeaping', 'Team1_Goalkeaping', 'Team1_Market_Value'], axis=1)
    df = df[df["Result"] != "NA"]
    

    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.drop_duplicates(inplace=True)
    df['History_Total_Games'] = df['History_Team1_Win'] + df['History__Draw'] + df['History_Team2_Win']
    df['History_Team1_Win_Perc'] = df['History_Team1_Win'] / df['History_Total_Games']
    df['History_Draw_Perc'] = df['History__Draw'] / df['History_Total_Games']
    df['History_Team2_Win_Perc'] = df['History_Team2_Win'] / df['History_Total_Games']
    df = df.drop(['History_Team1_Win', 'History__Draw', 'History_Team2_Win'], axis=1)
    return df

def filter_dataset(df, combined_remove, keep_mask=True):
    team_pairs = df['Teams'].str.split('-vs-', expand=True)
    team1, team2 = team_pairs[0], team_pairs[1]
    mask = team1.isin(combined_remove) | team2.isin(combined_remove)
    return df[mask] if keep_mask else df[~mask]

def get_models():
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_leaf=5,
        min_samples_split=10, max_features="sqrt", random_state=42
    )
    svm_model = SVC(probability=True)
    nn_model = MLPClassifier(
        solver='adam', learning_rate='adaptive', hidden_layer_sizes=(100, 50),
        alpha=0.01, activation='tanh', max_iter=1000, random_state=42
    )
    stacking_model = StackingClassifier(
        estimators=[
            ('rf', rf_model),
            ('gb', GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.05))
        ],
        final_estimator=LogisticRegression()
    )
    return {
        # "Stacking": stacking_model,
        "Random Forest": rf_model,
        # "SVM": svm_model,
        # "Neural Network": nn_model
    }


def train_and_evaluate_models(X, y, dataset_name):
    models = get_models()
    best_score = 0.0
    best_pipeline = None
    best_model_name = ""
    print(f"\n Evaluation on dataset: {dataset_name}")
    for name, model in models.items():
        pipeline = Pipeline([
            ("scaler", MinMaxScaler()),
            ("model", model)
        ])
        scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
        mean_score = np.mean(scores)
        print(f"{name}: {scores.mean():.4f} ± {scores.std():.4f}")
        if name == 'Random Forest':
                mean_accuracy = np.mean(scores)
                pipeline.fit(X, y)
                best_model = pipeline.named_steps["model"]
                importances = best_model.feature_importances_
                importance_df = (
                pd.DataFrame({"Feature": X.columns, "Importance": importances})
                .sort_values(by="Importance", ascending=False)
                )
                print(importance_df.head(30))

        if mean_score > best_score:
            best_score = mean_score
            best_pipeline = pipeline
            best_model_name = name

            best_y_true = []
            best_y_pred = []
            skf = StratifiedKFold(n_splits=5)
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)

                best_y_true.extend(y_test)
                best_y_pred.extend(y_pred)

    if best_pipeline:
        best_pipeline.fit(X, y)
        # dump(best_pipeline, f"pipeline_{dataset_name}.joblib")
        # dump(best_score, f"accuracy_{dataset_name}.joblib")
        # print(f"✅ Saved best model '{best_model_name}' for '{dataset_name}' with accuracy {best_score:.4f}")
        # print(f"\n📄 Classification Report for best model ({best_model_name}):\n")
        # print(classification_report(best_y_true, best_y_pred))
    return best_pipeline

def balance_dataset(X, y, sample_size):
    df_combined = pd.concat([X, y], axis=1)
    balanced_df = pd.concat([
        df_combined.query("Result == 1").sample(sample_size, random_state=42),
        df_combined.query("Result == 2").sample(sample_size, random_state=42),
        df_combined.query("Result == 3").sample(sample_size, random_state=42)
    ])
    X_balanced = balanced_df.drop(columns=["Result"])
    y_balanced = balanced_df["Result"]
    return X_balanced, y_balanced

df_full = load_and_prepare_data()

female_leagues = leagues.bene_league_teams + leagues.we_league_teams + leagues.liga_f_teams + leagues.frauen_bundesliga_teams + leagues.serie_a_women_teams + leagues.eredivisie_women_teams + leagues.primera_federacion_femenina_teams + leagues.damallsvenskan_teams + leagues.swiss_womens_super_league_teams + leagues.premiere_ligue_teams + leagues.a_league_women + leagues.toppserien_teams + leagues.oefb_frauen_bundesliga_teams
df_female = filter_dataset(df_full, female_leagues, keep_mask=True)

male_leagues = leagues.bene_league_teams + leagues.we_league_teams + leagues.liga_f_teams + leagues.frauen_bundesliga_teams + leagues.serie_a_women_teams + leagues.eredivisie_women_teams + leagues.primera_federacion_femenina_teams + leagues.damallsvenskan_teams + leagues.swiss_womens_super_league_teams + leagues.premiere_ligue_teams + leagues.a_league_women + leagues.toppserien_teams + leagues.oefb_frauen_bundesliga_teams
df_male = filter_dataset(df_full, male_leagues, keep_mask=False)

europe_leagues = leagues.la_liga_teams + leagues.russian_premier_league_teams + leagues.czech_first_league_teams + leagues.efl_league_one_teams + leagues.liga_portugal_teams + leagues.serie_a_italy_teams + leagues.swiss_super_league_teams + leagues.liga_1_romania_teams + leagues.scottish_premiership_teams + leagues.danish_superliga_teams + leagues.scottish_championship_teams + leagues.nemzeti_bajnoksag_i_teams + leagues.eerste_divisie_teams + leagues.ligue_2_teams + leagues.swiss_promotion_league_teams + leagues.serbian_superliga_teams + leagues.national_league_teams + leagues.efl_championship_teams + leagues.super_lig_teams + leagues.ligue_1_teams + leagues.ukrainian_premier_league_teams + leagues.serie_b_teams + leagues.bundesliga_teams + leagues.bundesliga_teams + leagues.german_3_liga_teams + leagues.ekstraklasa_teams + leagues.super_league_greece_teams + leagues.premier_league_teams + leagues.efl_league_two_teams + leagues.league_of_ireland_premier_division_teams + leagues.bundesliga_2_teams + leagues.eredivisie_teams + leagues.croatian_football_league_teams + leagues.austrian_bundesliga_teams + leagues.segunda_division_teams + leagues.bulgarian_first_league_teams + leagues.challenger_pro_league_teams + leagues.belgian_pro_league_teams + leagues.cypriot_first_division_teams + leagues.bulgarian_second_league_teams + leagues.eliteserien_teams + leagues.veikkausliiga_teams + leagues.superettan_teams + leagues.premijer_liga_bih + leagues.allsvenskan + leagues.slovenian_prvaliga + leagues.nemzeti_bajnoksag_ii + leagues.championnat_national
df_europe = filter_dataset(df_full, europe_leagues, keep_mask=True)

asia_leagues = leagues.saudi_pro_league_teams + leagues.chinese_super_league_teams + leagues.i_league_2_teams + leagues.indian_super_league_teams + leagues.j1_league_teams + leagues.j2_league_teams + leagues.k_league_1_teams + leagues.iran_pro_league_teams + leagues.azadegan_league_teams + leagues.i_league_teams
df_asia = filter_dataset(df_full, asia_leagues, keep_mask=True)

america_leagues = leagues.liga_1_peru_teams + leagues.uruguayan_primera_division_teams + leagues.argentine_primera_division_teams + leagues.ligapro_ecuador_teams + leagues.usl_league_one_teams + leagues.paraguayan_primera_division_teams + leagues.chilean_primera_division_teams + leagues.liga_futve_teams + leagues.usl_championship_teams + leagues.bolivian_primera_division_teams + leagues.liga_mx_teams + leagues.mls_teams + leagues.categoria_primera_a_teams + leagues.brazil_serie_a_teams + leagues.nwsl_teams + leagues.serie_b_brazil_teams + leagues.primera_b_chile_teams + leagues.canadian_premier_league + leagues.mls_next_pro_teams + leagues.npsl_teams
df_america = filter_dataset(df_full, america_leagues, keep_mask=True)


datasets = {
    "All": df_full,
    "Women_Only": df_female,
    "Men": df_male,
    "Europe": df_europe,
    "Asia": df_asia,
    "America": df_america

}

sample_sizes = {
    "All": 1000,
    "Women_Only": 30,
    "Men": 1000,
    "Europe": 800,
    "Asia": 100,
    "America": 300
}

features_to_drop = {
    "All": ['Team2_Red', 'Team2_Accuracy', 'Team2_Yellow','Team1_Red', 'Team1_Accuracy', 'Team1_Yellow','Team2_Goalkeaping', 'Team1_Goalkeaping', 'Team1_Market_Value'],
    "Women_Only": ['Team1_Games', 'Team2_Goalkeaping', 'Team1_Accuracy', 'Team2_Accuracy', 'Team2_Games', 'Team2_Yellow', 'Team2_Red', 'Team2_Market_Value', 'Team1_Yellow', 'Team1Shot_on_Target/_Game', 'Team1_Goals/Game', 'History_Total_Games', 'Team1_Market_Value', 'Team2_Absorbs/Game'],
    "Men": ['Team1_Red', 'Team1_Games', 'Team2_Red', 'Team1_Yellow', 'Team2_Games', 'Team2_Yellow'],
    "Europe": ['Team2_Accuracy', 'Team1_Red', 'Team1_Yellow', 'Team2_Red', 'Team2_Yellow', 'Team1_Games', 'Team1_Accuracy', 'Team2_Games', 'Team2_Market_Value', 'Team2_Goalkeaping', 'Team1_Market_Value', 'Team1_Goalkeaping'],
    "Asia": ['Team2_Red', 'Team1_Red', 'Team2_Goalkeaping', 'Team1_Yellow', 'Team1_Accuracy', 'Team2_Games', 'Team2_Accuracy', 'Team1_Games', 'Team1Shot_on_Target/_Game', 'History_Total_Games', 'Team2_Yellow', 'Team2_Shot_on_Target/_Game'],
    "America": ['Team2_Red', 'Team1_Yellow', 'Team1_Red', 'Team2_Yellow', 'Team1Shot_on_Target/_Game', 'Team1_Games', 'Team2_Goalkeaping', 'Team2_Market_Value', 'Team2_Games', 'Team2_Shot_on_Target/_Game', 'Team1_Accuracy', 'Team1_Market_Value']
}

for name, df in datasets.items():
    to_drop = [col for col in features_to_drop[name] if col in df.columns]
    df = df.drop(columns=to_drop)
    df = df[df['Result'].isin([1, 2, 3])]
    df['Result'] = df['Result'].astype(int)
    dates = df['Date']
    teams = df['Teams']
    df = df.drop(['Date', 'Teams'], axis=1)
    X = df.drop(columns=["Result"])
    # print(X.columns)
    X.replace('-', np.nan, inplace=True)
    X = X.apply(pd.to_numeric, errors='coerce')
    valid_rows = X.notna().all(axis=1)
    X = X[valid_rows]
    y = df.loc[valid_rows, "Result"]

    z_scores = np.abs(zscore(X))
    non_outliers_mask = (z_scores < 7).all(axis=1)
    X = X[non_outliers_mask]
    y = y[non_outliers_mask]

    if name in sample_sizes:
        sample_size = sample_sizes[name]
        if all(y.value_counts() >= sample_size):
            X, y = balance_dataset(X, y, sample_size)
        else:
            print(f"No enough data in dataset {name} to balance in size of {sample_size}, skip.")
            continue

    best_pipeline  = train_and_evaluate_models(X, y, name)
    if name == "All" and best_pipeline is not None:
        dates = dates.loc[X.index]
        teams = teams.loc[X.index]
        predictions = best_pipeline.predict(X)
        results_df = pd.DataFrame({
            'Date': dates.values,
            'Teams': teams.values,
            'Prediction': predictions,
            'Actual': y,
            'Correct': predictions == y
        })
        def label_result(result, teams):
            home, away = teams.split("-vs-")
            if result == 1:
                return home
            elif result == 2:
                return away
            elif result == 3:
                return "Draw"
            return "Unknown"
        results_df["Predicted_Label"] = results_df.apply(lambda row: label_result(row["Prediction"], row["Teams"]), axis=1)
        results_df["Actual_Label"] = results_df.apply(lambda row: label_result(row["Actual"], row["Teams"]), axis=1)
        results_df["Correct"] = results_df["Predicted_Label"] == results_df["Actual_Label"]
        results_df.to_csv("results.csv", index=False)

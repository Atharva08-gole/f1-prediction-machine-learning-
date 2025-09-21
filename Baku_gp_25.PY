import fastf1
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer  # NEW

fastf1.Cache.enable_cache("f1_cache")
season = 2024
round_input = 17

# Race Session
try:
    session = fastf1.get_session(season, round_input, "R")
    session.load()
except Exception as e:
    print(f"Failed to load session: {e}")
    exit()

laps = session.laps[["Driver", "LapTime"]].dropna()
laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()

# 2025 Qualifying data
data = [
    ["Max Verstappen", 101.331, 101.255, 101.117, 23],
    ["Carlos Sainz", 102.635, 101.675, 101.595, 23],
    ["Liam Lawson", 102.257, 101.537, 101.707, 26],
    ["Kimi Antonelli", 102.247, 101.464, 101.717, 21],
    ["George Russell", 101.646, 101.455, 102.070, 22],
    ["Yuki Tsunoda", 102.347, 101.788, 102.143, 23],
    ["Lando Norris", 101.322, 101.396, 102.239, 21],
    ["Isack Hadjar", 101.656, 101.647, 102.372, 25],
    ["Oscar Piastri", 101.839, 101.414, np.nan, 18],
    ["Charles Leclerc", 101.458, 101.519, np.nan, 19],
    ["Fernando Alonso", 102.211, 101.857, np.nan, 18],
    ["Lewis Hamilton", 101.821, 102.183, np.nan, 17],
    ["Gabriel Bortoleto", 102.511, 102.277, np.nan, 16],
    ["Lance Stroll", 102.101, 103.061, np.nan, 16],
    ["Oliver Bearman", 102.666, np.nan, np.nan, 11],
    ["Franco Colapinto", 102.779, np.nan, np.nan, 8],
    ["Nico Hulkenberg", 102.916, np.nan, np.nan, 8],
    ["Pierre Gasly", 103.139, np.nan, np.nan, 8],
    ["Alexander Albon", 103.778, np.nan, np.nan, 4],
    ["Esteban Ocon", np.nan, np.nan, np.nan, 0],
]

qualifying_df = pd.DataFrame(data, columns=["Driver", "Q1", "Q2", "Q3", "Laps"])

def weighted_quali(row):
    if pd.notna(row["Q3"]):
        return 0.6 * row["Q3"] + 0.3 * row["Q2"] + 0.1 * row["Q1"]
    elif pd.notna(row["Q2"]):
        return 0.7 * row["Q2"] + 0.3 * row["Q1"]
    else:
        return row["Q1"]

qualifying_df["WeightedQuali"] = qualifying_df.apply(weighted_quali, axis=1)

# Driver dictionary
driver_data = {
    "Lando Norris": {"code": "NOR", "team": "McLaren", "country": "GBR", "logo": "🏎️"},
    "Oscar Piastri": {"code": "PIA", "team": "McLaren", "country": "AUS", "logo": "🏎️"},
    "Max Verstappen": {"code": "VER", "team": "Red Bull", "country": "NED", "logo": "🐂"},
    "George Russell": {"code": "RUS", "team": "Mercedes", "country": "GBR", "logo": "⚙️"},
    "Yuki Tsunoda": {"code": "TSU", "team": "Red Bull", "country": "JPN", "logo": "🐂"},
    "Alexander Albon": {"code": "ALB", "team": "Williams", "country": "THA", "logo": "🔵"},
    "Charles Leclerc": {"code": "LEC", "team": "Ferrari", "country": "MON", "logo": "🐎"},
    "Lewis Hamilton": {"code": "HAM", "team": "Ferrari", "country": "GBR", "logo": "🐎"},
    "Pierre Gasly": {"code": "GAS", "team": "Alpine", "country": "FRA", "logo": "⛰️"},
    "Carlos Sainz": {"code": "SAI", "team": "Williams", "country": "ESP", "logo": "🔵"},
    "Lance Stroll": {"code": "STR", "team": "Aston Martin", "country": "CAN", "logo": "🟢"},
    "Fernando Alonso": {"code": "ALO", "team": "Aston Martin", "country": "ESP", "logo": "🟢"},
    "Isack Hadjar": {"code": "HAD", "team": "RB", "country": "FRA", "logo": "🐂"},
    "Kimi Antonelli": {"code": "ANT", "team": "Mercedes", "country": "ITA", "logo": "⚙️"},
    "Esteban Ocon": {"code": "OCO", "team": "Haas", "country": "FRA", "logo": "🟥"},
    "Nico Hulkenberg": {"code": "HUL", "team": "Sauber", "country": "GER", "logo": "⬜"},
    "Oliver Bearman": {"code": "BEA", "team": "Haas", "country": "GBR", "logo": "🟥"},
    "Gabriel Bortoleto": {"code": "BOR", "team": "Sauber", "country": "BRA", "logo": "⬜"},
    "Liam Lawson": {"code": "LAW", "team": "RB", "country": "NZL", "logo": "🐂"},
}

qualifying_df["DriverCode"] = qualifying_df["Driver"].map(lambda d: driver_data.get(d, {}).get("code", None))
laps = laps[laps["Driver"].isin(qualifying_df["DriverCode"])]

# Training features
X = qualifying_df.set_index("DriverCode").loc[laps["Driver"], ["WeightedQuali", "Laps"]].values
y = laps["LapTime (s)"].values

# 🚨 Fix: remove or impute NaNs
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

if len(X) == 0:
    print("Error: No data to train on after filtering. Check driver codes.")
    exit()

# Polynomial regression model
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=42)
model = LinearRegression().fit(X_train, y_train)

# Prediction for race time (51 laps for Baku)
laps_count = 51
X_all = qualifying_df[["WeightedQuali", "Laps"]].values
X_all = imputer.transform(X_all)   # 🚨 Fix: also impute before prediction
X_all_poly = poly.transform(X_all)

qualifying_df["PredictedLapTime (s)"] = model.predict(X_all_poly)
qualifying_df["PredictedRaceTime (s)"] = qualifying_df["PredictedLapTime (s)"] * laps_count

# Displaying Results
top5 = qualifying_df.sort_values("PredictedRaceTime (s)").head(5).reset_index(drop=True)
trophies = ["🥇", "🥈", "🥉", "🎖️", "🎖️"]
top5["Trophy"] = trophies

print("\n🏁 Predicted Top 5 - BAKU GRAND PRIX 2025 🏁\n")
for _, row in top5.iterrows():
    driver = row["Driver"]
    info = driver_data.get(driver, {"team": "Unknown", "country": "Unknown", "logo": "❓"})
    total_time = row["PredictedRaceTime (s)"]
    mins = int(total_time // 60)
    secs = total_time % 60
    print(
        f"{row['Trophy']} {info['logo']} {driver} ({info['team']} - {info['country']}) — "
        f"Predicted Total Race Time: {mins}:{secs:06.3f}"
    )

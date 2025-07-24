import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Enable FastF1 caching
fastf1.Cache.enable_cache("f1_cache")

# Define the season and round
season = 2024
round_input = 24  # Chinese GP

# Load Race Session
try:
    session = fastf1.get_session(season, round_input, "R")
    session.load()
except Exception as e:
    print(f"Failed to load session: {e}")
    exit()

# Get lap times
laps = session.laps[["Driver", "LapTime"]].dropna()
laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()

# Qualifying data for 2025
qualifying_2025 = pd.DataFrame({
    "Driver": [
        "Lando Norris", "Oscar Piastri", "Max Verstappen", "George Russell", "Yuki Tsunoda",
        "Alexander Albon", "Charles Leclerc", "Lewis Hamilton", "Pierre Gasly", "Carlos Sainz",
        "Isack Hadjar", "Fernando Alonso", "Lance Stroll", "Jack Doohan", "Gabriel Bortoleto",
        "Kimi Antonelli", "Nico Hulkenberg", "Esteban Ocon", "Oliver Bearman"
    ],
    "QualifyingTime (s)": [
        75.096, 75.180, 75.481, 75.546, 75.670, 75.737, 
        75.755, 75.973, 75.980, 76.062, 76.175, 76.453, 
        76.483, 76.863, 77.520, 76.525, 76.579, 77.094, 77.147
    ]
})

# Driver Info Dictionary
driver_info = {"Lando Norris": {"code": "NOR", "team": "McLaren", "country": "GBR"},
                  "Oscar Piastri": {"code": "PIA", "team": "McLaren", "country": "AUS"},
                  "Max Verstappen": {"code": "VER", "team": "Red Bull", "country": "NED"},
                  "George Russell": {"code": "RUS", "team": "Mercedes", "country": "GBR"},
                  "Yuki Tsunoda": {"code": "TSU", "team": "Red Bull", "country": "JPN"},
                  "Alexander Albon": {"code": "ALB", "team": "Williams", "country": "THA"},
                  "Charles Leclerc": {"code": "LEC", "team": "Ferrari", "country": "MON"},
                  "Lewis Hamilton": {"code": "HAM", "team": "Ferrari", "country": "GBR"},  
                  "Pierre Gasly": {"code": "GAS", "team": "Alpine", "country": "FRA"},
                  "Carlos Sainz": {"code": "SAI", "team": "Williams","country": "ESP"},
                  "Lance Stroll": {"code": "STR", "team": "Aston Martin", "country": "CAN"},
                  "Fernando Alonso": {"code": "ALO", "team": "Aston Martin", "country": "ESP"},
                  "Isack Hadjar": {"code": "HAD", "team": "RB", "country": "FRA"},
                  "Kimi Antonelli": {"code": "ANT", "team": "Mercedes", "country": "ITA"},
                  "Esteban Ocon": {"code": "OCO", "team": "Haas", "country": "FRA"},
                  "Nico Hulkenberg": {"code": "HUL", "team": "Sauber", "country": "GER"},
                  "Oliver Bearman": {"code": "BEA", "team": "Haas", "country": "GBR"},
                  "Jack Doohan": {"code": "DOO", "team": "Alpine", "country": "AUS"},
                  "Gabriel Bortoleto": {"code": "BOR", "team": "Sauber", "country": "BRA"},  
                  "Liam Lawson": {"code": "LAW", "team": "RB", "country": "NZL"}
}

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(lambda d: driver_info[d]["code"])

# Filter laps to match drivers in qualifying list
laps = laps[laps["Driver"].isin(qualifying_2025["DriverCode"])]
lap_qual_times = laps["Driver"].map(
    qualifying_2025.set_index("DriverCode")["QualifyingTime (s)"]
)

# Drop NaNs in case of mismatches
valid_mask = ~lap_qual_times.isna()
X = lap_qual_times[valid_mask].values.reshape(-1, 1)
y = laps.loc[valid_mask, "LapTime (s)"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
qual_times = qualifying_2025[["QualifyingTime (s)"]]
qualifying_2025["PredictedRaceTime (s)"] = model.predict(qual_times)

# Displaying results
results = qualifying_2025.sort_values("PredictedRaceTime (s)").reset_index(drop=True)
top5 = results.head(5).copy()
trophies = ["🥇", "🥈", "🥉", "🎖️", "🎖️"]
top5["Trophy"] = trophies
print("\n🏁 Predicted Top 5 Finishers for LOUIS VUITTON AUSTRALIAN GRAND PRIX 2025 🏁\n")
print("🏎️" * 22)
for idx, row in top5.iterrows():
    info = driver_info[row['Driver']]
    print(f"{row['Trophy']} {row['Driver']} {info['team']}({info['country']}) — Predicted Race Time: {row['PredictedRaceTime (s)']:.3f} s")
mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"\n🔍 Model MAE: {mae:.2f} seconds")

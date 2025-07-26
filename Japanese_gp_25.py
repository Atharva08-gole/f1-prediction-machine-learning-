import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
fastf1.Cache.enable_cache("f1_cache")

season = 2024
round_input = 24  # Chinese GP

# Race Session
try:
    session = fastf1.get_session(season, round_input, "R")
    session.load()
except Exception as e:
    print(f"Failed to load session: {e}")
    exit()
laps = session.laps[["Driver", "LapTime"]].dropna()
laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()

# Qualifying data for 2025
qualifying_2025 = pd.DataFrame({
    "Driver": [
        "Max Verstappen","Lando Norris","Oscar Piastri","Charles Leclerc","George Russell",
        "Kimi Antonelli","Isack Hadjar", "Lewis Hamilton","Alexander Albon","Oliver Bearman",
        "Pierre Gasly","Carlos Sainz","Fernando Alonso","Liam Lawson","Yuki Tsunoda",
        "Nico Hulkenberg", "Gabriel Bortoleto","Esteban Ocon","Jack Doohan","Lance Stroll"
    ],
    "QualifyingTime (s)": [
        86.983,86.995,87.027,87.299,87.318,87.555,87.569,87.610,87.615,87.867,
        87.822,87.836,87.897,87.906,88.000,88.570,88.622,88.696,88.877,89.271
    ]
})
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

qualifying_2025["DriverCode"] = qualifying_2025["Driver"].map(lambda x: driver_info[x]["code"])

# Filter lap data for only those drivers in qualifying_2025
laps = laps[laps["Driver"].isin(qualifying_2025["DriverCode"])]
X = qualifying_2025.set_index("DriverCode").loc[laps["Driver"]]["QualifyingTime (s)"].values.reshape(-1, 1)
y = laps["LapTime (s)"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
predicted_laps = model.predict(qualifying_2025[["QualifyingTime (s)"]])
qualifying_2025["PredictedRaceTime (s)"] = predicted_laps

# Display results
results = qualifying_2025.sort_values("PredictedRaceTime (s)").reset_index(drop=True)
top5 = results.head(5).copy()
trophies = ["🥇", "🥈", "🥉", "🎖️", "🎖️"]
top5["Trophy"] = trophies
print("\n🏁 Predicted Top 5 Finishers for LENOVO JAPANESE GRAND PRIX 2025 🏁\n")
print("🏎️" * 22)
for idx, row in top5.iterrows():
    info = driver_info[row['Driver']]
    print(f"{row['Trophy']} {row['Driver']} {info['team']}({info['country']}) — Predicted Race Time: {row['PredictedRaceTime (s)']:.3f} s")
mae = mean_absolute_error(y_test, model.predict(X_test))
print(f"\n🔍 Model MAE: {mae:.2f} seconds")

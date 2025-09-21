import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
fastf1.Cache.enable_cache("f1_cache")
season = 2024
round_input = 24

# Race Session
try:
    session = fastf1.get_session(season, round_input, "R")
    session.load()
except Exception as e:
    print(f"Failed to load session: {e}")
    exit()

# Prepare lap data
laps = session.laps[["Driver", "LapTime"]].dropna()
laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()

# 2025 Qualifying Data
qualifying_2025 = pd.DataFrame({
    "Driver": [
        "Oscar Piastri", "George Russell", "Charles Leclerc", "Kimi Antonelli", "Pierre Gasly",
        "Lando Norris", "Max Verstappen", "Carlos Sainz", "Lewis Hamilton", "Yuki Tsunoda",
        "Jack Doohan", "Isack Hadjar", "Fernando Alonso", "Esteban Ocon", "Alexander Albon",
        "Nico Hulkenberg", "Liam Lawson", "Gabriel Bortoleto", "Lance Stroll", "Oliver Bearman"
    ],
    "QualifyingTime (s)": [
        89.841, 90.009, 90.175, 90.213, 90.216, 90.267, 90.423,
        90.680, 90.772, 91.303, 91.245, 91.271, 91.886, None, 92.040,
        92.067, 92.165, 92.186, 92.283, 92.373
    ]
})
qualifying_2025.dropna(inplace=True)

# Driver Info
driver_info = {
    "Lando Norris": {"code": "NOR", "team": "McLaren", "country": "GBR"},
    "Oscar Piastri": {"code": "PIA", "team": "McLaren", "country": "AUS"},
    "Max Verstappen": {"code": "VER", "team": "Red Bull", "country": "NED"},
    "George Russell": {"code": "RUS", "team": "Mercedes", "country": "GBR"},
    "Yuki Tsunoda": {"code": "TSU", "team": "Red Bull", "country": "JPN"},
    "Alexander Albon": {"code": "ALB", "team": "Williams", "country": "THA"},
    "Charles Leclerc": {"code": "LEC", "team": "Ferrari", "country": "MON"},
    "Lewis Hamilton": {"code": "HAM", "team": "Ferrari", "country": "GBR"},
    "Pierre Gasly": {"code": "GAS", "team": "Alpine", "country": "FRA"},
    "Carlos Sainz": {"code": "SAI", "team": "Williams", "country": "ESP"},
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


laps = laps[laps["Driver"].isin(qualifying_2025["DriverCode"])]
X = qualifying_2025.set_index("DriverCode").loc[laps["Driver"]]["QualifyingTime (s)"].values.reshape(-1, 1)
y = laps["LapTime (s)"].values

# Polynomial Regression for winner prediction(degree = 2)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
X_full_poly = poly.transform(qualifying_2025[["QualifyingTime (s)"]])
predicted_laps = model.predict(X_full_poly)
qualifying_2025["PredictedRaceTime (s)"] = predicted_laps

# Display Top 5 Result
results = qualifying_2025.sort_values("PredictedRaceTime (s)").reset_index(drop=True)
top5 = results.head(5).copy()
trophies = ["🥇", "🥈", "🥉", "🎖️", "🎖️"]
top5["Trophy"] = trophies
print("\n🏁 Predicted Top 5 Finishers for GULF AIR BAHRAIN GRAND PRIX 2025 🏁\n")
print("🏎️" * 22)
for _, row in top5.iterrows():
    info = driver_info[row['Driver']]
    print(f"{row['Trophy']} {row['Driver']} {info['team']}({info['country']}) — Predicted Race Time: {row['PredictedRaceTime (s)']:.3f} s")

# Evaluation
X_test_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, X_test_pred)
print(f"\n🔍 Model MAE (Polynomial Regression): {mae:.2f} seconds")

import fastf1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
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
laps = session.laps[["Driver", "LapTime"]].dropna()
laps["LapTime (s)"] = laps["LapTime"].dt.total_seconds()

#  Based on 2025 Qualifying data for the race
data = [
    ["Lando Norris", 101.010, 100.715, 100.562, 20],
    ["Oscar Piastri", 101.201, 100.626, 100.647, 21],
    ["Charles Leclerc", 101.635, 101.084, 100.900, 18],
    ["Max Verstappen", 101.334, 100.951, 100.903, 15],
    ["Alexander Albon", 101.772, 101.505, 101.201, 20],
    ["George Russell", 101.784, 101.254, 101.260, 18],
    ["Yuki Tsunoda", 101.840, 101.245, 101.284, 17],
    ["Isack Hadjar", 101.572, 101.281, 101.310, 19],
    ["Liam Lawson", 101.748, 101.297, 101.328, 20],
    ["Gabriel Bortoleto", 101.908, 101.336, 102.387, 18],
    ["Esteban Ocon", 101.884, 101.525, None, 14],
    ["Oliver Bearman", 101.617, 101.617, None, 13],
    ["Pierre Gasly", 101.800, 101.633, None, 14],
    ["Nico Hulkenberg", 101.844, 101.707, None, 14],
    ["Carlos Sainz", 101.691, 101.758, None, 13],
    ["Lewis Hamilton", 101.939, None, None, 8],
    ["Franco Colapinto", 102.022, None, None, 8],
    ["Kimi Antonelli", 102.139, None, None, 6],
    ["Fernando Alonso", 102.385, None, None, 8],
    ["Lance Stroll", 102.502, None, None, 8],
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

# Combined driver data dictionary with team logos as emoji placeholders
driver_data = {
    "Lando Norris": {
        "code": "NOR",
        "team": "McLaren",
        "country": "GBR",
        "logo": "🏎️"  # McLaren emoji placeholder
    },
    "Oscar Piastri": {
        "code": "PIA",
        "team": "McLaren",
        "country": "AUS",
        "logo": "🏎️"
    },
    "Max Verstappen": {
        "code": "VER",
        "team": "Red Bull",
        "country": "NED",
        "logo": "🐂"  # Bull emoji for Red Bull
    },
    "George Russell": {
        "code": "RUS",
        "team": "Mercedes",
        "country": "GBR",
        "logo": "⚙️"  # Gear emoji for Mercedes (just symbolic)
    },
    "Yuki Tsunoda": {
        "code": "TSU",
        "team": "Red Bull",
        "country": "JPN",
        "logo": "🐂"
    },
    "Alexander Albon": {
        "code": "ALB",
        "team": "Williams",
        "country": "THA",
        "logo": "🔵"  # Blue circle for Williams
    },
    "Charles Leclerc": {
        "code": "LEC",
        "team": "Ferrari",
        "country": "MON",
        "logo": "🐎"  # Horse emoji for Ferrari
    },
    "Lewis Hamilton": {
        "code": "HAM",
        "team": "Ferrari",
        "country": "GBR",
        "logo": "🐎"
    },
    "Pierre Gasly": {
        "code": "GAS",
        "team": "Alpine",
        "country": "FRA",
        "logo": "⛰️"  # Mountain emoji for Alpine
    },
    "Carlos Sainz": {
        "code": "SAI",
        "team": "Williams",
        "country": "ESP",
        "logo": "🔵"
    },
    "Lance Stroll": {
        "code": "STR",
        "team": "Aston Martin",
        "country": "CAN",
        "logo": "🟢"  # Green circle for Aston Martin
    },
    "Fernando Alonso": {
        "code": "ALO",
        "team": "Aston Martin",
        "country": "ESP",
        "logo": "🟢"
    },
    "Isack Hadjar": {
        "code": "HAD",
        "team": "RB",
        "country": "FRA",
        "logo": "🐂"
    },
    "Kimi Antonelli": {
        "code": "ANT",
        "team": "Mercedes",
        "country": "ITA",
        "logo": "⚙️"
    },
    "Esteban Ocon": {
        "code": "OCO",
        "team": "Haas",
        "country": "FRA",
        "logo": "🟥"  # Red square for Haas
    },
    "Nico Hulkenberg": {
        "code": "HUL",
        "team": "Sauber",
        "country": "GER",
        "logo": "⬜"  # White square for Sauber
    },
    "Oliver Bearman": {
        "code": "BEA",
        "team": "Haas",
        "country": "GBR",
        "logo": "🟥"
    },
    "Gabriel Bortoleto": {
        "code": "BOR",
        "team": "Sauber",
        "country": "BRA",
        "logo": "⬜"
    },
    "Liam Lawson": {
        "code": "LAW",
        "team": "RB",
        "country": "NZL",
        "logo": "🐂"
    },
}
qualifying_df["DriverCode"] = qualifying_df["Driver"].map(lambda d: driver_data.get(d, {}).get("code", None))
laps = laps[laps["Driver"].isin(qualifying_df["DriverCode"])]
X = qualifying_df.set_index("DriverCode").loc[laps["Driver"], ["WeightedQuali", "Laps"]].values
y = laps["LapTime (s)"].values

if len(X) == 0:
    print("Error: No data to train on after filtering. Check driver codes.")
    exit()

# Polynomial regression model
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, random_state=42)
model = LinearRegression().fit(X_train, y_train)

# Prediction for race time (laps)
X_all_poly = poly.transform(qualifying_df[["WeightedQuali", "Laps"]])
qualifying_df["PredictedLapTime (s)"] = model.predict(X_all_poly)
qualifying_df["PredictedRaceTime (s)"] = qualifying_df["PredictedLapTime (s)"] * 44 # May change according to the number of laps

# Displaying Results
top5 = qualifying_df.sort_values("PredictedRaceTime (s)").head(5).reset_index(drop=True)
trophies = ["🥇", "🥈", "🥉", "🎖️", "🎖️"]
top5["Trophy"] = trophies
print("\n🏁 Predicted Top 5 - MOËT & CHANDON BELGIAN GRAND PRIX 2025 🏁\n")
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

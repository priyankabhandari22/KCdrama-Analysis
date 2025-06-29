# analyze_and_predict.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import os


# 1. Load Data-
os.makedirs("output", exist_ok=True)

# Load Excel dataset
df = pd.read_excel(
    r"C:\Users\Admin\Desktop\drama.csv.xlsx",
    engine="openpyxl"
)

print("✅ Data Preview:")
print(df.head())


# 2. Clean Data

df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
df = df.dropna(subset=["Year", "Country", "Genre", "Rating"])

# 3. Set Seaborn Theme

sns.set_theme(
    style="whitegrid",
    font_scale=1.1,
    rc={"figure.dpi": 120}
)


# 4. Drama Count by Country

plt.figure(figsize=(8,5))
sns.countplot(
    data=df,
    x="Country",
    hue="Country",
    palette="Set2",
    legend=False
)
plt.title("Drama Count by Country", fontsize=16, fontweight="bold")
plt.xlabel("")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("output/drama_count_country.png")
plt.close()


# 5. Genre Popularity

plt.figure(figsize=(10,5))
sns.countplot(
    data=df,
    x="Genre",
    hue="Genre",
    order=df["Genre"].value_counts().index,
    palette="Set1",
    legend=False
)
plt.title("Genre Popularity", fontsize=16, fontweight="bold")
plt.xlabel("")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("output/genre_popularity.png")
plt.close()


# 6. Rating Distribution by Country

plt.figure(figsize=(8,5))
sns.boxplot(
    data=df,
    x="Country",
    y="Rating",
    hue="Country",
    palette="Pastel1",
    dodge=False,
    legend=False
)
plt.title("Rating Distribution by Country", fontsize=16, fontweight="bold")
plt.xlabel("")
plt.ylabel("Rating")
plt.tight_layout()
plt.savefig("output/rating_distribution_country.png")
plt.close()


# 7. Encode Categorical Features

le_country = LabelEncoder()
le_genre = LabelEncoder()
df["Country_encoded"] = le_country.fit_transform(df["Country"])
df["Genre_encoded"] = le_genre.fit_transform(df["Genre"])


# 8. Prepare Features and Target

X = df[["Year", "Country_encoded", "Genre_encoded"]]
y = df["Rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 9. Train Random Forest Model

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# 10. Predict and Evaluate

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n✅ Model Performance:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")


# 11. Feature Importance Plot

importance = pd.Series(model.feature_importances_, index=X.columns)

plt.figure(figsize=(8,5))
importance.sort_values().plot(
    kind="barh",
    color=sns.color_palette("viridis", len(importance))
)
plt.title("Feature Importance", fontsize=16, fontweight="bold")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("output/feature_importance.png")
plt.close()

print("\n✅ All analysis complete! Plots saved in the 'output' folder.")

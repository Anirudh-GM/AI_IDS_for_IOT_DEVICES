import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
df = pd.read_csv("sample_iot_data.csv")

# Encode the labels
df["label"] = LabelEncoder().fit_transform(df["label"])  # normal=0, attack=1

# Split features and labels
X = df[["packet_rate", "avg_packet_size"]]
y = df["label"]

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Train model
model = RandomForestClassifier(n_estimators=50, max_depth=5)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "rf_model.pkl")
print("✅ Model trained and saved as rf_model.pkl")

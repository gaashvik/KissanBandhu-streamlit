# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
# Load the dataset
url = "fp.csv"
df = pd.read_csv(url)

# Encode categorical variables separately
le_soil = LabelEncoder()
df['Soil Type'] = le_soil.fit_transform(df['Soil Type'])

le_crop = LabelEncoder()
df['Crop Type'] = le_crop.fit_transform(df['Crop Type'])

le_fertilizer = LabelEncoder()
df['Fertilizer Name'] = le_fertilizer.fit_transform(df['Fertilizer Name'])

# Split the dataset into features and target variable
X = df.drop('Fertilizer Name', axis=1)
y = df['Fertilizer Name']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Predict the fertilizer for a new sample
new_sample = pd.DataFrame([[30, 60, 42, 'Sandy', 'Millets', 21, 0, 18]])
new_sample[3] = le_soil.transform(new_sample[3])
new_sample[4] = le_crop.transform(new_sample[4])
prediction = model.predict(new_sample)
print(f"Predicted Fertilizer: {le_fertilizer.inverse_transform(prediction)}")
joblib.dump(model, "fp.joblib")
joblib.dump(le_soil, "le_soil.joblib")
joblib.dump(le_crop, "le_crop.joblib")
joblib.dump(le_fertilizer, "le_fertilizer.joblib")
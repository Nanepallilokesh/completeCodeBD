# import pandas as pd
# import pickle
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split

# # Step 1: Load the dataset
# df = pd.read_csv("donor_meals.csv")

# # Step 2: Encode categorical variables
# encoders = {}
# for col in ["gender", "side_effects", "health_conditions", "preferences", "meal"]:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])
#     encoders[col] = le

# # Step 3: Split features and target
# X = df.drop("meal", axis=1)  # Features
# y = df["meal"]               # Target (meal)

# # Step 4: Split dataset into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 5: Train the Random Forest Classifier
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train, y_train)

# # Step 6: Save the model and encoders
# pickle.dump(model, open("donor_meal_model.pkl", "wb"))
# for col, encoder in encoders.items():
#     pickle.dump(encoder, open(f"{col}_encoder.pkl", "wb"))

# print("Model and encoders saved successfully!")
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the dataset
df = pd.read_csv('Large_Diet_Recommendation_Dataset.csv')

# Replace missing or empty values with 'None'
df['side_effects'] = df['side_effects'].fillna('None')
df['health_conditions'] = df['health_conditions'].fillna('None')

# Encode categorical features
encoders = {}
categorical_columns = ['gender', 'preferences', 'side_effects', 'health_conditions', 'meal']
for col in categorical_columns:
    encoders[col] = LabelEncoder()
    df[col] = encoders[col].fit_transform(df[col])

# Split features and target
X = df[['age', 'gender', 'preferences', 'side_effects', 'health_conditions']]
y = df['meal']

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model and encoders
with open('diet_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('diet_encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("Model trained with more records and saved successfully!")

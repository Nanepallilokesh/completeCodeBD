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


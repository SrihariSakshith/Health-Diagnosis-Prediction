import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load the dataset
url = "heart-disease.csv"
data = pd.read_csv(url)

# Preprocess the data
X = data.drop('target', axis=1)  # Features
y = data['target']                # Target variable

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_scaled, y)

# Function to get user input with validation
def get_valid_input(prompt, valid_options=None, is_float=False):
    while True:
        try:
            user_input = input(prompt)
            if is_float:
                user_input = float(user_input)
            else:
                user_input = int(user_input)
            
            if valid_options is not None and user_input not in valid_options:
                raise ValueError("Invalid input. Please try again.")
            
            return user_input
        except ValueError as e:
            print(e)

# Function to get user input
def get_user_input():
    print("Please enter the following details:")
    age = get_valid_input("Age: ")
    sex = get_valid_input("Sex (0 = female, 1 = male): ", valid_options=[0, 1])
    cp = get_valid_input("Chest Pain Type (0-3): ", valid_options=[0, 1, 2, 3])
    trestbps = get_valid_input("Resting Blood Pressure (mm Hg): ")
    chol = get_valid_input("Serum Cholesterol (mg/dl): ")
    fbs = get_valid_input("Fasting Blood Sugar > 120 mg/dl (0 = false, 1 = true): ", valid_options=[0, 1])
    restecg = get_valid_input("Resting Electrocardiographic Results (0-2): ", valid_options=[0, 1, 2])
    thalach = get_valid_input("Maximum Heart Rate Achieved: ")
    exang = get_valid_input("Exercise Induced Angina (0 = no, 1 = yes): ", valid_options=[0, 1])
    oldpeak = get_valid_input("Depression Induced by Exercise Relative to Rest: ", is_float=True)
    slope = get_valid_input("Slope of the Peak Exercise ST Segment (0-2): ", valid_options=[0, 1, 2])
    ca = get_valid_input("Number of Major Vessels (0-3): ", valid_options=[0, 1, 2, 3])
    thal = get_valid_input("Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect): ", valid_options=[0, 1, 2])
    
    return [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

# Get user input
user_input = get_user_input()

# Convert user input into a DataFrame for proper scaling
user_input_df = pd.DataFrame([user_input], columns=X.columns)

# Scale the input using the same scaler
user_input_scaled = scaler.transform(user_input_df)

# Make a prediction
prediction = rf_model.predict(user_input_scaled)

# Output the result
if prediction[0] == 1:
    print("The model predicts: Heart Disease Present.")
else:
    print("The model predicts: No Heart Disease.")

# Please enter the following details:
# Age: 54
# Sex (0 = female, 1 = male): 1
# Chest Pain Type (0-3): 2
# Resting Blood Pressure (mm Hg): 120
# Serum Cholesterol (mg/dl): 230
# Fasting Blood Sugar > 120 mg/dl (0 = false, 1 = true): 0
# Resting Electrocardiographic Results (0-2): 1
# Maximum Heart Rate Achieved: 150
# Exercise Induced Angina (0 = no, 1 = yes): 0
# Depression Induced by Exercise Relative to Rest: 2.5
# Slope of the Peak Exercise ST Segment (0-2): 1
# Number of Major Vessels (0-3): 0
# Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect): 2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Load the diabetes dataset
url = "diabetes.csv"  # Change this to your actual file path if needed
data = pd.read_csv(url)

# Preprocess the data
X = data.drop('Outcome', axis=1)  # Features
y = data['Outcome']                # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

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

def get_valid_input(prompt, valid_options=None, is_float=False, min_value=None, max_value=None):
    while True:
        try:
            user_input = input(prompt)
            if is_float:
                user_input = float(user_input)
            else:
                user_input = int(user_input)

            # Check for valid options
            if valid_options is not None and user_input not in valid_options:
                raise ValueError("Invalid input. Please try again.")

            # Check for min and max values
            if (min_value is not None and user_input < min_value) or (max_value is not None and user_input > max_value):
                raise ValueError(f"Input must be between {min_value} and {max_value}. Please try again.")

            return user_input
        except ValueError as e:
            print(e)

# Function to get user input
def get_user_input():
    print("Please enter the following details:")
    pregnancies = get_valid_input("Number of Pregnancies (0-20): ", min_value=0, max_value=20)
    glucose = get_valid_input("Glucose Level (0-200 mg/dL): ", min_value=0, max_value=200)
    blood_pressure = get_valid_input("Blood Pressure (mm Hg) (40-200): ", min_value=40, max_value=200)
    skin_thickness = get_valid_input("Skin Thickness (mm): (0-99): ", min_value=0, max_value=99)
    insulin = get_valid_input("Insulin Level (µU/ml) (0-900): ", min_value=0, max_value=900)
    bmi = get_valid_input("Body Mass Index (BMI) (10-50): ", min_value=10, max_value=50, is_float=True)
    diabetes_pedigree = get_valid_input("Diabetes Pedigree Function (0.0-2.5): ", min_value=0.0, max_value=2.5, is_float=True)
    age = get_valid_input("Age (0-120): ", min_value=0, max_value=120)

    return [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree, age]

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
    print("The model predicts: Diabetes Present.")
else:
    print("The model predicts: No Diabetes.")

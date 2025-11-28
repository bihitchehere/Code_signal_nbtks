import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# Fetch the Bank Marketing dataset from the UCI repository
dataset_id = 222
bank_marketing = fetch_ucirepo(id=dataset_id)

# Create a single DataFrame from the fetched data
df = pd.concat([bank_marketing.data.features, bank_marketing.data.targets], axis=1)

# EDA
print(df.info())
print(df.describe())

# --- Data Preprocessing ---
# Select numeric features (excluding duration to avoid data leakage)
numeric_features = ['age', 'balance', 'campaign']

# Select categorical features with no missing values
categorical_features = ['marital', 'default', 'housing', 'loan']

# Create feature matrix
X_numeric = df[numeric_features]
X_categorical = df[categorical_features]

# Simple categorical encoding using LabelEncoder
le = LabelEncoder()
X_categorical_encoded = X_categorical.apply(le.fit_transform)

# Combine numeric and categorical features
X = pd.concat([X_numeric, X_categorical_encoded], axis=1)

# Convert the target variable 'y' from strings ('yes'/'no') to integers (1/0)
y = df['y'].map({'yes': 1, 'no': 0})

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Model Training and Evaluation ---
# Initialize a Decision Tree Classifier
# We use random_state for reproducibility
# TODO: Create a DecisionTreeClassifier with random_state=42
decision_tree = DecisionTreeClassifier(random_state=42)
# TODO: Train the model on the training data
model  = decision_tree.fit(X_train,y_train)
# TODO: Make predictions on the test data
predicted = model.predict(X_test)
# TODO: Calculate the classification report
result = classification_report(y_test, predicted)
# TODO: Print the classification report 
print(
    f"Classification report for classifier {model}:\n"
    f"{result}\n"
)
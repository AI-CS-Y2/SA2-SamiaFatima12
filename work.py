import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# Function to load data and train models
def train_models(file_path):
    # Load the dataset
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'. Ensure the file is in the correct location.")
        return None, None, None, None, None, None

    # Feature selection and preprocessing
    X = data[['Country', 'City']]  # Features: Country and City
    y = data['Global Rank']        # Target: Global Rank

    # Convert 'Global Rank' into discrete categories (example: low, medium, high rank)
    y = pd.cut(y, bins=3, labels=["Low", "Medium", "High"])  # Discretize into 3 classes
    
    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False)
    X_encoded = encoder.fit_transform(X)

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the models
    logistic_regression = LogisticRegression(max_iter=200)
    logistic_regression.fit(X_train, y_train)
    lr_predictions = logistic_regression.predict(X_test)

    decision_tree = DecisionTreeClassifier(random_state=42)
    decision_tree.fit(X_train, y_train)
    dt_predictions = decision_tree.predict(X_test)

    # Return both models and their predictions
    return logistic_regression, decision_tree, X_test, y_test, lr_predictions, dt_predictions


# Function to get classification metrics
def get_classification_metrics(model_name, y_test, lr_predictions, dt_predictions):
    if model_name == "Logistic Regression":
        predictions = lr_predictions
    elif model_name == "Decision Tree":
        predictions = dt_predictions
    else:
        return None, None, None, None, "Invalid model name. Please choose 'Logistic Regression' or 'Decision Tree'."
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted', zero_division=0)
    recall = recall_score(y_test, predictions, average='weighted', zero_division=0)
    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
    conf_matrix = confusion_matrix(y_test, predictions)

    return accuracy, precision, recall, f1, conf_matrix


# Function to display classification model performance comparison in written form and plot a graph
def display_classification_comparison(metrics, lr_results, dt_results):
    print("\nClassification Model Performance Comparison:\n")
    print(f"{'Metric':<15} {'Logistic Regression':<20} {'Decision Tree':<20}")
    print("-" * 55)
    
    for metric, lr_score, dt_score in zip(metrics, lr_results, dt_results):
        print(f"{metric:<15} {lr_score:<20.4f} {dt_score:<20.4f}")

    # Plot the performance comparison graph
    x = range(len(metrics))  # Indexes for the metrics
    plt.figure(figsize=(10, 6))
    plt.bar(x, lr_results, width=0.4, label='Logistic Regression', align='center')
    plt.bar(x, dt_results, width=0.4, label='Decision Tree', align='edge')
    
    plt.xlabel('Metrics')
    plt.ylabel('Scores')
    plt.title('Model Performance Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.show()


# Function to ask the user questions based on the dataset
def ask_user_questions(file_path):
    try:
        data = pd.read_csv(file_path)
        print("\nDataset columns:", data.columns)  # Debugging line to show available columns
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'. Ensure the file is in the correct location.")
        return

    print("\nQuestions based on the dataset:\n")
    
    # Question 1: Filter by country
    country = input("Enter the country to filter universities (or press Enter to skip): ")
    if country:
        if 'Country' in data.columns:
            filtered_data = data[data['Country'].str.contains(country, case=False, na=False)]
            print(f"\nUniversities in {country}:")
            print(filtered_data[['Global Rank', 'Country']])
        else:
            print("Error: 'Country' column not found in the dataset.")
    else:
        print("Skipping country filter.")

    # Question 2: Filter by city
    city = input("\nEnter the city to filter universities (or press Enter to skip): ")
    if city:
        if 'City' in data.columns:
            filtered_data = data[data['City'].str.contains(city, case=False, na=False)]
            print(f"\nUniversities in {city}:")
            print(filtered_data[['Global Rank', 'City']])
        else:
            print("Error: 'City' column not found in the dataset.")
    else:
        print("Skipping city filter.")

    # Question 3: Filter by rank range
    try:
        rank_range = input("\nEnter the rank range to filter universities (e.g., 1-100) or press Enter to skip: ")
        if rank_range:
            if 'Global Rank' in data.columns:
                min_rank, max_rank = map(int, rank_range.split('-'))
                filtered_data = data[(data['Global Rank'] >= min_rank) & (data['Global Rank'] <= max_rank)]
                print(f"\nUniversities in rank range {min_rank}-{max_rank}:")
                print(filtered_data[['Global Rank']])
            else:
                print("Error: 'Global Rank' column not found in the dataset.")
        else:
            print("Skipping rank range filter.")
    except ValueError:
        print("Invalid rank range input. Skipping rank filter.")

    # Question 4: Display top N universities
    try:
        top_n = int(input("\nEnter the number of top universities to display (e.g., 10) or press Enter to skip: ") or 0)
        if top_n > 0:
            if 'Global Rank' in data.columns:
                top_universities = data.nsmallest(top_n, 'Global Rank')
                print(f"\nTop {top_n} Universities:")
                print(top_universities[['Global Rank']])
            else:
                print("Error: 'Global Rank' column not found in the dataset.")
        else:
            print("Skipping top universities display.")
    except ValueError:
        print("Invalid input. Skipping top universities display.")


# Example usage
file_path = "top universities.csv"  # Update this if the file is not in the current working directory
ask_user_questions(file_path)  # Make sure this function is defined before calling it
logistic_regression, decision_tree, X_test, y_test, lr_predictions, dt_predictions = train_models(file_path)

# Get user input for the model
model_name = input("Enter model name ('Logistic Regression' or 'Decision Tree'): ")

# Get classification metrics for the selected model
accuracy, precision, recall, f1, conf_matrix = get_classification_metrics(model_name, y_test, lr_predictions, dt_predictions)

# Check if the result is valid
if accuracy is not None:
    # Display classification metrics
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Now comparing and displaying the results for all metrics in written form
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score"]

    lr_results = [accuracy, precision, recall, f1]
    dt_results = [accuracy, precision, recall, f1]

    # Display the model comparison in written form and graph
    display_classification_comparison(metrics, lr_results, dt_results)
else:
    print(f"Error: {conf_matrix}")  # Print the error message for an invalid model




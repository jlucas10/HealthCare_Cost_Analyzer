import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Define file path to local datasets folder
file_path = "datasets/healthcare_dataset.csv"

# Load and clean data
def load_data(file_path):
    df = pd.read_csv(file_path)
    # Rename relevant columns
    df = df.rename(columns={
        "Medical Condition": "condition",
        "Billing Amount": "cost",
        "Date of Admission": "date"
    })
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['cost', 'date', 'condition'], inplace=True)
    return df

# Analyze costs
def analyze_costs(df):
    avg_costs = df.groupby("condition")["cost"].mean()
    total_costs = df.groupby(df["date"].dt.to_period("M"))["cost"].sum()
    return avg_costs, total_costs

# Visualize
def plot_costs(avg_costs, total_costs):
    # Bar chart: Avg cost by condition
    plt.figure(figsize=(10, 6))
    sns.barplot(x=avg_costs.index, y=avg_costs.values)
    plt.title("Average Healthcare Costs by Medical Condition")
    plt.xlabel("Medical Condition")
    plt.ylabel("Average Cost ($)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("cost_by_condition.png")
    plt.close()
    # Line plot: Monthly trends
    plt.figure(figsize=(10, 6))
    total_costs.plot()
    plt.title("Monthly Healthcare Cost Trends")
    plt.xlabel("Month")
    plt.ylabel("Total Cost ($)")
    plt.tight_layout()
    plt.savefig("cost_trends.png")
    plt.close()

# Predict next month's costs
def predict_next_cost(df):
    monthly = df.groupby(df["date"].dt.to_period("M"))["cost"].sum().reset_index()
    monthly["month_num"] = range(len(monthly))
    X = monthly["month_num"].values.reshape(-1, 1)
    y = monthly["cost"].values
    model = LinearRegression()
    model.fit(X, y)
    next_month = np.array([[len(monthly)]])
    prediction = model.predict(next_month)[0]
    return prediction

# Main function 
def main():
    df = load_data(file_path)
    avg_costs, total_costs = analyze_costs(df)
    print("Average Costs by Medical Condition:")
    for condition, cost in avg_costs.items():
        print(f"{condition:<15} ${cost:.2f}")
    
    prediction = predict_next_cost(df)
    print(f"Predicted next month total cost: ${prediction:.2f}")
    
    plot_costs(avg_costs, total_costs)
    
    # Console menu
    while True:
        choice = input("1: View Avg Costs, 2: See Trends, 3: Predict, 4: Exit\n> ")
        if choice == "1":
            print("Average Costs by Medical Condition:")
            for condition, cost in avg_costs.items():
                print(f"{condition:<15} ${cost:.2f}")
        elif choice == "2":
            total_costs.plot()
            plt.show()
        elif choice == "3":
            print(f"Predicted next month total cost: ${prediction:.2f}")
        elif choice == "4":
            break

if __name__ == "__main__":
    main()

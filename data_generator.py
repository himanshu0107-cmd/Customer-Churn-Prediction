import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000

age = np.random.randint(18, 70, N)
tenure = np.random.randint(1, 72, N)
monthly_charges = np.round(np.random.uniform(20, 120, N), 2)
total_charges = np.round(monthly_charges * tenure + np.random.normal(0, 50, N), 2)
num_products = np.random.randint(1, 5, N)
support_calls = np.random.poisson(2, N)
contract_type = np.random.choice(["Month-to-Month", "One Year", "Two Year"], N, p=[0.5, 0.3, 0.2])
payment_method = np.random.choice(["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"], N)
internet_service = np.random.choice(["DSL", "Fiber Optic", "No"], N, p=[0.4, 0.4, 0.2])

# Churn probability influenced by key factors
churn_score = (
    0.3 * (monthly_charges / 120)
    + 0.3 * (support_calls / 10)
    - 0.2 * (tenure / 72)
    - 0.1 * (num_products / 4)
    + 0.1 * (contract_type == "Month-to-Month").astype(float)
    + np.random.normal(0, 0.1, N)
)
churn = (churn_score > churn_score.mean()).astype(int)

df = pd.DataFrame({
    "age": age,
    "tenure": tenure,
    "monthly_charges": monthly_charges,
    "total_charges": total_charges,
    "num_products": num_products,
    "support_calls": support_calls,
    "contract_type": contract_type,
    "payment_method": payment_method,
    "internet_service": internet_service,
    "churn": churn,
})

df.to_csv("customer_data.csv", index=False)
print(f"Dataset saved: {df.shape[0]} rows, churn rate: {df['churn'].mean():.1%}")

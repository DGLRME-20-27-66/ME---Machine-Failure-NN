import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

# 1. DATA LADEN
try:
    # We gebruiken de exacte naam uit je dataset
    df = pd.read_csv('equipment_failure_data_1.csv') 
    print("Data loaded")
except FileNotFoundError:
    print("File not found")
    exit()

# 2. DEFINIEER FEATURES EN TARGET
# We gebruiken de sensoren (S..) en AGE_OF_EQUIPMENT als voorspellers
TARGET = 'EQUIPMENT_FAILURE'
REGION_COL = 'REGION_CLUSTER'
FEATURES = ['S15', 'S17', 'S13', 'S5', 'S16', 'S19', 'S18', 'S8', 'AGE_OF_EQUIPMENT']

# Verwijder rijen met missende data om errors te voorkomen
df = df.dropna(subset=FEATURES + [TARGET])

def train_and_evaluate(train_df, test_df, title):
    """Functie om een model te trainen en resultaten te tonen."""
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]
    
    # Random Forest Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Voorspellingen
    preds = model.predict(X_test)
    
    # Statistieken
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    
    print(f"\n=== {title} ===")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return model

# 3. UITVOEREN VAN DE EXPERIMENTEN UIT HET PROPOSAL

# Pak twee regio's om te vergelijken
regions = df[REGION_COL].unique()
reg_a = regions[0]
reg_b = regions[1]

print(f"\nOnderzoek naar regio's: {reg_a} (Source) en {reg_b} (Target)")

# --- EXPERIMENT 1: The 'Local' Benchmark ---
# Train en test binnen dezelfde regio (Region A)
data_a = df[df[REGION_COL] == reg_a]
train_l, test_l = train_test_split(data_a, test_size=0.3, random_state=42)
model_local = train_and_evaluate(train_l, test_l, f"Local Benchmark: {reg_a} -> {reg_a}")

# --- EXPERIMENT 2: The 'Foreign' Test (Stress Test) ---
# Train op de volledige Region A, test op de volledig nieuwe Region B
data_b = df[df[REGION_COL] == reg_b]
model_foreign = train_and_evaluate(data_a, data_b, f"Foreign Test: {reg_a} -> {reg_b}")

# --- EXPERIMENT 3: Feature Stability Analysis ---
# Welke sensoren zijn het belangrijkst?
print("\n=== Feature Importance (Top 5) ===")
importances = pd.Series(model_local.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(importances.head(5))

# Bonus: Een simpele grafiek van de belangrijkste features
importances.plot(kind='barh', color='skyblue')
plt.title(f'Belangrijkste sensoren voor {reg_a}')
plt.xlabel('Belangrijkheid')
plt.tight_layout()
plt.show()
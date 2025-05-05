import pickle
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

with open("./iterative_model_output/model.pkl", "rb") as f:
    model = pickle.load(f)

new_df = pd.read_csv("./gpt_features/articles_with_features.csv")

exclude_cols = [
    'Author', 'Source', 'NYT', 'Genre', 'PubDate',
    'Article Title', 'Article Text', 'URL', 'index'
]
exclude_cols = [c for c in exclude_cols if c in new_df.columns]
feature_cols = [c for c in new_df.columns if c not in exclude_cols]

X_raw = new_df[feature_cols].apply(pd.to_numeric, errors="coerce").values

imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()

X_imp = imputer.fit_transform(X_raw)
X_scaled = scaler.fit_transform(X_imp)

expected_features = 669
current_features = X_scaled.shape[1]

print(f"Model expects {expected_features} features, but current data has {current_features} features")

if current_features < expected_features:
    padding = np.zeros((X_scaled.shape[0], expected_features - current_features))
    X_padded = np.hstack((X_scaled, padding))
    print(f"Padded {expected_features - current_features} missing features with zeros")
    X_scaled = X_padded
elif current_features > expected_features:
    X_scaled = X_scaled[:, :expected_features]
    print(f"Truncated {current_features - expected_features} extra features")

print(f"Final feature matrix shape: {X_scaled.shape}")

y_pred = model.predict(X_scaled)
y_proba = model.predict_proba(X_scaled)

out = new_df.copy()
out["pred_label"] = y_pred
out["prob_class_0"] = y_proba[:,0]
out["prob_class_1"] = y_proba[:,1]

print(out[["pred_label", "prob_class_0", "prob_class_1"]])
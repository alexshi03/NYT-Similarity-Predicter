import pickle
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. Load your trained model
with open("./iterative_model_output/model.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Load your new data (must have all the same raw columns as in training)
new_df = pd.read_csv("./gpt_features/articles_with_features.csv")

# 3. Reproduce the same “exclude” + feature-column logic:
exclude_cols = [
    'Author', 'Source', 'NYT', 'Genre', 'PubDate',
    'Article Title', 'Article Text', 'URL', 'index'
]
exclude_cols = [c for c in exclude_cols if c in new_df.columns]
feature_cols = [c for c in new_df.columns if c not in exclude_cols]

# 4. Convert to numeric and build your X matrix
#    (for single-article predictions, treat each row as “one sample”,
#     or if you’re predicting on article-pairs, compute diffs exactly as you did in training)
X_raw = new_df[feature_cols].apply(pd.to_numeric, errors="coerce").values

# 5. Impute & scale just like in training
imputer = SimpleImputer(strategy="mean")
scaler  = StandardScaler()

#    If you still have access to your original training features (X_train),
#    you should .fit() the imputer+scaler on those. If not, you can fit on X_raw:
X_imp    = imputer.fit_transform(X_raw)
X_scaled = scaler .fit_transform(X_imp)

# 6. Now predict
y_pred    = model.predict(X_scaled)
y_proba   = model.predict_proba(X_scaled)  # if you want class probabilities

# 7. Put them back into a DataFrame for easy viewing
out = new_df.copy()
out["pred_label"]   = y_pred
out["prob_class_0"] = y_proba[:,0]
out["prob_class_1"] = y_proba[:,1]

print(out[["pred_label","prob_class_0","prob_class_1"]].head())

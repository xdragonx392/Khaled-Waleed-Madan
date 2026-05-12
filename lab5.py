# Task 1 — Create a new engineered feature

df["delivery_speed"] = df["Distance_km"] / df["Delivery_Time_min"]

# Task 2 — Try a different rule for is_peak_hour

df["hour"] = pd.to_datetime(df["Order_Time"]).dt.hour
df["is_peak_hour"] = df["hour"].apply(lambda x: 1 if (12 <= x <= 14) or (18 <= x <= 22) else 0)

# Task 3 — Change top_k in Item_Name_reduced

top_k = 30
top_items = df["Item_Name"].value_counts().nlargest(top_k).index

df["Item_Name_reduced"] = df["Item_Name"].apply(
    lambda x: x if x in top_items else "Other"
)

# Task 4 — Run feature selection

from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

selector = SelectFromModel(
    estimator=RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    ),
    threshold="median"
)

model_fs = Pipeline(steps=[
    ("preprocess", preprocess),
    ("select", selector),
    ("rf", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample"
    ))
])

model_fs.fit(X_train, y_train)

y_pred_fs = model_fs.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_fs))
print(classification_report(y_test, y_pred_fs))

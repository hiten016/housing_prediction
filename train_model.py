import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib


df = pd.read_csv('Housing.csv')

df.columns = df.columns.str.strip().str.lower()

binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning']
for col in binary_cols:
    df[col] = df[col].map({'yes': 1, 'no': 0})

df['price'] = np.log1p(df['price'])

X = df.drop('price', axis=1)
y = df['price']

numeric_features = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']
categorical_features = ['furnishingstatus']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ],
    remainder='passthrough'  
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

param_grid = {
    'regressor__n_estimators': [100, 200],
    'regressor__max_depth': [None, 10, 20],
}

grid = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X, y)

best_model = grid.best_estimator_

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
best_model.fit(X_train, y_train)
preds = best_model.predict(X_test)
print("MSE:", mean_squared_error(y_test, preds))
print("R2 Score:", r2_score(y_test, preds))


joblib.dump(best_model, 'housing_price_model.pkl')
print("Model saved as housing_price_model.pkl")

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

boston = fetch_openml(name="boston", version=1, as_frame=True)
df = boston.frame
# print(df.head())
df = df.apply(pd.to_numeric, errors = "coerce")
df = df.fillna(df.mean())

# categorize into x and y 
x = df.drop("MEDV", axis=1)
y = df["MEDV"]

#testing and training 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 42)

x_train = x_train.astype(float)
y_train = y_train.astype(float)

#Linear regression
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

#Ridge regression 
ridge = Ridge(alpha = 1.0)
ridge.fit(x_train, y_train)

# Lasso regression
lasso = Lasso(alpha=0.1)
lasso.fit(x_train, y_train)

# random forest regression
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(x_train,y_train)

models = {
    "Linear regression": lin_reg,
    "Ridge regression": ridge,
    "Lasso regression": lasso,
    "Random forest": rf
}

for name, model in models.items():
    preds = model.predict(x_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"{name}:")
    print("  R² Score:", r2)
    print("  RMSE:", rmse)
    print("-" * 30)
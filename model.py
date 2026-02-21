from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def train_models(X_train, y_train):
    
    linear = LinearRegression()
    ridge = Ridge()
    lasso = Lasso()

    linear.fit(X_train, y_train)
    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)

    return linear, ridge, lasso

def evaluate_model(model, X_test, y_test):
    
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return y_pred, rmse, r2
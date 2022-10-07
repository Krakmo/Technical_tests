import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from xgboost import XGBRegressor
from xgboost import plot_importance
import xgboost as xgb

from sklearn.metrics import mean_squared_error

from data_preparation import main_data_preparation

from params import main_filepath, filepath_datas, filename, columns_boolean, columns_categorical, columns_numerical, features_model, target, model_list_str, filepath_model

def search_best_params_xgbregressor(X_train, y_train, X_test, y_test):

    # search best params for XGBoost Regressor

    params = { 'max_depth': [3,6,10],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 500, 1000],
            'colsample_bytree': [0.3, 0.7]}

    fit_params = {"eval_set" : [[X_test, y_test]]}

    xgbr = xgb.XGBRegressor(seed = 20)
    clf = GridSearchCV(estimator=xgbr, 
                    param_grid=params,
                    scoring='neg_mean_squared_error', 
                    verbose=0)
                    
    clf.fit(X_train, y_train, **fit_params, verbose=0)

    best_params = clf.best_params_
    rmse = (-clf.best_score_)**(1/2.0)

    print("Best parameters:", best_params)
    print("Lowest RMSE: ", rmse)

    return best_params, rmse


def best_param_lasso_ridge(model, X_train, y_train, X_test, y_test):

    # function that determines best params for lasso and ridge model
    
    liste_alphas_rmse = []

    for alpha in np.arange(0.01, 3, 0.01):
        lr = model(alpha)
        lr.fit(X_train, y_train)

        y_pred = lr.predict(X_test)

        rmse = mean_squared_error(y_test, y_pred, squared=False)
        liste_alphas_rmse.append((alpha, rmse))

    liste_alphas_rmse.sort(key=lambda y: y[1])
    
    alpha_optimal = liste_alphas_rmse[0][0]
    rmse = liste_alphas_rmse[0][1]

    return alpha_optimal, rmse

def search_best_model(model_list_str, X_train, y_train, X_test, y_test):

    # search best model amongst a list of models with hyperparameters optimization

    liste_model_rmse = []

    for model_str in model_list_str:

        if model_str == 'LinearRegression':

            lr = LinearRegression()
            lr.fit(X_train, y_train)

            y_pred = lr.predict(X_test)

            rmse = mean_squared_error(y_test, y_pred, squared=False)
            liste_model_rmse.append((lr, rmse, model_str))

        elif model_str == 'Lasso':
            alpha_optimal, rmse = best_param_lasso_ridge(Lasso, X_train, y_train, X_test, y_test)
            lr = Lasso(alpha_optimal)
            liste_model_rmse.append((lr, rmse, model_str))

        elif model_str == 'Ridge':
            alpha_optimal, rmse = best_param_lasso_ridge(Ridge, X_train, y_train, X_test, y_test)
            lr = Ridge(alpha_optimal)
            liste_model_rmse.append((lr, rmse, model_str))

        elif model_str == 'XGBRegressor':
            best_params, rmse = search_best_params_xgbregressor(X_train, y_train, X_test, y_test)
            lr = XGBRegressor(max_depth = best_params['max_depth'],
                              learning_rate = best_params['learning_rate'],
                              n_estimators = best_params['n_estimators'],
                              colsample_bytree = best_params['colsample_bytree']
                              )                       
            liste_model_rmse.append((lr, rmse, model_str))

    liste_model_rmse.sort(key=lambda y: y[1])

    print(liste_model_rmse)

    best_model = liste_model_rmse[0][0]
    lowest_rmse = liste_model_rmse[0][1]
    best_model_str = liste_model_rmse[0][2]

    best_model.fit(X_train, y_train)

    print("best model is :", best_model)

    return best_model, best_model_str

def save_best_model(best_model, best_model_str, filepath_model):

    # save model in pickle

    with open(filepath_model + best_model_str + ".reg", "wb") as f_out:
        pickle.dump(best_model, f_out)

if __name__ == "__main__":

    df_student = pd.read_csv(filepath_datas + filename)
    
    X_train, y_train, X_test, y_test = main_data_preparation(df_student, columns_boolean, columns_categorical, features_model, target)
    best_model, best_model_str = search_best_model(model_list_str, X_train, y_train, X_test, y_test)
    save_best_model(best_model, best_model_str, filepath_model)

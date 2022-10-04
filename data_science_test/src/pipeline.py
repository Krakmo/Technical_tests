import pandas as pd
from pandas import MultiIndex, Int16Dtype

import numpy as np
from sklearn.feature_extraction import DictVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from xgboost import XGBRegressor
from xgboost import plot_importance
import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from sklearn.metrics import mean_squared_error

def data_preparation(df_student, columns_boolean, columns_categorical):
    def convert_bool_in_number(df, column_name):
        df[column_name] = pd.Series(np.where(df[column_name].values == 'yes', 1, 0),
            df.index)
            
        return df

    df_student_2 = df_student.copy()

    for column in columns_boolean:
        df_student_2 = convert_bool_in_number(df_student_2, column)

    def encode_and_concat(df, categorical_column):

        dummies = pd.get_dummies(df[categorical_column], prefix= str(categorical_column))
        df_concat = pd.concat([df, dummies], axis=1)
        df_concat = df_concat.drop(categorical_column, axis = 1)

        return(df_concat)

    df_concat = df_student_2.copy()

    for column in columns_categorical:
        df_concat = encode_and_concat(df_concat, column)

    return df_concat

def split_dataset(df_preprocessed, features_test, target):
    X = df_preprocessed[features_test]
    y = df_preprocessed[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    return X_train, X_test, y_train, y_test

def train_model(model, X_train, X_test, y_train, y_test):

    lr = model()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    #print('RMSE = ', rmse)

    return y_pred

def visualisation(df_student, y_test, y_pred, filepath_figures, filename_figure):

    y_tot = y_test.copy()
    y_tot['score'] = y_pred
    df_tot = y_tot.merge(df_student[['FirstName', 'FamilyName']], left_index = True, right_index = True)
    df_tot['Full_Name'] = df_tot['FirstName'] + ' ' + df_tot['FamilyName']

    fig = px.scatter(df_tot, x="FinalGrade", y="score", hover_name = "Full_Name")
    fig.update_traces(textposition='top center', marker= dict(color = 'blue'))
    fig.update_layout(
        height= 600,
        width = 1000,
        title_text='Score in function of FinalGrade'
    )

    return fig

if __name__ == "__main__":

    filepath = 'D:/Python/Ekinox/data_science_test/datas/'
    filename =  'student_data.csv'
    df_student = pd.read_csv(filepath + filename)

    target = ['FinalGrade']
    columns_numerical = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
    columns_boolean = ['schoolsup', 'paid', 'famsup', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    columns_categorical = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']

    features_test = ['failures', 'schoolsup', 'absences', 'Medu', 'Walc', 'goout', 'Mjob_other', 'Dalc', 'age', 'Mjob_health']

    model = LinearRegression
    filepath_figures = 'D:/Python/Ekinox/data_science_test/figures/'
    filename_figure = 'LinearRegression'

    df_preprocessed = data_preparation(df_student, columns_boolean, columns_categorical)
    X_train, X_test, y_train, y_test = split_dataset(df_preprocessed, features_test, target)
    y_pred = train_model(model, X_train, X_test, y_train, y_test)
    plot = visualisation(df_student, y_test, y_pred, filepath_figures, filename_figure)
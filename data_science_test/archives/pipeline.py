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

import pickle 

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

def train_model(model, X_train, X_test, y_train, y_test, filepath_model, model_str):

    lr = model()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    #print('RMSE = ', rmse)

    with open(filepath_model + model_str + ".reg", "wb") as f_out:
        pickle.dump(lr, f_out)

    return lr

def load_and_use_model(df_preprocessed, filepath_model, str_model):

    with open(filepath_model+ str_model+".reg", "rb") as f_out:
        model = pickle.load(f_out)

    y_pred = model.predict(df_preprocessed)

    return y_pred

def visualisation(df_student, y_test, y_pred, filepath_figures, filename_figure):

    y_tot = y_test.copy()
    y_tot['score'] = y_pred
    df_tot = y_tot.merge(df_student[['FirstName', 'FamilyName']], left_index = True, right_index = True)
    df_tot['Full_Name'] = df_tot['FirstName'] + ' ' + df_tot['FamilyName']

    fig = px.scatter(df_tot, x="FinalGrade", y="score", hover_name = "Full_Name")
    fig['layout']['xaxis']['autorange'] = "reversed"
    fig.update_traces(textposition='top center', marker= dict(color = 'blue'))
    fig.update_layout(
        height= 600,
        width = 1000,
        title_text='Score in function of FinalGrade',
    )
    fig.update_layout({'plot_bgcolor': 'rgba(119, 181, 254, 255)'
                      })

    fig['layout']['xaxis']['autorange'] = "reversed"
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='Black')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='Black')
    fig.update_traces(marker_size=10)

    return fig

def build(df_student, columns_boolean, columns_categorical, features_model, target, model, str_model, filepath_model, filepath_figures):

    df_preprocessed = data_preparation(df_student, columns_boolean, columns_categorical)

    X_train, X_test, y_train, y_test = split_dataset(df_preprocessed, features_model, target)

    lr = train_model(model, X_train, X_test, y_train, y_test, filepath_model, str_model)

    y_pred = load_and_use_model(df_preprocessed[features_model], filepath_model, str_model)

    plot = visualisation(df_student, y_test, y_pred, filepath_figures, str_model)

    return plot
    
if __name__ == "__main__":

    main_filepath = 'D:/Python/Ekinox/Technical_tests/data_science_test/'

    filepath_datas = main_filepath + '/datas/'

    filename =  'student_data.csv'
    df_student = pd.read_csv(filepath_datas + filename)

    columns_numerical = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
    columns_boolean = ['schoolsup', 'paid', 'famsup', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    columns_categorical = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']

    features_model = ['failures', 'schoolsup', 'absences', 'Medu', 'Walc', 'goout', 'Mjob_other', 'Dalc', 'age', 'Mjob_health']
    target = ['FinalGrade']

    model = LinearRegression
    str_model = 'LinearRegression'
    filepath_figures = main_filepath + '/figures/'
    filepath_model = main_filepath + '/models/'

    plot = build(df_student, columns_boolean, columns_categorical, features_model, target, model, str_model, filepath_model, filepath_figures)
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

import pickle 

from params import main_filepath, filepath_datas, filename, columns_numerical, columns_boolean, columns_categorical, features_model, target

def data_prep(df_student, columns_boolean, columns_categorical):

    # Preprocess our dataset to fit with our goals and our model

    def convert_bool_in_number(df, column_name):

        # convert yes in 1 and no in 0

        df[column_name] = pd.Series(np.where(df[column_name].values == 'yes', 1, 0),
            df.index)
            
        return df

    df_student_2 = df_student.copy()

    for column in columns_boolean:
        df_student_2 = convert_bool_in_number(df_student_2, column)

    def encode_and_concat(df, categorical_column):

        # use OneHotEncoding method to encode our categorical features

        dummies = pd.get_dummies(df[categorical_column], prefix= str(categorical_column))
        df_concat = pd.concat([df, dummies], axis=1)
        df_concat = df_concat.drop(categorical_column, axis = 1)

        return(df_concat)

    df_concat = df_student_2.copy()

    for column in columns_categorical:
        df_concat = encode_and_concat(df_concat, column)

    return df_concat

def split_dataset(df_preprocessed, features_test, target):

    # Split our dataset in training and testing sets

    X = df_preprocessed[features_test]
    y = df_preprocessed[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    return X_train, X_test, y_train, y_test

def main_data_preparation(df_student, columns_boolean, columns_categorical, features_model, target):

    # Preprocess our dataset and split into training and testing sets

    df_preprocessed = data_prep(df_student, columns_boolean, columns_categorical)
    X_train, X_test, y_train, y_test = split_dataset(df_preprocessed, features_model, target)

    return X_train, y_train, X_test, y_test
    
if __name__ == "__main__":

    df_student = pd.read_csv(filepath_datas + filename)
    
    X_train, y_train, X_test, y_test = main_data_preparation(df_student, columns_boolean, columns_categorical, features_model, target)
    print('X_train', X_train)
    print('X_test', X_test)
    print('y_train', y_train)
    print('y_test', y_test)
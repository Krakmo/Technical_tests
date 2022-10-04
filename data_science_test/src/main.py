import streamlit as st
import pandas as pd
from pipeline import *

st.title("Eleves en difficulté")

st.write("""
## Eleves à aider
""")

dataset_name = st.sidebar.selectbox("Select Dataset", ("student_data : University of Minho", "")) 

def get_dataset(dataset_name):

    if dataset_name == "student_data : University of Minho":
        df_student = pd.read_csv('D:/Python/Ekinox/data_science_test/datas/student_data.csv')

    else:
        df_student = pd.read_csv('D:/Python/Ekinox/data_science_test/datas/student_data.csv')

    if st.checkbox('Show Raw Data'):
        "Dataset", df_student

    return df_student

df_student = get_dataset(dataset_name)
st.write("shape of the dataset", df_student.shape)

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

st.write(plot)
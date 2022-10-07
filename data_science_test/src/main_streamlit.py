import streamlit as st
import pandas as pd

from data_preparation import data_prep
from load_best_model_and_visualisation import load_model, use_model, visualisation

from params import main_filepath, filepath_datas, filename, columns_boolean, columns_categorical, columns_numerical, features_model, target, filepath_model, model_str, filepath_figures

st.title("Elèves en difficultés")

dataset_name = st.sidebar.selectbox("Select Dataset", ("student_data : University of Minho", "")) 

def get_dataset(dataset_name, filepath_datas, filename):

    # choose between several datasets (only one available for now)
    
    if dataset_name == "student_data : University of Minho":
        df_student = pd.read_csv(filepath_datas + filename)

    else:
        df_student = pd.read_csv(filepath_datas + filename)

    if st.checkbox('Show Raw Data'):
        "Dataset", df_student

    return df_student

df_student = get_dataset(dataset_name, filepath_datas, filename)

st.write("shape of the dataset", df_student.shape)

df_preprocessed = data_prep(df_student, columns_boolean, columns_categorical)

model = load_model(filepath_model, model_str)
y_pred = use_model(df_preprocessed[features_model], model)

y_test = df_preprocessed[target]

plot = visualisation(df_student, y_test, y_pred, filepath_figures, model_str)

st.write('## Graphe intéractif')
st.write(plot)
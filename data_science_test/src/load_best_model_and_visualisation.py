import pickle
import plotly.express as px
import plotly.io as pio

import matplotlib.pyplot as plt

import pandas as pd

from data_preparation import data_prep

from params import main_filepath, filepath_datas, filename, columns_boolean, columns_categorical, columns_numerical, features_model, target, filepath_model, model_str, filepath_figures

def load_model(filepath_model, str_model):

    with open(filepath_model+ str_model+".reg", "rb") as f_out:
        model = pickle.load(f_out)

    return model

def use_model(df_preprocessed, model):

    y_pred = model.predict(df_preprocessed)

    return y_pred

def visualisation(df_student, y_test, y_pred, filepath_figures, filename_figure):

    # use plotly to create interactive graph using our datas and model

    y_tot = y_test.copy()
    y_tot['score'] = y_pred
    
    df_tot = y_tot.merge(df_student[['FirstName', 'FamilyName']], left_index = True, right_index = True)
    df_tot['Full_Name'] = df_tot['FirstName'] + ' ' + df_tot['FamilyName']

    fig = px.scatter(df_tot, x="FinalGrade", y="score", hover_name = "Full_Name")
    fig['layout']['xaxis']['autorange'] = "reversed"
    fig['layout']['yaxis']['autorange'] = "reversed"

    fig.update_traces(textposition='top center', marker= dict(color = 'blue'))
    fig.update_layout(
        height= 600,
        width = 1000,
        title_text="Score de prédiction en fonction de la note finale : plus le score est faible, plus il y a de l'intérêt à accompagner l'élève (axes inversés)",
    )
    fig.update_layout({'plot_bgcolor': 'rgba(119, 181, 254, 255)'
                      })
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='Black')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='Black')
    fig.update_traces(marker_size=10)

    pio.write_image(fig, filepath_figures + filename_figure + '.png', format = 'png')

    return fig

if __name__ == "__main__":

    df_student = pd.read_csv(filepath_datas + filename)

    df_preprocessed = data_prep(df_student, columns_boolean, columns_categorical)

    model = load_model(filepath_model, model_str)
    y_pred = use_model(df_preprocessed[features_model], model)

    y_test = df_preprocessed[target]

    plot = visualisation(df_student, y_test, y_pred, filepath_figures, model_str)

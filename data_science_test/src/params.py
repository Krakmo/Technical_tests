import os

main_filepath = '..'

filepath_datas = main_filepath + '/datas/'

filename =  'student_data.csv'

columns_numerical = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
columns_boolean = ['schoolsup', 'paid', 'famsup', 'activities', 'nursery', 'higher', 'internet', 'romantic']
columns_categorical = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']

# features selected after correlation study in notebook that will be used in our model
features_model = ['failures', 'schoolsup', 'absences', 'Walc', 'goout', 'Dalc', 'studytime', 'internet']

target = ['FinalGrade']

model_list_str = ['LinearRegression', 'Lasso', 'Ridge', 'XGBRegressor']

filepath_model = main_filepath + '/models/'

model_str = 'Lasso'

filepath_figures = main_filepath + '/figures/'

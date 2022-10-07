main_filepath = 'D:/Python/Ekinox/Technical_tests/data_science_test/'

filepath_datas = main_filepath + '/datas/'

filename =  'student_data.csv'

columns_numerical = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
columns_boolean = ['schoolsup', 'paid', 'famsup', 'activities', 'nursery', 'higher', 'internet', 'romantic']
columns_categorical = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian']

# features selected after correlation study in notebook
features_model = ['failures', 'schoolsup', 'absences', 'Walc', 'goout', 'Dalc', 'studytime', 'internet']

target = ['FinalGrade']

model_list_str = ['LinearRegression', 'Lasso', 'Ridge', 'XGBRegressor']

filepath_model = main_filepath + '/models/'

model_str = 'Lasso'

filepath_figures = main_filepath + 'figures/'

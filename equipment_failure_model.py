
#%% Libraries
'We start by importing some libraries'

import pandas as pd
import numpy as np
dedatetime import datetime
nan_value = ("NaN")
import matplotlib.pyplot as plt
desklearn.preprocessing import StandardScaler desklearn.linear_model import LogisticRegression desklearn.naive_bayes import GaussianNB
desklearn.ensemble import RandomForestClassifier, VotingClassifier desklearn.model_selection import train_test_split
desklearn.metrics import classification_report desklearn.preprocessing import LabelEncoder
descipy.stats import chi2_contingency
desklearn import svm
desklearn.metrics import confusion_matrix
import seaborn as sns
import tensorflow as tf

pd.set_option('display.max_rows', None) pd.set_option('display.max_columns', None)

#%% Upload data bases
Fields = list(range(0,45))
Fields = [str(i) for i in Fields]
Data_IW67 = pd.read_csv("C:/Users/laura/OneDrive/Escritorio/iw67.csv", encoding = 'latin-1', sep = ';', engine = "python", names=Fields)
Header = Data_IW67.iloc[0,:]
Header = Header.values.tolist()
Data_IW67 = Data_IW67[1:]
Data_IW67.columns = Header

#%% Functions declaration
def change_value(Data_Frame,column, String1, String2):
	Data_Frame.loc[:,column] = np.where((Data_Frame.loc[:,column] == String1), String2, Data_Frame.loc[:,column])

def label_encoder (Data_Frame, String):
	lb = LabelEncoder()
	Data_Frame.loc[:,String] = lb.fit_transform(Data_Frame.loc[:, String]) return lb

def delete_rows(Data_Frame, Column_1 ,String):
	Data_Frame = Data_Frame[Data_Frame[Column_1] != String] return Data_Frame

def date_generator(counter):
	Value = Data_ajustada_sin_columnas.loc[counter,'Planned start'] Day = Value.day Data_ajustada_sin_columnas.loc[counter,'Month'] = Value.month Data_ajustada_sin_columnas.loc[counter,'Year'] = Value.year
	if ((Day >=1) & (Day <=7)):
		Data_ajustada_sin_columnas.loc[counter, 'Week'] = 1
	elif ((Day >=8) & (Day <=15)): Data_ajustada_sin_columnas.loc[counter, 'Week'] = 2
	elif ((Day >=16) & (Day <=23)): Data_ajustada_sin_columnas.loc[counter, 'Week'] = 3
	elif ((Day >=24) & (Day <=31)): Data_ajustada_sin_columnas.loc[counter, 'Week'] = 4

def fill_nas(Data_Frame, column):
	Data_Frame[column] = Data_Frame[column].fillna(int(Data_Frame[column].mean()))
	#%% Data - Cleaning
	Data_IW67['Planned start'] = Data_IW67['Planned start'].astype(str) for i in range(1,len(Data_IW67)):
	if ('-' in Data_IW67.loc[i,"Planned start"]) or ('n' in Data_IW67.loc[i,"Planned start"]) : Data_IW67 = Data_IW67.drop(index = i)
		Data_IW67 = Data_IW67.reset_index() for i in range(1,len(Data_IW67)):
		Date = Data_IW67.loc[i, 'Planned start']
		Data_IW67.loc[i,"Planned start"] = datetime.strptime(Date, '%d/%m/%Y') 'We only take prevetive and corrective maintenances'
		Data_ajustada = Data_IW67.loc[(Data_IW67["Notifictn type"] == "MM") | (Data_IW67["Notifictn type"] == "MS")]
		Data_ajustada_sin_columnas = Data_ajustada.drop_duplicates(subset = "Notification", keep ='last')

'Variable City Cleaning'
change_value(Data_ajustada_sin_columnas, 'City', 'BOGOTA, D.C.', 'BOGOTA')
change_value(Data_ajustada_sin_columnas, 'City', 'CARTAGENA DE INDIAS', 'CARTAGENA')
change_value(Data_ajustada_sin_columnas, 'City', 'BELLO', 'MEDELLIN')
change_value(Data_ajustada_sin_columnas, 'City', 'CHIA', 'BOGOTA')
change_value(Data_ajustada_sin_columnas, 'City', 'Villavicencio', 'VILLAVICENCIO')
change_value(Data_ajustada_sin_columnas, 'City', 'ITAGUI', 'MEDELLIN')
change_value(Data_ajustada_sin_columnas, 'City', 'BOGOTÁ', 'BOGOTA')
change_value(Data_ajustada_sin_columnas, 'City', 'BOGOTÁ, D.C.', 'BOGOTA')
change_value(Data_ajustada_sin_columnas, 'City', 'Bogotá', 'BOGOTA')
change_value(Data_ajustada_sin_columnas, 'City', 'CALDAS', 'MANIZALES')
change_value(Data_ajustada_sin_columnas, 'City', 'CALDAS-ANTIOQUIA', 'MANIZALES')
change_value(Data_ajustada_sin_columnas, 'City', 'Cartagena', 'CARTAGENA')
change_value(Data_ajustada_sin_columnas, 'City', 'ESPINAL, TOLIMA', 'ESPINAL')
change_value(Data_ajustada_sin_columnas, 'City', 'FLORIDA', 'FLORIDABLANCA')
change_value(Data_ajustada_sin_columnas, 'City', 'GUADALAJARA DE BUGA', 'BUGA')
change_value(Data_ajustada_sin_columnas, 'City', 'Florencia', 'FLORENCIA')
change_value(Data_ajustada_sin_columnas, 'City', 'MANIZALES', 'MANIZALEZ')
change_value(Data_ajustada_sin_columnas, 'City', 'Medellin', 'MEDELLIN')
change_value(Data_ajustada_sin_columnas, 'City', 'PERERIA', 'PEREIRA')
change_value(Data_ajustada_sin_columnas, 'City', 'SABANETA','MEDELLIN')
change_value(Data_ajustada_sin_columnas, 'City', 'TIERRA ALTA', 'TIERRALTA')
change_value(Data_ajustada_sin_columnas, 'City', 'Yopal', 'YOPAL')
change_value(Data_ajustada_sin_columnas, 'City', 'ZIPAQUIRÁ', 'ZIPAQUIRA')
change_value(Data_ajustada_sin_columnas, 'City', 'BOGOTA D.C.', 'BOGOTA')
change_value(Data_ajustada_sin_columnas, 'City', 'BUGALAGRANDE', 'BUGA')
change_value(Data_ajustada_sin_columnas, 'City', 'Barranquilla', 'BARRANQUILLA')
change_value(Data_ajustada_sin_columnas, 'City', 'CALDAS - ANTIOQUIA', 'MANIZALES')
change_value(Data_ajustada_sin_columnas, 'City', 'MANIZALEZ', 'MANIZALES')
change_value(Data_ajustada_sin_columnas, 'City', 'ENVIGADO', 'MEDELLIN')

'We are going to take only the columns that we believe are the ones that describe the data'
'We need to clean the Location and the Division variables'
change_value(Data_ajustada_sin_columnas, 'Location' , 'EPD-CLO','HSC-CLO')
change_value(Data_ajustada_sin_columnas, 'Location' , 'EPD-BAQ','HSC-BAQ')
change_value(Data_ajustada_sin_columnas, 'Location' , 'EFP-BOG','HSC-BOG')
change_value(Data_ajustada_sin_columnas, 'Location' , 'EPD-MED','HSC-MED')
change_value(Data_ajustada_sin_columnas, 'Location' , 'HSC-BUC','HSC-BOG')

'We need to change the name of some of the columns in order to be able to delete them'
Data_ajustada_sin_columnas.columns = ['index', 'Division', 'Notification', 'Task', 'Notifictn type',
'Notif.date', 'City', 'Notif. Time', 'MaintPlant', 'Description1', 'Description2', 'Priority1', 'Priority2', 'Sort number', 'Task code', 'Task code text', 'System status', 'Task group text', 'Task processor', 'Created by', 'Created at', 'Reference date', 'Completed By',
'Task text', 'DescEmpl.Resp.', 'Effect', 'Functional Loc.', 'Description3', 'Planned start', 'Planned time1', 'Planned time2', 'Material', 'Description4', 'Req. start', 'Req. start time',
'Req. end time', 'Sales Document', 'Item', 'Sales Order',
'Serial Number', 'Equipment', 'EquipmtAffctd', 'Main WorkCtr', 'Location', 'Customer', 'Item']
'We need to clean the Division fields. We see that all of them have disorganized values'
'We need to separate the column Description 1 to the first word in order to see which is the Division of the equipment'
'We remove the first space in order to do not loose data'

Data_ajustada_sin_columnas['Description5'] = Data_ajustada_sin_columnas['Description1'].str.split().str.get(0)
'Values to drop of data frame = Description1'
Drop_values = [] # Confidential 

for column in Drop_values:
	Data_ajustada_sin_columnas = delete_rows(Data_ajustada_sin_columnas, 'Description5', column)
	Data_ajustada_sin_columnas = Data_ajustada_sin_columnas.reset_index(drop = True)
	Key_Strings_1 = [] #Confidential
	Data_ajustada_sin_columnas['Description5'] = Data_ajustada_sin_columnas['Description1'].str.split().str.get(0)
'alues to drop of data frame = Description1'

Drop_values = [] #Condifential 
for column in Drop_values:
		
	'Generation of Division'
	Division = pd.DataFrame(Key_Strings_1,Key_Strings_2) counter = 0

for column in Key_Strings_1:
	change_value(Data_ajustada_sin_columnas,'Description5', column, Key_Strings_2[counter])
	counter = counter +1
	'After we see the data is most clean as possible, we take the columns that we think will give us information'
	Data = Data_ajustada_sin_columnas.loc[:,['Notifictn type','Year', 'Month', 'Week','Activity Time','City','Equipment','Material','Customer',
	'Location','Description5','Description1', 'Effect','DescEmpl.Resp.','Priority2', 'System status','Main WorkCtr']]
	'Correlation Analysis of variables. All columns are categorical' counter = 0
	NA_columns = pd.DataFrame(columns = ['column'])
	Labels = []
	Column_labels = []

for column in Data.columns:
	value = Data.loc[:,column].isna().sum() if value > 0:
	NA_columns.loc[counter,'column'] = column
	counter = counter + 1
	na_values = Data[[column]][Data[column].isnull()] not_na_values = Data[[column]][Data[column].notnull()] label_value = label_encoder(not_na_values,column)
	'Generation of Division'
	Division = pd.DataFrame(Key_Strings_1,Key_Strings_2) counter = 0

for column in Key_Strings_1:
	change_value(Data_ajustada_sin_columnas,'Description5', column, Key_Strings_2[counter])
	counter = counter +1
	'After we see the data is most clean as possible, we take the columns that we think will give us information'
	Data = Data_ajustada_sin_columnas.loc[:,['Notifictn type','Year', 'Month', 'Week','Activity Time','City','Equipment','Material','Customer',
	'Location','Description5','Description1', 'Effect','DescEmpl.Resp.','Priority2', 'System status','Main WorkCtr']]
	'Correlation Analysis of variables. All columns are categorical' counter = 0
	NA_columns = pd.DataFrame(columns = ['column'])
	Labels = []
	Column_labels = []

for column in Data.columns:
	value = Data.loc[:,column].isna().sum() if value > 0:
	NA_columns.loc[counter,'column'] = column
	counter = counter + 1
	na_values = Data[[column]][Data[column].isnull()] not_na_values = Data[[column]][Data[column].notnull()] label_value = label_encoder(not_na_values,column)

#%% Voting classifier
'Now we are going to do two prediction models. Voting and ANN ' 'We are going to take the variables that describe our objective variable'
X = Data.loc[:,['Activity Time','City', 'Equipment','Material','Customer','Description5','Description1','Effect', 'DescEmpl.Resp.','Priority2']]
y = Data.loc[:,['Notifictn type']]
'''
index = Column_labels.index('Notifictn type')
label_value = Labels[index]
y = label_value.inverse_transform(y)
'''
'Divide train and data test'
Xtrain , Xtest , Ytrain , Ytest = train_test_split(X, y , test_size = 0.25)
sc = StandardScaler()
Xtrain = sc.fit_transform(Xtrain)
Xtest = sc.transform(Xtest)

'Create voting classifier:'
clf_log = LogisticRegression(multi_class='ovr', random_state=1)
clf_random = RandomForestClassifier(n_estimators=25, random_state=1)
clf_voting = VotingClassifier(estimators=[('Logistic', clf_log), ('Random', clf_random)], voting='soft')
clf_voting = clf_voting.fit(Xtrain, Ytrain)
Ypred = clf_voting.predict(Xtest)
Report_model = classification_report(Ypred, Ytest) #%% Model evaluation
print(Report_model)
cf_matrix = confusion_matrix(Ytest,Ypred) sns.heatmap( (cf_matrix), annot=True,
fmt='g', cmap='Blues')

#%% Neural Network classifier
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu')) ann.add(tf.keras.layers.Dense(units = 6, activation = 'relu')) ann.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
ann.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"]) ann.fit(Xtrain, Ytrain, batch_size = 32 , epochs = 50)
#%% Cross Valitadion
print(cross_val_score(clf_voting, Xtest, Ytest, cv=5))
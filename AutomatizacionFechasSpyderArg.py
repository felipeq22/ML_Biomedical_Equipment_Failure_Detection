# -*- coding: utf-8 -*-

'Proyecto Maestria Universidad Catolica de Avila'

#%% Libraries

'We start by importing some libraries'
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import timedelta
from pandas.tseries.offsets import BDay
nan_value = ("NaN")
import time
import os.path
import sys
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
#%% Upload data bases 

"In the next segment of code we are going to import the Data"
Fields = list(range(0,70))
Fields = [str(i) for i in Fields]
Data_IP24 = pd.read_csv("/Users/felipe_q/Desktop/Proyecto Automatizacion UCAV/Proyecto Automatizacion/Proyecto Automatizacion/IP24.csv", encoding = 'latin-1', sep = ';', engine = "python", names=Fields)
Header = Data_IP24.iloc[0,:]
Header = Header.values.tolist()
Data_IP24 = Data_IP24[1:]
Data_IP24.columns = Header
Fields = list(range(0,45))
Fields = [str(i) for i in Fields]
Data_IW67 = pd.read_csv("/Users/felipe_q/Desktop/Proyecto Automatizacion UCAV/Proyecto Automatizacion/Proyecto Automatizacion/IW67.csv", encoding = 'latin-1', sep = ';', engine = "python", names=Fields)
Header = Data_IW67.iloc[0,:]
Header = Header.values.tolist()
Data_IW67 = Data_IW67[1:]
Data_IW67.columns = Header
Fields = list(range(0,10))
Fields = [str(i) for i in Fields]
Data_IW75 = pd.read_csv("/Users/felipe_q/Desktop/Proyecto Automatizacion UCAV/Proyecto Automatizacion/Proyecto Automatizacion/IW75.csv", encoding = 'latin-1', sep = ';', engine = "python", names=Fields)
Header = Data_IW75.iloc[0,:]
Header = Header.values.tolist()
Data_IW75 = Data_IW75[1:]
Data_IW75.columns = Header

#%% Functions declaration
'La ciudad Bogota es la misma que Bogota D.C.'
def change_value(Data_Frame,column, String1, String2):
    
    Data_Frame.loc[:,column]  = np.where((Data_Frame.loc[:,column] == String1), String2, Data_Frame.loc[:,column])
    
def label_encoder (Data_Frame, String):
    
    lb = LabelEncoder()
    Data_Frame.loc[:,String] = lb.fit_transform(Data_Frame.loc[:, String])
    return lb

def delete_rows(Data_Frame, Column_1 ,String):
    
    
    Data_Frame = Data_Frame[Data_Frame[Column_1] != String]
    return Data_Frame

#%% Data - Cleaning"Limpieza de datos"
Data_IW67['Planned start'] = Data_IW67['Planned start'].astype(str)

"Luego del análisis realizado, evidenciamos que las variables que nos pueden aportar a nuestro modelo de ML"
"son las variables de Tipo de Equipo, Material del equipo, Ciudad y Ubicación de donde proviene el ingeniero para atender el equipo"

"Se eliminan los campos de las fechas que no nos sirven"
    
for i in range(1,len(Data_IW67)):
    if ('-'  in Data_IW67.loc[i,"Planned start"]) or ('n'  in Data_IW67.loc[i,"Planned start"]) :
        Data_IW67 = Data_IW67.drop(index = i)

"Reiniciamos los index del data frame"       
Data_IW67 = Data_IW67.reset_index()

"Pasamos las variables strings a tipo date"
for i in range(1,len(Data_IW67)):
    Date = Data_IW67.loc[i, 'Planned start']
    Data_IW67.loc[i,"Planned start"] = datetime.strptime(Date, '%d/%m/%Y')    

'We only take prevetive and corrective maintenances'
Data_ajustada = Data_IW67.loc[(Data_IW67["Notifictn type"] == "MM") | (Data_IW67["Notifictn type"] == "MS")]
Data_ajustada_sin_columnas = Data_ajustada.drop_duplicates(subset = "Notification", keep ='last')

'Proceso para limpieza de datos. Esto teniendo en cuenta que hay ciudades que pueden ser diferentes'
  
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
       'Notif.date', 'City', 'Notif. Time', 'MaintPlant', 'Description1',
       'Description2', 'Priority', 'Priority', 'Sort number', 'Task code',
       'Task code text', 'System status', 'Task group text', 'Task processor',
       'Created by', 'Created at', 'Reference date', 'Completed By',
       'Task text', 'DescEmpl.Resp.', 'Effect', 'Functional Loc.',
       'Description3', 'Planned start', 'Planned time', 'Planned time',
       'Material', 'Description4', 'Req. start', 'Req. start time',
       'Req. end time', 'Sales Document', 'Item', 'Sales Order',
       'Serial Number', 'Equipment', 'EquipmtAffctd', 'Main WorkCtr',
       'Location', 'Customer', 'Item']

#%%
'We need to clean the Division fields. We see that all of them have disorganized values'
'We need to separate the column Description 1 to the first word in order to see which is the Division of the equipment'
'We remove the first space in order to do not loose data'

Data_ajustada_sin_columnas['Description1'] = Data_ajustada_sin_columnas['Description1'].str.split().str.get(0)

'Values to drop of data frame = Description1'

Drop_values = ['SERVICIO','Patient','COROSKOP','MULTIMOBIL', 'MULTISTAR','MAGICVIEW','MAMMOTEST','MAGICWEB','VERTIX','WH',
               'GENERATOR', '6/18MV','87' ,'LEONARDO','X-LEONARDO','ANALYZER','MODULARIS']

for column in Drop_values:
    
    Data_ajustada_sin_columnas = delete_rows(Data_ajustada_sin_columnas, 'Description1', column)  

'PASAR TODOS LOS DE ESA COLUMNA A STRIN'
#%%

Key_Strings_1 = ['SOMATOM' , 'MAGNETOM', 'ACUSON', 'AXIOM','MULTIX', 'Eclipse', 'ARCADIS','ARTIS','SIREGRAPH','MAMMOMAT', 'ONCOR'          ,  'PRIMUS',  'LUMINOS',
               'Symbia'   , 'SIREMOBIL' , 'E.CAM' , 'POLYMOBIL' ,'LITHOSTAR-MULTILINE' , 'Signature' , 'Symbia_T6' ,  'MEVATRON', 'POWERMOBIL'  , 'ARTISTE' , 'BIOGRAPH',
               'LITHOSTAR' , 'SONOLINE' , 'MOBILETT' ,'Artis', 'syngo', 'SYNGO' , 'SYNGO.PLAZA' ,
               'SYNGO.VIA' , 'syngo.via' , 'Ysio' , 'MammoTest'  , 'Symbia_T' , 'VERTIX', 'Mevatron','C.CAM','System',
                'SIGNATURE', 'ANGIOSTAR', 'LITHOSKOP', 'e.cam' , 'LEONARDO' ,'LANTIS','X-LEONARDO', 'EXPLORA' ,'ASM',
               'ANALYZER', 'MOSAIQ', 'RL1265', 'Cios' ,'P300', 'Luminos', 'Multix' , 'SYMBIA', 'syngo.plaza', 'Mammomat', 'CIOS',
               'MODULEAF' , 'e.soft' ,'RAPIDPOINT' , 'teamplay' , 'ADVIA'     ,'VersaCell'  , 'EPOC' ,'SENSIS' , 'Atellica', 'Symbia.net',
               '348', 'RAPIDPoint' , 'E.SOFT' , 'SIRESKOP', 'POLYMOBIL']
                 
Key_Strings_2 = ['CT' ,        'MR',       'US',   'AT',   'XP',      'CICLOTRON', 'AXA',    'AT',  'XP',      'XP'   ,  'RO',  'RO'    , 'XP',
               'MI-SPECT' ,  'XP'   ,   'CICLOTRON' ,'XP'   , 'AXA'                 ,  'MI-SPECT'   ,   'MI-SPECT',    'RO'     , 'XP'         , 'RO'       , 'MI-SPECT',
               'AXA'      ,   'US'    , 'XP'        ,'AT',        'SY'    , 'SY'     ,'SY'      , 
               'SY'        , 'SY'        , 'XP'    , 'XP'        ,   'MI-SPECT',        'XP'   , 'RO'      , 'CICLOTRON','SY',
               'MI-SPECT' ,        'AT'      , 'AXA'     , 'CICLOTRON' ,  'CT'    ,  'RO'  , 'MR' , 'CICLOTRON', 'CICLOTRON',
               'COAG', 'RO'       , 'POC'    ,  'AXA'   ,'US'  ,'XP'   , 'XP'     , 'MI-SPECT'      , 'SY'        ,'XP', 'AXA',
               'RO'      , 'MI-SPECT' , 'POC' ,      'SY'      , 'HEMATO'   , 'AUTOMAT'      , 'POC'  ,'AT' , 'ATELLICA' , 'MI',
               'POC' , 'POC'        , 'MI-SPECT', 'XP' ,'XP']


'System'

Revision = pd.DataFrame(columns = ['EQUIPO','REFERENCIA','PKD','PRIMEVIEW','SIMVIEW'])
Revision['EQUIPO'] = Key_Strings_1
Revision['REFERENCIA'] = Key_Strings_2

'HACER UN FILTRO CON LAS SEGUNDAS PALABRAS'


'''
'MD'- RUIDO, 'MH - RUIDO', 'MI - RUIDO', 'MB-RUIDO', 'MF', 'MA', 'MG', 'MQ', 'M1', 'MZ',
       'MR', '50', 'MV'
'''

'We are going to delete the nan values'

Data_ajustada_sin_columnas = Data_ajustada_sin_columnas.dropna()


'''

'Here we do the categorization of the variablfses'
'A city y a material no se le hace label encoder'

Label_encoders = []

for Column in Data_ajustada_sin_columnas.columns:
    
    lb = label_encoder(Data_ajustada_sin_columnas, Column)
    Label_encoders.append(lb)

#%%
" Se quiere probar si se puede predecir si la proxima actividad de un equipo sera preventiva o correctiva de manera que haremos una prediccion"

X = Data_ajustada_sin_columnas[['City','Division','Material','Location']]
y = Data_ajustada_sin_columnas[['Notifictn type']]

MM = y[y.loc[:,'Notifictn type'] == 'MM']
MS = y[y.loc[:,'Notifictn type'] == 'MS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

classifier = ComplementNB()
classifier.fit(X_train,y_train)

prediction = classifier(X_test)
Accuracy_NB = accuracy_score(prediction, y_test)


'''
#SE CARGAN LOS DIAS FERIADOS EN COLOMBIA
FestivosColombia = pd.read_csv("C:\\Users\\Felipe Q\\Desktop\\Proyecto Automatizacion\\Dias_no_trabajo.csv")
#FestivosColombia = pd.read_csv("D:\\temp\Dias_no_trabajo.csv")
FestivosColombia = pd.DataFrame(FestivosColombia)
#SE LIMPIAN LOS DATOS DEL ARCHIVO
FestivosColombia = FestivosColombia.iloc[:,0]
FestivosColombiaClean = FestivosColombia

#CICLO PARA LIMPIAR LOS DATOS SUCIOS DEL ARCHIVO 
for i in range(len(FestivosColombia)):
    "print(i)"
    FestivosColombiaClean[i] = FestivosColombia[i].replace(';',"")
    if FestivosColombiaClean[i] == "":
        
        FestivosColombiaClean[i] = FestivosColombiaClean[i].replace('',nan_value)

#SE CARGAN LOS ARCHIVOS REQUERIDOS BAJADOS DE SAP/ DEPENDIENDO DE LA UBICAION DEL COMPUTADOR SE CAMBIA LA RUTA

Data_Base = pd.read_csv("C:\\Users\Felipe Q\Desktop\Proyecto Automatizacion\IW75_CHA_00.txt", sep = " \t ",
                        engine = 'python',encoding = 'utf16')

Data_Base2 = pd.read_csv("C:\\Users\Felipe Q\Desktop\Proyecto Automatizacion\IW67B_00.txt", sep = " \t ",
                        engine = 'python',encoding = 'utf16')


Data_Base3 = pd.read_csv("C:\\Users\Felipe Q\Desktop\Proyecto Automatizacion\IP24_00.txt", sep = " \t ",
                        engine = 'python',encoding = 'utf16')

'''
'''
Data_Base = pd.read_csv("D:\\Temp\IW75_CHA_00.txt", sep = " \t ",
                        engine = 'python',encoding = 'utf16')

Data_Base2 = pd.read_csv("D:\\Temp\IW67B_00.txt", sep = " \t ",
                        engine = 'python',encoding = 'utf16')

#Data_Base3 = pd.read_csv("D:\\Temp\IP24_00.txt", sep = " \t ",
                      #  engine = 'python',encoding = 'utf16')
'''
'''

# SE PROCEDE A GENERAR EL DATA FRAME DE LA IW75
Size_IW75 = len(Data_Base)
#Creacion de DataFrames
String = Data_Base.iloc[0]
Conversion_String = String.tolist()
Valor_String = Conversion_String[0]
Lista_De_Strings = Valor_String.split('\t')
IW75 = pd.DataFrame([Lista_De_Strings], columns = ["Equipment" , "Cont.Start", "Cont.End", "AWV Status" , "Response T" , "Sales Doc.", "Item", "Soldto pt", "Bill-to Pa", "Payer", "Ship-to","Material Description","MaterialDescription","Sorg" ])

for i in range(1,Size_IW75):
    String = Data_Base.iloc[i]
    Conversion_String = String.tolist()
    Valor_String = Conversion_String[0]
    Lista_De_Strings = Valor_String.split('\t')
    IW75.loc[len(IW75.index)] = Lista_De_Strings


#SE PROCEDE A GENERAR EL DATA FRAME DE LA IW67

#Se crea el segundo data frame
Size_IW67 = len(Data_Base2)
#Creacion de Data Frames
String = Data_Base2.iloc[0]
Conversion_String = String.tolist()
Valor_String = Conversion_String[0]
Lista_De_Strings = Valor_String.split('\t')
IW67 = pd.DataFrame([Lista_De_Strings], columns = ["Code" , "Noti", "Task Text","task", "Plnd start" , "PTM" , "Pl.finish", "PTM", "Created On"	,"CreatTme",	"Completed",	"CompTime",	"DescEmployeeResp.for job",	"PartResp.",	"SysStatus" ])

for i in range(1,Size_IW67):
    String = Data_Base2.iloc[i]
    Conversion_String = String.tolist()
    Valor_String = Conversion_String[0]
    Lista_De_Strings = Valor_String.split('\t')
    IW67.loc[len(IW67.index)] = Lista_De_Strings


#SE PROCEDE A GENERAR EL DATA FRAME DE LA IP24

Size_IP24 = len(Data_Base3)
#Creacion de Data Frames

String = Data_Base3.iloc[0]
Conversion_String = String.tolist()
Valor_String = Conversion_String[0]

Lista_De_Strings = Valor_String.split('\t')
IP24 = pd.DataFrame([Lista_De_Strings], columns = ["CoCd",	"MntPlan", 	"Maintenance item description",	"Object Description","	Equipment",	"Serial No.",	"Sales Doc.","Group" ])

for i in range(1,Size_IP24):
    String = Data_Base3.iloc[i]
    Conversion_String = String.tolist()
    Valor_String = Conversion_String[0]
    Lista_De_Strings = Valor_String.split('\t')
    IP24.loc[len(IP24.index)] = Lista_De_Strings


#CREAMOS PROCESO DE GENERACION DE PREVENTIVOS

#SE REFERNCIA UN CONTRATO DE EJEMPLO PARA EVALUAR FUNCIONAMIENTO DEL CODIGO
SalesDocument = "2600009556" #Columna 6 de IW75
#ESTE ITEM SE DEBE DESBLOQUEAR CUANDO LE PASEMOS PARAMETROS AL .EXE
#SalesDocument = SLA

#EN ESTE PROCESO CREAMOS EL .TXT QUE SERÁ ALIMENTADO AL ROBOT
path = "D:\\Temp"
Direccion_Documento = SalesDocument + ".txt" 
#Direccion_Documento = ArchivoPlanFinal
completeName = os.path.join(path, Direccion_Documento)
Documento_Variables = open(Direccion_Documento, "w+")

#INICIALIZACION DE VARIABLES
FechasPreventivas = []
FechasPreventivastxt =[]
FrecuenciaPreventiva = 2 #ENTRADA POR PARTE DEL ROBOT
#InicioContrato = '1/1/2021' #Entrada 
DuracionPreventivo = 4
#FinalContrato = '31/12/2021' #Entrada duracion de contrato

#AQUI SACAMOS EL INIICIO Y FIN DEL CONTRATO BASADO EN EL SLA DADO

for i in range(0,len(IW75)):
               
               ActualContract = IW75.iloc[i,6]
               
               if ActualContract == SalesDocument:
                   
                   InicioContrato = IW75.iloc[i,2]
                   FinalContrato = IW75.iloc[i,3]
                   
                   break
               
               else:
                   ActualContract = 'NA'

#OBTENEMOS LOS EQUIPOS QUE ESTAN BAJO LA FRECUENCIA DE ESE EQUIPO

Equipments= []
Frequencies = []

for i in range(0, len(IP24)):
    
    if ActualContract == IP24.iloc[i,6]:
        
        Equipments.append(IP24.iloc[i,4])

Equipments = list(dict.fromkeys(Equipments))

# A CONTINUACION OBTENEMOS LA FRECUENCIA DE CADA PREVENTIVO PARA CADA EQUIPMENT
for i in range(0, len(Equipments)):
 
    for j in range(0, len(IP24)):
        
        if Equipments[i] == IP24.iloc[j,4]:
            
            String_Frequency = IP24.iloc[j,7]
            Frequency = int(String_Frequency[5])
            Frequencies.append(Frequency)
            break



#LAS FECHAS DE INICIO Y FIN DE CONTRATO LAS PASAMOS A TIPO DATE
InicioContrato = InicioContrato.replace('.','/')
FinalContrato = FinalContrato.replace('.','/')   
InicioContrato = datetime.strptime(InicioContrato, '%d/%m/%Y') #Esto lo toma del ROBOT ENTRADA 
FinalContrato = datetime.strptime(FinalContrato,'%d/%m/%Y') #Fecha que no aporta valor al código
DuracionContratoAnios = FinalContrato - InicioContrato # ENTRADA DEL ROBOT
DuracionContratoAnios = (DuracionContratoAnios+timedelta(1))/365
DuracionContratoAnios = DuracionContratoAnios.days

#SE CREAN LAS FECHAS DE LOS NUEVOS PREVENTIVOS BASADOS EN LA FRECUENCIA DE LOS PREVENTIVOS POR EQUIPO

for i in range(0, len(Equipments)):
    
    FrecuenciaPreventiva = Frequencies[i]
    "Anual = 1 vez al año"
    "Semestral = 2 veces al año" 
    "Trimestral = 4 Veces al año"  
    "Cuatrimestral = 3 veces al año"
    if FrecuenciaPreventiva == 1: 
        Dias = 261 
    elif FrecuenciaPreventiva == 6: 
        Dias = 130 
    elif FrecuenciaPreventiva == 4: 
        Dias = 65 
    elif FrecuenciaPreventiva == 3: 
        Dias = 87 

    #Creación de fechas del contrato
    ContadorAnios = 1
    ContadorFechas = 0
    while ContadorAnios <= DuracionContratoAnios:
    
        for j in range(FrecuenciaPreventiva):
        
            print(ContadorFechas)
            if ContadorFechas == 0:
            
                FechasPreventivas.append( InicioContrato + BDay(Dias))
            else:
                FechasPreventivas.append( FechasPreventivas[ContadorFechas-1] + BDay(Dias))
        
            ContadorFechas = ContadorFechas +1
        
        ContadorAnios = ContadorAnios + 1

"Revisión de fechas no sean festivos."

for i in range(len(FechasPreventivas)):
    FechaTest = FechasPreventivas[i]
    
    for j in range(len(FestivosColombia)):
        
        if FechaTest == FestivosColombia[j]:
            
            FechasPreventivas[i] = FechaTest + BDay(1) 

Counter = 0


#SE PROCEDE A GENERAR EL .TXT POR CADA EQUIPO

DuracionPreventivo = 4
InicioContrato = InicioContrato.strftime("%d.%m.%Y")
FinalContrato = FinalContrato.strftime("%d.%m.%Y")
InicioContrato = InicioContrato + "\n"
DuracionPreventivo = str(DuracionPreventivo)
DuracionPreventivo = DuracionPreventivo +"\n"
for i in range(0,len(Equipments)):
    Equipment = "*" + Equipments[i] + "\n" 
    Documento_Variables.write(Equipment)
    Documento_Variables.write(InicioContrato)
    Documento_Variables.write(DuracionPreventivo)
    Counter2 = 1
    while Counter2 <= (len(FechasPreventivas)/len(Equipments)):

        FechaStr= FechasPreventivas[Counter].strftime("%m.%d.%Y \n")
        #FechaStr= FechasPreventivas[i].strftime("%d.%m.%Y")
        FechasPreventivastxt.append(FechaStr)
        Documento_Variables.write(FechaStr)
        Counter = Counter + 1
        Counter2 = Counter2 + 1
    
    InicioContrato2 = '1/1/2021' #Entrada 
   

Documento_Variables.close()
#Recibir Datos
#Entregar el Arrelgo
#EQUIPMENT BAJO CONTRATO ENTRADA.
#HISTORIAL DE PREVENTIVOS  
#Parte intelingente

#x = input('Ingresar Duración Contrato')
'''

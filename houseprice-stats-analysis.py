# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-09-29T18:41:05.346699Z","iopub.execute_input":"2024-09-29T18:41:05.347218Z","iopub.status.idle":"2024-09-29T18:41:05.781072Z","shell.execute_reply.started":"2024-09-29T18:41:05.347078Z","shell.execute_reply":"2024-09-29T18:41:05.779766Z"}}


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
print ("--Start")
for dirname, _, filenames in os.walk('c:/temp'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


print ("End***")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-09-29T18:46:19.025290Z","iopub.execute_input":"2024-09-29T18:46:19.025704Z","iopub.status.idle":"2024-09-29T18:46:19.128232Z","shell.execute_reply.started":"2024-09-29T18:46:19.025666Z","shell.execute_reply":"2024-09-29T18:46:19.127077Z"}}
import seaborn as sns
import matplotlib.pyplot as plt


# Cargar los datos
dir_Input='c:/temp'
dir_Output='c:/temp'
file='train.csv'
output='temp.csv'
df_train = pd.read_csv(dir_Input+file)  # Sustit#uye por tu dataset

#df['Alley'].fillna('Desconocido', inplace=True)
#df.fillna({'PoolQC': 'Non#e'}, inplace=True)
df=df_train.copy()

df.fillna({'MSZoning': 'UnKnow','LotFrontage':'UnKnow','MasVnrArea':'UnKnow',
          'Street': 'UnKnow','Alley': 'UnKnow','MSZoning': 'UnKnow','LotShape': 'UnKnow','LandContour': 'UnKnow','Utilities': 'UnKnow',
          'LotConfig': 'UnKnow','LandSlope': 'UnKnow','Neighborhood': 'UnKnow','Condition1': 'UnKnow','Condition2': 'UnKnow','BldgType': 'UnKnow',
          'HouseStyle': 'UnKnow','RoofStyle': 'UnKnow','RoofMatl': 'UnKnow','Exterior1st': 'UnKnow','MasVnrType': 'UnKnow',
          'ExterQual': 'UnKnow','ExterCond': 'UnKnow','Foundation': 'UnKnow','BsmtQual': 'UnKnow','BsmtCond': 'UnKnow',
          'BsmtExposure': 'UnKnow','BsmtFinType1': 'UnKnow','BsmtFinType2': 'UnKnow','Heating': 'UnKnow','HeatingQC': 'UnKnow',
          'CentralAir': 'UnKnow','Electrical': 'UnKnow','KitchenQual': 'UnKnow','Functional': 'UnKnow','FireplaceQu': 'UnKnow',
          'GarageType': 'UnKnow','GarageFinish': 'UnKnow','GarageQual': 'UnKnow','GarageCond': 'UnKnow','PavedDrive': 'UnKnow',
          'Fence': 'UnKnow','MiscFeature': 'UnKnow','MoSold': 'UnKnow','SaleType': 'UnKnow','SaleCondition': 'UnKnow',}, inplace=True)




#df = df[df['MasVnrArea'].notna()]
#df = df[df['Electrical'].notna()]
#df = df[df['LotFrontage'].notna()] 

                                  
#Convierte Categorical en Numerical 1/0 - 
#df_encoded = pd.get_dummies(df, columns=['GarageType','GarageFinish','GarageQual','GarageCond'])
#df_encoded = pd.get_dummies(df, columns=['PoolQC','Fence','MiscFeature'])
#df_encoded = pd.get_dummies(df, columns=['Street','Alley','Utilities','CentralAir','GarageType'])
df_encoded = pd.get_dummies(df, columns=['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
                                         'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',
                                         'Exterior1st','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',
                                         'BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual',
                                         'Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive',
                                         'Fence','MiscFeature','MoSold','SaleType','SaleCondition'])
categorical_columns = df_encoded.filter(like='_')
#print(categorical_columns)
# Configurar para mostrar todas las filas y columnas
pd.set_option('display.max_rows', 10)   # Mostrar todas las filas
pd.set_option('display.max_columns', 10)  # Mostrar todas las columnas

print(df_encoded.filter(like='_'))

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-09-29T18:46:22.208568Z","iopub.execute_input":"2024-09-29T18:46:22.208955Z","iopub.status.idle":"2024-09-29T18:46:22.358439Z","shell.execute_reply.started":"2024-09-29T18:46:22.208919Z","shell.execute_reply":"2024-09-29T18:46:22.357321Z"}}
#Creo una nueva medida

print(df['YearBuilt'])

df['YearsBuiltSold'] = df['YrSold'] - df['YearBuilt']
df['YearsRemoSold'] = df['YrSold'] - df['YearRemodAdd']
df['YearsGarageBuiltSold'] = df['YrSold'] - df['GarageYrBlt']

df.fillna({'YearsBuiltSold': 999}, inplace=True)
df.fillna({'YearsRemoSold': 999}, inplace=True)
df.fillna({'YearsGarageBuiltSold': 999}, inplace=True)
                                  
df.drop(columns=['Exterior2nd','YearBuilt','YearRemodAdd','GarageYrBlt'], axis=1,inplace=True)   

cols_numeriques=df[['Id','MSSubClass','LotArea','LotFrontage','OverallQual','OverallCond','MasVnrArea','BsmtFinSF1',
                    'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath',
                    'FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF',
                    'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YrSold','YearsBuiltSold','YearsRemoSold',
                    'YearsGarageBuiltSold','SalePrice']]

#,'YearsBuiltSold','YearsGarageBuiltSold']]
train_data=pd.concat([categorical_columns,cols_numeriques], axis=1)
# Guardar el DataFrame en un archivo CSV
train_data.to_csv(dir_Output+output, index=False)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2024-09-29T18:46:38.720321Z","iopub.execute_input":"2024-09-29T18:46:38.720701Z","iopub.status.idle":"2024-09-29T18:46:39.661834Z","shell.execute_reply.started":"2024-09-29T18:46:38.720667Z","shell.execute_reply":"2024-09-29T18:46:39.660468Z"}}
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.pipeline import make_pipeline


# Estandarización de los datos después de PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(train_data)

#Eliminar columnas altamente correlacionadas

correlation_matrix = train_data.corr().abs()

upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool_))

to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
train_data.drop(columns=to_drop, inplace=True)

to_drop = [column for column in upper.columns if any(upper[column] < -0.95)]
train_data.drop(columns=to_drop, inplace=True)

# %% [code] {"execution":{"iopub.status.busy":"2024-09-29T18:46:48.096138Z","iopub.execute_input":"2024-09-29T18:46:48.096552Z","iopub.status.idle":"2024-09-29T18:46:48.595677Z","shell.execute_reply.started":"2024-09-29T18:46:48.096516Z","shell.execute_reply":"2024-09-29T18:46:48.594403Z"},"jupyter":{"outputs_hidden":false}}
# Calcular la matriz de correlación
train_data['SalePrice']=df['SalePrice']
corr_matrix = train_data.corr()

# Ver las correlaciones con la variable objetivo (supongamos que es 'target_column')
corr_target = corr_matrix['SalePrice'].sort_values(ascending=False)

pd.set_option('display.max_rows', None)   # Mostrar todas las filas
pd.set_option('display.max_columns', None)  # Mostrar todas las columnas

# Print completo del DataFrame

print(corr_target)

#elimina automaticamente las columnas que VIF es >10
to_drop = corr_target[corr_target < 0].index.tolist()
print(to_drop)
train_data.drop(columns=to_drop, inplace=True)

correlation_matrix.to_csv(dir_Output+'matriz_correlacion', index=False)

# %% [markdown]
# 

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Importar las bibliotecas necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np  # Importar NumPy para cálculos

# Asignar variable dependiente e independiente
y = df['SalePrice']
train_data.drop(columns=['SalePrice'], axis=1,inplace=True) 
x = train_data

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# Crear el modelo de Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # Puedes ajustar n_estimators

# Mostrar las columnas que tienen valores NaN
nan_columns = X_train.columns[X_train.isna().any()]

# Imprimir el listado de las columnas con NaN
print("Columnas con NaN:", nan_columns)

# Ajustar el modelo a los datos de entrenamiento
rf_model.fit(X_train, y_train)

# Hacer predicciones en los datos de prueba
y_pred = rf_model.predict(X_test)

# Evaluar el rendimiento del modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calcular el RMSD
rmsd = np.sqrt(mse)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')
print(f'Root Mean Square Deviation: {rmsd:.2f}')  # Imprimir RMSD

# Importancia de las características
importances = rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nImportancia de las características:")
print(feature_importance_df)

pd.set_option('display.max_columns', None)  # Mostrar todas las filas
print(list(X.columns))
pd.reset_option('display.max_columns')

# %% [code] {"execution":{"iopub.status.busy":"2024-09-29T10:56:42.597732Z","iopub.execute_input":"2024-09-29T10:56:42.598223Z","iopub.status.idle":"2024-09-29T10:56:42.731328Z","shell.execute_reply.started":"2024-09-29T10:56:42.598178Z","shell.execute_reply":"2024-09-29T10:56:42.729870Z"},"jupyter":{"outputs_hidden":false}}
import seaborn as sns
import matplotlib.pyplot as plt


# Cargar los datos
dir_Input='c:/temp'
dir_Output='c:/temp'
file_test='test.csv'
df_test = pd.read_csv(dir_Input+file_test)  # Sustit#uye por tu dataset

#df['Alley'].fillna('Desconocido', inplace=True)
#df.fillna({'PoolQC': 'Non#e'}, inplace=True)
df_new=df_test.copy()
df_submision=[]
df_new.fillna({'PoolQC': 'Desconegut'}, inplace=True)
df_new.fillna({'MasVnrType': 'UnKnow'}, inplace=True)
df_new = df_new[df_new['MasVnrArea'].notna()]
df_new = df_new[df_new['Electrical'].notna()]
df_new = df_new[df_new['LotFrontage'].notna()] 
                                  
#Convierte Categorical en Numerical 1/0 - 

df_new_categorical = pd.get_dummies(df_new, columns=['MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood',
                                         'Condition1','Condition2','BldgType','HouseStyle','RoofStyle','RoofMatl',
                                         'Exterior1st','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',
                                         'BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC','CentralAir','Electrical','KitchenQual',
                                         'Functional','FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive',
                                         'Fence','MiscFeature','MoSold','SaleType','SaleCondition'])

df_new_categorical.drop(columns=['Id','Street_Grvl','CentralAir_N','GarageType_Attchd','1stFlrSF','GrLivArea','MSZoning_RL','LotShape_IR1','LandContour_Lvl','LotConfig_Corner','LandSlope_Gtl','RoofMatl_Tar&Grv','RoofStyle_Gable',
                                 'RoofStyle_Hip','RoofMatl_CompShg','ExterCond_Ex','BsmtQual_Ex','BsmtQual_TA','BsmtExposure_No','Heating_OthW',
                                 'GarageQual_TA','GarageCond_TA','SaleCondition_Normal','OverallQual','Utilities_AllPub', 
                                   'Electrical_Mix','SaleCondition_Partial'], axis=1,inplace=True, errors='ignore')

#df_test.drop(columns=['Exterior2nd','YearBuilt','YearRemodAdd','GarageYrBlt'], axis=1,inplace=True)   

df_new_numeriques=df_test[['Id','MSSubClass','LotArea','LotFrontage','OverallQual','OverallCond','MasVnrArea','BsmtFinSF1',
                    'BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath',
                    'FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageCars','GarageArea','WoodDeckSF',
                    'OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','YrSold']]

#Creo una nueva medida
df_new_numeriques['YearsBuiltSold'] = df_test['YrSold'] - df_test['YearBuilt']
df_new_numeriques['YearsRemoSold'] = df_test['YrSold'] - df_test['YearRemodAdd']
df_new_numeriques['YearsGarageBuiltSold'] = df_test['YrSold'] - df_test['GarageYrBlt']

df_new_numeriques.fillna({'YearsBuiltSold': 999}, inplace=True)
df_new_numeriques.fillna({'YearsRemoSold': 999}, inplace=True)
df_new_numeriques.fillna({'YearsGarageBuiltSold': 999}, inplace=True)

df_new=pd.concat([df_new_categorical,df_new_numeriques], axis=1)

df_submision =df_new.loc[:, df_new.columns.intersection(X.columns)].copy()
df_submision = df_submision.loc[:, ~df_submision.columns.duplicated()]

print(df_submision)

# %% [code] {"execution":{"iopub.status.busy":"2024-09-29T10:56:53.068330Z","iopub.execute_input":"2024-09-29T10:56:53.068815Z","iopub.status.idle":"2024-09-29T10:56:53.292573Z","shell.execute_reply.started":"2024-09-29T10:56:53.068771Z","shell.execute_reply":"2024-09-29T10:56:53.291254Z"},"jupyter":{"outputs_hidden":false}}



# Configurar para mostrar todas las filas y columnas
pd.set_option('display.max_rows', 10)   # Mostrar todas las filas
pd.set_option('display.max_columns', None)  # Mostrar todas las columnas

print(df_submision)


# Imprimir las columnas que están en x_col pero no en df
missing_columns = [col for col in X.columns if col not in df_submision.columns]

print("Columnas que están en x_col pero no en df:")
print(missing_columns)
print(df_submision)

pd.set_option('display.max_rows', 10)   # Mostrar todas las filas
pd.set_option('display.max_columns', 10)  # Mostrar todas las columnas

for col in missing_columns:
    # Si el nombre de la columna contiene '_', se le asigna el valor False
    if '_' in col:
        df_submision[col] = False
    else:
        # Si no contiene '_', se le puede asignar otro valor o NaN
        df_submision[col] = 0  # O usa el valor que consideres necesario

#df_submision.drop(columns=['SalePrice'], axis=1,inplace=True) 
# Ver el DataFrame actualizado
print(df_submision)
# Guardar el DataFrame en un archivo CSV
df_submision.to_csv(dir_Output+'df_new.csv', index=False)

# %% [code] {"jupyter":{"outputs_hidden":false}}
########################Modelo de predicio--------------------
#####INPUT: df_submision - amb els camps que determinen el model-------------------

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Asumiendo que tienes un DataFrame 'df_new' con las mismas columnas que usaste para entrenar el modelo
# df_new contiene las características sin la columna de salida (por ejemplo, 'SalePrice')

df_submision_order = df_submision.reindex(columns=X_train.columns, fill_value=0)

# Realiza las predicciones con el modelo de Random Forest previamente entrenado
predictions = rf_model.predict(df_submision_order)

# Crear un DataFrame con los resultados de las predicciones
predictions_df = pd.DataFrame(predictions, columns=['Predicted_SalePrice'])

# Mostrar los primeros resultados
print(predictions_df.head())

# Si deseas guardar los resultados en un archivo CSV
predictions_df.to_csv('predicciones_rf.csv', index=False)
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, KBinsDiscretizer, MinMaxScaler

# Cargar el archivo CSV proporcionado por el usuario
file_path = r'C:\Users\s2dan\OneDrive\Documentos\WorkSpace\PrimerParcia_IA\ObesityDataSet_raw_and_data_sinthetic.csv'
data = pd.read_csv(file_path)

# 1. LabelEncoder: Aplicar a las columnas categóricas 'Gender' y 'NObeyesdad'
label_encoder = LabelEncoder()

# Aplicar LabelEncoder a las columnas categóricas
data['Gender'] = label_encoder.fit_transform(data['Gender'])
data['NObeyesdad'] = label_encoder.fit_transform(data['NObeyesdad'])

# 2. OneHotEncoder: Aplicar a las columnas categóricas restantes
categorical_columns = ['CALC', 'FAVC', 'SCC', 'SMOKE', 'family_history_with_overweight', 'CAEC', 'MTRANS']

# Cambiar 'sparse=False' por 'sparse_output=False'
onehot_encoder = OneHotEncoder(sparse_output=False)

# OneHotEncoder transforma las variables categóricas en variables binarias
encoded_columns = onehot_encoder.fit_transform(data[categorical_columns])

# Convertir los valores codificados en un DataFrame y agregar nombres de columnas
encoded_df = pd.DataFrame(encoded_columns, columns=onehot_encoder.get_feature_names_out(categorical_columns))

# Eliminar las columnas originales categóricas y agregar las columnas codificadas
data = data.drop(columns=categorical_columns)
data = pd.concat([data, encoded_df], axis=1)

# 3. Discretización: Dividir los valores numéricos en intervalos (por ejemplo, en 5 categorías)
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')

# Seleccionamos las columnas numéricas que vamos a discretizar
numerical_columns = ['Age', 'Height', 'Weight', 'CH2O', 'FAF', 'TUE']

# Aplicar discretización
discretized_data = discretizer.fit_transform(data[numerical_columns])

# Convertir los datos discretizados en un DataFrame y renombrar las columnas discretizadas
discretized_df = pd.DataFrame(discretized_data, columns=[col + "_discretized" for col in numerical_columns])

# Reemplazar las columnas originales por las discretizadas
data[numerical_columns] = discretized_df

# 4. Normalización: Escalar los atributos numéricos a un rango estándar (0 a 1)
scaler = MinMaxScaler()

# Aplicar normalización a las columnas numéricas
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Mostrar los datos preprocesados
print(data.head())

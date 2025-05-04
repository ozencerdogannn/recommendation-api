import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors
import pickle


data = pd.read_csv("D:/datas/projemldatasi.csv", delimiter=";")


data['ORDERQTY'] = data['ORDERQTY'].fillna(0)
data['LISTPRICE'] = data['LISTPRICE'].astype(str).str.replace(',', '.', regex=False).astype(float)
data.loc[data['LISTPRICE'] == 0, 'LISTPRICE'] = np.random.uniform(1000, 2000, size=(data['LISTPRICE'] == 0).sum())


color_encoder = LabelEncoder()
category_encoder = LabelEncoder()
data['COLOR_ENCODED'] = color_encoder.fit_transform(data['COLOR'].astype(str))
data['CATEGORYNAME_ENCODED'] = category_encoder.fit_transform(data['CATEGORYNAME'].astype(str))

data['COLOR_WEIGHTED'] = data['COLOR_ENCODED'] * 0.1
x = data[['LISTPRICE', 'CATEGORYNAME_ENCODED', 'COLOR_WEIGHTED', 'ORDERQTY']]
# x = data[['LISTPRICE', 'CATEGORYNAME_ENCODED', 'COLOR_ENCODED', 'ORDERQTY']]


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


knn = NearestNeighbors(n_neighbors=6, metric='euclidean')
knn.fit(x_scaled)


with open("model.pkl", "wb") as f:
    pickle.dump(knn, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("color_encoder.pkl", "wb") as f:
    pickle.dump(color_encoder, f)

with open("category_encoder.pkl", "wb") as f:
    pickle.dump(category_encoder, f)

data.to_csv("data_encoded.csv", index=False)
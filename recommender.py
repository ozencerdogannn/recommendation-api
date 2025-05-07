from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle


with open('model.pkl', 'rb') as f:
    knn = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

data = pd.read_csv("data_encoded.csv")


x = data[['LISTPRICE', 'CATEGORYNAME_ENCODED', 'COLOR_WEIGHTED', 'ORDERQTY']]
x_scaled = scaler.transform(x)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Veya spesifik olarak ["http://127.0.0.1:5500"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ProductRequest(BaseModel):
    index: int

@app.post("/recommendations/")
def get_recommendations(request: ProductRequest):
    index = request.index
    if index < 0 or index >= len(data):
        return {"error": "Geçersiz index."}

    product_vector = x_scaled[index].reshape(1, -1)
    distances, indices = knn.kneighbors(product_vector)

    recommended = data.iloc[indices[0][1:]][['PRODUCTID', 'NAME', 'LISTPRICE', 'CATEGORYNAME', 'COLOR']].fillna("Unknown")
    selected = data.loc[index][['PRODUCTID', 'NAME', 'LISTPRICE', 'CATEGORYNAME', 'COLOR']].fillna("Unknown")

    return {
        "selected_product": selected.to_dict(),
        "recommended_products": recommended.to_dict(orient='records')
    }



# @app.post("/recommendations/")
# def get_recommendations(request: ProductRequest):
#     index = request.index
#     if index < 0 or index >= len(data):
#         return {"error": "Geçersiz index."}

#     product_vector = x_scaled[index].reshape(1, -1)
#     distances, indices = knn.kneighbors(product_vector)

#     recommended = data.iloc[indices[0][1:]][['PRODUCTID', 'NAME', 'LISTPRICE', 'CATEGORYNAME', 'COLOR']]
#     selected = data.loc[index][['PRODUCTID', 'NAME', 'LISTPRICE', 'CATEGORYNAME', 'COLOR']]

#     return {
#         "selected_product": selected.to_dict(),
#         "recommended_products": recommended.to_dict(orient='records')
#     }
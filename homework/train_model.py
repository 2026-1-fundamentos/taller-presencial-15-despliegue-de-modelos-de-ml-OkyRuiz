"""Build, deploy and access a model using scikit-learn"""

import pickle

import pandas as pd  # type: ignore
from sklearn.linear_model import LinearRegression  # type: ignore

df = pd.read_csv("files/input/house_data.csv", sep=",")

features = df[         # Las X o variables independientes
    [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "condition",
    ]
]

target = df[["price"]] # Las Y o variable dependiente

estimator = LinearRegression() # Objeto que se va a entrenar, en este caso un modelo de regresión lineal
estimator.fit(features, target) # Para entrenar el modelo, se le pasan las X y las Y

with open("homework/house_predictor.pkl", "wb") as file: # Se abre un archivo en modo escritura binaria para guardar el modelo entrenado
    pickle.dump(estimator, file) # Se guarda el modelo entrenado en el archivo usando pickle.dump, que serializa el objeto estimator y lo escribe en el archivo file
    
    
    
    
    
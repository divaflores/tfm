from fastapi import FastAPI
import numpy as np
import io
import pandas as pd
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from tensorflow.python.ops.numpy_ops import np_config
import tensorflow as tf

app = FastAPI()

# Cargamos el modelo
model = TFAutoModelForSequenceClassification.from_pretrained('models')
tokenizer = AutoTokenizer.from_pretrained('lxyuan/distilbert-base-multilingual-cased-sentiments-student')


model_name = "Naturgy NLP"
model_file = 'model'
version = "v1.0.0"


@app.get("/")
async def root():
    return {
        "Welcome"
    }


@app.get("/info")
async def info():
    return {
        "Model": model_name,
        "Model file": model_file,
        "Version": version
    }



@app.post("/predict")
async def predict(text):

    #X = pd.DataFrame.from_dict(json_data, orient="index")
    #text = X[0]
    
    tf_inputs = tokenizer(text, padding=True, truncation=True, return_tensors="tf")
    tf_outputs = model(tf_inputs)
    prediction = model(**tf_inputs)

    tf_predictions = tf.math.softmax(tf_outputs.logits, axis=-1)
    np_config.enable_numpy_behavior()
    lista = tf_predictions.tolist()
    lista2 = lista[0]
    max_value = max(lista2)

    if lista2[0] == max_value:
        sent = 'positive'
    elif lista2[1] == max_value:
        sent = 'neutral'
    else:
        sent = 'negative'


    lista_sent = [sent, max_value]
    
    
    return {
        'status': 200,
        'label': lista_sent[0],
        'score': lista_sent[1]
    }
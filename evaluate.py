from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import pickle
model=load_model('model.h5')
tokenizer=load_model("tokenizer.pkl")
def predict(data):
    seq_data=tokenizer.texts_to_sequences([data])
    pad_data=pad_sequences(seq_data,maxlen=200)
    Y_pred=model.predict(pad_data)
    print(Y_pred[0])
    return Y_pred

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
model=pickle.load(open('model.pkl','rb'))
tokenizer=pickle.load(open("tokenizer.pkl",'rb'))
def predict(data):
    seq_data=tokenizer.texts_to_sequences([data])
    pad_data=pad_sequences(seq_data,maxlen=200)
    Y_pred=model.predict(pad_data)
    print(Y_pred[0])
    return Y_pred

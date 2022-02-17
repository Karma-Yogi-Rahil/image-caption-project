import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time
from PIL import Image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img


# modele -> resnet model
# model.h5 -> main model

new_dict = np.load("vocab.npy", allow_pickle=True)
dictio = dict(enumerate(new_dict.flatten(), 1))

final = {}

for k in dictio:
    for j,y in dictio[k].items():
        final[j] = y

modele = keras.models.load_model("modele.h5")
model = keras.models.load_model("model.h5")

inv_dict = {i:x for x,i in final.items()}

def getImage(x):
    
    

    test_img = np.array(Image.open(uploaded_file))
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    test_img = cv2.resize(test_img, (224,224))

    test_img = np.reshape(test_img, (1,224,224,3))
    
    return test_img

uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
if uploaded_file is not None:
    with st.spinner("Please wait for few seconds"):
        time.sleep(5)
    
    image = Image.open(uploaded_file)
    st.image(image)
    st.write(uploaded_file)
    test_feature = modele.predict(getImage(uploaded_file)).reshape(1,2048)

    #test_img_path = uploaded_file
    #test_img = cv2.imread(test_img_path)
    #test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    text_inp = ['startofseq']

    count = 0
    caption = ''
    while count < 25:
        count += 1

        encoded = []
        for i in text_inp:
            encoded.append(final[i])

        encoded = [encoded]

        encoded = pad_sequences(encoded, padding='post', truncating='post', maxlen=36)

        prediction = np.argmax(model.predict([test_feature, encoded]))

        sampled_word = inv_dict[prediction]

        caption = caption + ' ' + sampled_word
            
        if sampled_word == 'endofseq':
            break

        text_inp.append(sampled_word)
    
    
    st.write(caption[:-8])




















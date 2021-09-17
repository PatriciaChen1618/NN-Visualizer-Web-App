import streamlit as st
import requests
import json
import numpy as np
import matplotlib.pyplot as plt


URI = 'http://127.0.0.1:5000'

st.title('Neural Network Visualizer')
st.sidebar.markdown('### Input Image')

if st.button('Get Random Prediction'):
    response = requests.post(URI, data={})
    response = json.loads(response.text)
    predictions = response.get('prediction')
    image = response.get('image')
    image = np.reshape(image, (28, 28))
    
    st.image(image, width=150)

    for layer, p in enumerate(predictions):
        numbers = np.squeeze(np.array(p))
        plt.figure(figsize=(32, 4))
        
        if layer == 2:
            row = 1
            col = 10
        else:
            row = 2
            col = 16
        
        for i, number in enumerate(numbers):
            plt.subplot(row, col, i+1)
            plt.imshow(number * np.ones(8, 8, 3).astype('float32'))
            plt.xticks([])
            plt.yticks([])

import json
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.datasets.mnist import load_data
import numpy as np

from flask import Flask, request


app = Flask(__name__)

model = load_model('model.model')
feature_model = Model(
    model.inputs,
    [layer.output for layer in model.layers]
)

_, (x_test, _) = load_data()
x_test = x_test / 255.

def get_prediction():
    index = np.random.choice(x_test.shape[0])
    image = x_test[index, :, :]
    image_arr = np.reshape(image, (1, 784))
    return feature_model.predict(image_arr), image


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        predictions, image = get_prediction()
        final_predict =[p.tolist() for p in predictions]
        return json.dumps({
            'prediction': final_predict,
            'image': image.tolist()
        })
    return "This is the model server"


if __name__ == '__main__':
    app.run()

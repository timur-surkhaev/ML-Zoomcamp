import numpy as np

from io import BytesIO
from urllib import request

from PIL import Image

import tflite_runtime.interpreter as tflite


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

target_size = (200, 200)


interpreter = tflite.Interpreter(model_path='model_2024_hairstyle_v2.tflite')
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def predict(url):

    image = download_image(url)
    image_resized = prepare_image(image, target_size)

    x = np.array(image_resized, dtype='float32')
    X = np.array([x])
    X /= 255

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_index)
    
    print(preds)

    float_predictions = preds[0].tolist()

    return {'prediction': float_predictions}


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result

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



url = 'https://habrastorage.org/webt/yf/_d/ok/yf_dokzqy3vcritme8ggnzqlvwa.jpeg'
target_size = (200, 200)

tflite_model_name='model_2024_hairstyle_converted.tflite'




image = download_image(url)
image.save('original.jpeg')

image_resized = prepare_image(image, target_size)
image_resized.save('resized.jpeg')


x = np.array(image_resized, dtype='float32')
X = np.array([x])
X /= 255 #rescailing
print(X)

interpreter = tflite.Interpreter(model_path=tflite_model_name)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds = interpreter.get_tensor(output_index)
print(preds)





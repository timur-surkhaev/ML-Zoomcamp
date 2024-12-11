import tensorflow as tf
from tensorflow import keras

import tflite_runtime.interpreter as tflite

tflite_model_name = 'model_2024_hairstyle_converted.tflite'

model = keras.models.load_model('model_2024_hairstyle.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open(tflite_model_name, 'wb') as f_out:
    f_out.write(tflite_model)
    
interpreter = tflite.Interpreter(model_path=tflite_model_name)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

print(input_index)
print(output_index)

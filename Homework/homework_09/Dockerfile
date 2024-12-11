FROM agrigorev/model-2024-hairstyle:v3

COPY ["lambda_function.py", "./"]

RUN pip install pillow
RUN pip install keras_image_helper
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl
RUN pip install tflite_runtime --no-deps

RUN pip uninstall -y numpy
RUN pip install numpy==1.23.1

EXPOSE 8000

CMD [ "lambda_function.lambda_handler" ]

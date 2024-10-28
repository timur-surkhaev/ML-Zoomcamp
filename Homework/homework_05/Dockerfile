FROM svizor/zoomcamp-model:3.11.5-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model1.bin", "dv.bin","./"]

EXPOSE 8000

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:8000", "predict:app"]

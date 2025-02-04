FROM python:3.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

#instead of creating a virtual environment, we are installing the dependencies in the pyhton system
RUN pipenv install --system --deploy 

COPY ["scripts/predict.py", "./"]
COPY ["model/ridge_model.bin", "model/ridge_model.bin"]

#Open port 9696 to listen to requests
EXPOSE 9696

#Run the app using gunicorn
ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
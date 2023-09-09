# Base image
FROM python:3.9

COPY ./models /models
COPY ./app.py /app.py
COPY ./requirements.txt /requirements.txt

RUN pip install -r requirements.txt

CMD uvicorn app:app --host=0.0.0.0

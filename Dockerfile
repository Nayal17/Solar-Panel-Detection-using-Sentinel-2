FROM python:3-buster

RUN pip install --upgrade pip

WORKDIR /app

COPY ./README.md /app/README.md 
COPY ./setup.py /app/setup.py
COPY ./models /app/models/
COPY ./requirements.txt /app/requirements.txt
COPY ./src/SolarPanelDetection /app/src/SolarPanelDetection/

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

CMD ["python", "src/SolarPanelDetection/api.py"]

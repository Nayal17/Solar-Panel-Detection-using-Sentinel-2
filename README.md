# Solar-Panel-Detection-using-Sentinel-2

This project aims to develop a technique for accurately segmenting pixels with solar panels from Sentinel-2 optical satellite images, which have relatively low resolution.

### Dataset
Source: https://solafune.com/competitions/5dfc315c-1b24-4573-804f-7de8d707cd90?menu=data&tab=

#### Note:

- Kindly download data from above official link and update the gdrive link in ```./config/config.yaml ```
- This doesn't includes winning solutions from the competition (as winning solutions are prohibited to publish).

### Repository Guide:

- Import your repo to dagshub and add mlflow uri, token and username to your local evironment.
- Run ```python main.py``` to run complete pipeline including data ingestion, training, evaluation and saving weights.
- For testing model performance on already trained models, build a docker image which have fasapi prediction endpoints.
- Run ```docker build -t <image-name> .``` to build your own docker image.
- Run docker image ```docker run --name <container-name> -p 8000:8000 <image-name>```

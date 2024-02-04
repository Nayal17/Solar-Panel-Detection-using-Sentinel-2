from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException
from SolarPanelDetection.pipeline.prediction import PredictionPipeline
from pydantic import BaseModel

app = FastAPI(title="Prediction API")

class Img(BaseModel):
    img_url: str

@app.get("/")
async def ping():
    return "Hello World:)"

@app.post("/predict", status_code=200)
async def predict(request: Img):
    predictor = PredictionPipeline()
    prediction = predictor.predict(Path(request.img_url))
    if prediction is None:
        raise HTTPException(
            status_code=404, detail="Image could not be downloaded"
        )
    return prediction.tolist()

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, log_level="debug",
                proxy_headers=True, reload=True)
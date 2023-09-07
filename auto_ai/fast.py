import pandas as pd
import os
import csv
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from io import BytesIO
from auto_ai.object_detector import *
from auto_ai.demo import *
from PIL import Image, ImageDraw, ImageColor, ImageFont
import json

from fastapi import FastAPI, HTTPException, UploadFile
import shutil
from auto_ai.demo import demo



from auto_ai.registry import load_model, save_model
from auto_ai.preprocessing import *
from auto_ai.image_preprocessing import *

from pydantic import BaseModel

PATH_IMAGE =os.environ.get('PATH_IMAGE')
PATH_PROC_IMAGE = os.environ.get('PATH_PROC_IMAGE')
PATH_TEST_IMAGE=os.environ.get('PATH_TEST_IMAGE')

app=FastAPI()
app.state.model=load_model()
def read_image(image_encoded):
    pil_image=Image.open(BytesIO(image_encoded))
    return pil_image


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2

@app.get("/status")
def predict():
    return {'road_sign':'y'}


# @app.post("/api/predict")
# async def predict_image(file:UploadFile=File(...)):
#     contents = await file.read()
#     nparr = np.fromstring(contents, np.uint8)
#     image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     encoded_img = cv2.imencode('.PNG', image)



@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        raise HTTPException(status_code=400, detail="Wrong extension")
    if file is None:
        raise HTTPException(status_code=400, detail="No file uploaded")
    try:
        # Save the uploaded file to a temporary location
        with open(f'/opt/build/auto_ai/{file.filename}', 'wb') as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(file.filename)
        path=f'/opt/build/auto_ai/{file.filename}'
        print(__file__)
        prediction=demo(path)
        print('✅ Prediction data acquired')
        img=Image.fromarray(prediction)
        path_img=f'/opt/build/auto_ai/test.jpeg'
        img.save(path_img)
        return FileResponse(path_img,media_type='image/jpeg')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/processed')
def view_process():
    filename=f'{PATH_TEST_IMAGE}/Figure_1.png'
    return FileResponse(filename)

if __name__ == '__main__':
    X_test=object_detector('/home/parfait/code/Samuel151202/Autonomous-ai/data/test_images/Panneau-Stops.jpg')

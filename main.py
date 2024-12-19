import uvicorn 
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import pickle
import tensorflow as tf
import cv2
import base64
from fastapi.middleware.cors import CORSMiddleware
from tensorflow import keras
import tensorflow.compat.v2 as tf

app = FastAPI()

# Load your scikit-learn models
with open("data_files/crop_recom_using_weather.pkl", "rb") as f:
    classifier = pickle.load(f)

with open("data_files/wheat_msp.pkl", "rb") as f:
    regress = pickle.load(f)

# Load your Keras model
cnn_model = keras.models.load_model("data_files/cnn.h5")
classes = [
    {'Alluvial Soil': ['Rice', 'Wheat', 'Sugarcane', 'Jute', 'Pulses', 'Oilseeds']},
    {'Black Soil': ['Cotton', 'Sugarcane', 'Groundnut', 'Wheat', 'Millets', 'Pulses']},
    {'Clay Soil': ['Rice', 'Lettuce', 'Broccoli', 'Cabbage', 'Soybean', 'Peas']},
    {'Red Soil': ['Millets', 'Groundnut', 'Cotton', 'Potatoes', 'Oilseeds', 'Pulses']}
]


# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Index endpoint
@app.get('/')
def index():
    return {'message': 'Hello, World'}

# Endpoint for crop recommendation
@app.post("/predict")
def predict_crop(temperature: float, humidity: float, rainfall: float):
    prediction = classifier.predict([[temperature, humidity, rainfall]])
    response_content = {'prediction': prediction.tolist()}  
    return JSONResponse(content=response_content)

# Endpoint for wheat MSP prediction
@app.post("/predict1")
def predict_msp(arrival_date: float, average_temp: float, rainfall: float):
    prediction = regress.predict([[arrival_date, average_temp, rainfall]])
    response_content = {'prediction': prediction.tolist()} 
    return JSONResponse(content=response_content)

# Validate and read image functions
async def validate_image(file: UploadFile):
    if not file:
        raise HTTPException(status_code=400, detail="File is empty")

async def read_image(file: UploadFile):
    contents = await file.read()
    return contents

# Convert image to PNG and preprocess functions
def convert_to_png(image_data):
    image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    _, encoded_image = cv2.imencode(".png", image)
    png_base64 = base64.b64encode(encoded_image).decode("utf-8")
    return png_base64

def preprocess_image(png_base64):
    decoded_image = base64.b64decode(png_base64)
    image = cv2.imdecode(np.frombuffer(decoded_image, np.uint8), cv2.IMREAD_COLOR)
    resized_image = cv2.resize(image, (128, 128))
    normalized_image = resized_image / 255.0
    preprocessed_image = np.expand_dims(normalized_image, axis=0)
    return preprocessed_image

# Endpoint for soil prediction from image
@app.post("/predict2")
async def predict_soil(file: UploadFile = File(...)):
    try:
        await validate_image(file)
        image_data = await read_image(file)
        png_base64 = convert_to_png(image_data)
        processed_image = preprocess_image(png_base64)

        # Make predictions using the loaded CNN model
        dl_prediction = cnn_model.predict(processed_image)
        
        response_content = {'dl_prediction': dl_prediction.tolist()}
        dl_prediction_array = np.array(dl_prediction)
        
        max_index = np.argmax(dl_prediction_array)
        max_soil_type = classes[max_index]

        response_content['max_soil_type'] = max_soil_type

    except HTTPException as http_err:
        return JSONResponse(status_code=http_err.status_code, content={"error": str(http_err)})
    except Exception as err:
        return JSONResponse(status_code=500, content={"error": f"Internal Server Error: {str(err)}"})

    return JSONResponse(content=response_content)

# Run the application
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8080)

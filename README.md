# Crop and Soil Prediction API

This repository contains a FastAPI-based application for crop recommendation, wheat Minimum Support Price (MSP) prediction, and soil type classification and crop suggestion using machine learning and deep learning models. The API provides endpoints for:

1. **Crop Recommendation**: Predicts the best crop to grow based on environmental parameters.
2. **Wheat MSP Prediction**: Predicts the MSP of wheat based on environmental and market parameters.
3. **Soil Type Prediction and Crop Suggestion**: Classifies the type of soil based on an uploaded image and suggest the suitable crop for that soil.

## Features

- **Crop Recommendation**: Uses a scikit-learn classifier trained on temperature, humidity, and rainfall data.
- **Wheat MSP Prediction**: Utilizes a regression model to predict the MSP of wheat.
- **Soil Type Classification**: Employs a Convolutional Neural Network (CNN) model for image classification of soil types.
- **CORS Enabled**: Ensures cross-origin resource sharing for integration with other services.

## Requirements

- Python 3.8 or higher
- FastAPI
- Uvicorn
- NumPy
- TensorFlow
- scikit-learn
- OpenCV
- Base64

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/crop-soil-prediction-api.git
   cd crop-soil-prediction-api
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```



## API Endpoints

### 1. **Index Endpoint**

- **URL**: `/`
- **Method**: `GET`
- **Response**:
  ```json
  {
    "message": "Hello, World"
  }
  ```

### 2. **Crop Recommendation**

- **URL**: `/predict`
- **Method**: `POST`
- **Parameters**:
  - `temperature` (float): Average temperature in Celsius.
  - `humidity` (float): Relative humidity percentage.
  - `rainfall` (float): Average rainfall in mm.
- **Response**:
  ```json
  {
    "prediction": ["Recommended Crop"]
  }
  ```

### 3. **Wheat MSP Prediction**

- **URL**: `/predict1`
- **Method**: `POST`
- **Parameters**:
  - `arrival_date` (float): Market arrival date.
  - `average_temp` (float): Average temperature in Celsius.
  - `rainfall` (float): Average rainfall in mm.
- **Response**:
  ```json
  {
    "prediction": ["Predicted MSP"]
  }
  ```

### 4. **Soil Type Prediction**

- **URL**: `/predict2`
- **Method**: `POST`
- **Parameters**:
  - `file`: An image file of the soil.
- **Response**:
  ```json
  {
    "dl_prediction": [[Prediction Probabilities]],
    "max_soil_type": {
      "Soil Type": ["Recommended Crops"]
    }
  }
  ```

## Running the Application

1. Start the server:
   ```bash
   python main.py
   ```

2. Access the API at:
   ```
   http://127.0.0.1:8080
   ```

## Example Usage

- Use tools like Postman or `curl` to test the API endpoints.
- For soil type prediction, upload an image file of the soil in `.png` or `.jpg` format.

## Directory Structure

```
.
├── data_files
│   ├── crop_recom_using_weather.pkl
│   ├── wheat_msp.pkl
│   └── cnn.h5
├── main.py
├── requirements.txt
└── README.md
```

## Contributing

Feel free to open issues or submit pull requests for improvements and bug fixes.

## License

This project is licensed under the MIT License.


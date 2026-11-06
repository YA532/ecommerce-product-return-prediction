
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Load trained model
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="E-Commerce Product Return Prediction API",
    description="Predict whether an e-commerce product will be returned.",
    version="1.0"
)


class ProductDetails(BaseModel):
    Product_Category: str
    Product_Price_INR: float = Field(gt=0)
    Discount_Percent: float = Field(ge=0, le=100)
    Expected_Delivery_Days: int = Field(gt=0)
    Actual_Delivery_Days: float = Field(gt=0)
    Customer_Rating_PreOrder: float = Field(ge=1, le=5)
    Previous_Returns: int = Field(ge=0)
    Previous_Orders: int = Field(gt=0)
    Payment_Method: str
    Damaged_Package_Flag: str
    Package_Size: str
    Promotion_Active: str


@app.get("/")
def home():
    return {
        "message": "E-Commerce Product Return Prediction API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_name": "Logistic Regression"
    }


@app.post("/predict")
def predict_return(product: ProductDetails):

    data = product.model_dump()

    # Feature engineering
    data["Discount_Amount"] = (
        data["Product_Price_INR"] *
        data["Discount_Percent"] / 100
    )

    data["Delivery_Delay"] = (
        data["Actual_Delivery_Days"] -
        data["Expected_Delivery_Days"]
    )

    data["Customer_Return_Frequency"] = (
        data["Previous_Returns"] /
        data["Previous_Orders"]
    )

    # Convert input into DataFrame
    input_data = pd.DataFrame([data])

    # Predict return status
    prediction = int(model.predict(input_data)[0])
    probability = float(model.predict_proba(input_data)[0][1])

    return {
        "prediction": prediction,
        "return_status": (
            "Returned" if prediction == 1 else "Not Returned"
        ),
        "return_probability": round(probability, 4)
    }

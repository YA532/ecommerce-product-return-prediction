
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_prediction():
    data = {
        "Product_Category": "Beauty",
        "Product_Price_INR": 8772.58,
        "Discount_Percent": 20.1,
        "Expected_Delivery_Days": 5,
        "Actual_Delivery_Days": 7,
        "Customer_Rating_PreOrder": 4.2,
        "Previous_Returns": 2,
        "Previous_Orders": 12,
        "Payment_Method": "UPI",
        "Damaged_Package_Flag": "No",
        "Package_Size": "Small",
        "Promotion_Active": "Yes"
    }

    response = client.post("/predict", json=data)

    assert response.status_code == 200
    assert response.json()["return_status"] in [
        "Returned", "Not Returned"
    ]


def test_invalid_prediction():
    response = client.post("/predict", json={})
    assert response.status_code == 422

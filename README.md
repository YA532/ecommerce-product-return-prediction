# E-Commerce Product Return Prediction System using CI/CD

## 1. Project Description
This project predicts whether an e-commerce product order will be returned or not returned using Machine Learning. It uses two classification algorithms, Logistic Regression and Random Forest, and selects the best model based on F1-score.

The application is deployed using FastAPI and Docker. GitHub Actions automates model training, testing, Docker image building, and publishing to GitHub Container Registry (GHCR).

## 2. Technologies Used
- Python
- Pandas and NumPy
- Scikit-learn
- FastAPI
- Pytest
- Docker
- GitHub Actions
- GitHub Container Registry

## 3. Installation Commands
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 4. Execution Steps
Train the models:
```powershell
python train_model.py
```

Run API tests:
```powershell
python -m pytest tests -v
```

Start FastAPI:
```powershell
uvicorn app:app --host 127.0.0.1 --port 8000
```

## 5. API Endpoints
- GET `/` – Home endpoint.
- GET `/health` – Checks API and model health.
- POST `/predict` – Predicts whether an order will be returned.

Swagger UI: `http://127.0.0.1:8000/docs`

## 6. Model Results

| Model | Accuracy | F1-score |
|---|---:|---:|
| Logistic Regression | 60.83% | 0.3380 |
| Random Forest | 73.33% | 0.2000 |

Selected model: Logistic Regression.

## 7. Docker Commands
```powershell
docker build -t ecommerce-product-return:latest .
docker run -d -p 8000:8000 --name ecommerce-return-app ecommerce-product-return:latest
```

## 8. GitHub Repository
https://github.com/YA532/ecommerce-product-return-prediction

## 9. GitHub Actions
https://github.com/YA532/ecommerce-product-return-prediction/actions

GitHub Actions automatically trains the model, runs tests, builds the Docker image, and publishes it to GHCR.

## 10. Docker Image
`ghcr.io/ya532/ecommerce-product-return:latest`

## 11. Updated Deployment
The updated API version is 1.1. The updated GHCR Docker image was pulled and verified successfully.

Local verification URL: `http://127.0.0.1:8001`

**Note:** Localhost is not publicly accessible to faculty. Add a public application URL after deploying the API to a hosting provider.
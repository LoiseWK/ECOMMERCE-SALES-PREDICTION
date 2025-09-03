import uvicorn
import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# 1️⃣ Load trained model pipeline (make sure you saved it as model.pkl earlier)
with open("best_model.pkl", "rb") as f:
    model_pipeline = pickle.load(f)

# 2️⃣ Define input schema
class ForecastInput(BaseModel):
    Date: str
    Product_Category: str
    Units_Sold: float = None   # optional (for test data with true values)
    # add more exogenous vars if your model uses them, e.g. Price, Promo, etc.

# 3️⃣ Initialize FastAPI
app = FastAPI(title="Ecommerce Sales Forecasting API")

@app.get("/")
def home():
    return {"message": "Ecommerce Sales Forecasting API is running 🚀"}

# 4️⃣ Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "ok"}

# 5️⃣ Prediction endpoint
@app.post("/predict")
def predict_sales(data: ForecastInput):
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Predict using pipeline
    y_pred = model_pipeline.predict(input_df)

    return {
        "Product_Category": data.Product_Category,
        "Date": data.Date,
        "Forecast_Units_Sold": float(y_pred[0])
    }

# 6️⃣ Run with: uvicorn app:app --reload
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

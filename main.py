from fastapi import FastAPI, Request
from MODEL.predict import classify_transaction

app = FastAPI()

@app.post("/predict")
async def predict_transaction(request: Request):
    json_data = await request.json()
    result = classify_transaction(json_data)
    return result

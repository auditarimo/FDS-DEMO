from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import xml.etree.ElementTree as ET
from MODEL.predict import classify_transaction

app = FastAPI()

@app.get("/")
async def root():
    return JSONResponse(content={"message": "ISO 20022 Transaction Fraud Detection API is running."})

@app.post("/predict")
async def predict_transaction(request: Request):
    content_type = request.headers.get("content-type", "").lower()
    if "xml" not in content_type:
        raise HTTPException(
            status_code=415,
            detail="Unsupported Media Type. This endpoint only accepts application/xml."
        )

    # Read raw XML body
    body = await request.body()
    try:
        # Parse ISO 20022 pain.001 namespace
        ns = {"p": "urn:iso:std:iso:20022:tech:xsd:pain.001.001.09"}
        root = ET.fromstring(body)

        # Extract fields
        initiator = root.findtext(".//p:Dbtr/p:Id//p:Id", namespaces=ns)
        tx = root.find(".//p:CdtTrfTxInf", ns)
        recipient = tx.findtext("p:CdtrAcct/p:Id/p:IBAN", namespaces=ns)
        amount = float(tx.findtext("p:Amt/p:InstdAmt", namespaces=ns))

        # Build the payload dict
        xml_data = {
            "initiator": initiator,
            "recipient": recipient,
            "amount": amount,
            "transactionType": "TRANSFER",
            "oldBalInitiator": 0.0,
            "newBalInitiator": 0.0,
            "oldBalRecipient": 0.0,
            "newBalRecipient": 0.0
        }
    except ET.ParseError as e:
        raise HTTPException(status_code=400, detail=f"Malformed XML: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"XML processing error: {e}")

    # Delegate to your classifier
    result = classify_transaction(xml_data)
    return JSONResponse(result)

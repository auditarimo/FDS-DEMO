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

    try:
        body = await request.body()
        ns = {"p": "urn:iso:std:iso:20022:tech:xsd:pain.001.001.09"}
        root = ET.fromstring(body)

        # Extract initiator
        initiator = root.findtext(".//p:Dbtr//p:PrvtId//p:Othr//p:Id", namespaces=ns)
        if not initiator:
            raise ValueError("Initiator not found in XML")

        # Extract transaction node
        tx = root.find(".//p:CdtTrfTxInf", ns)
        if tx is None:
            raise ValueError("Transaction details not found in XML")

        # Extract recipient
        recipient = root.findtext(".//p:CdtTrfTxInf//p:PrvtId//p:Othr//p:Id", namespaces=ns)
        if not recipient:
            raise ValueError("Recipient IBAN not found in XML")

        # Extract amount
        amount = float(root.findtext(".//p:CdtTrfTxInf/p:Amt/p:InstdAmt", namespaces=ns))
        if not amount:
            raise ValueError("Amount not found in XML")

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

        # Classify
        result = classify_transaction(xml_data)
        return JSONResponse(result)

    except ET.ParseError as e:
        raise HTTPException(status_code=400, detail=f"Malformed XML: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"XML processing error: {e}")

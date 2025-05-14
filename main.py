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
        # Get raw body
        body = await request.body()
        print("Raw XML received:")
        print(body.decode())

        # Define XML namespace
        ns = {"p": "urn:iso:std:iso:20022:tech:xsd:pain.001.001.09"}
        root = ET.fromstring(body)

        # Extract initiator (e.g., phone number or custom ID)
        initiator = root.findtext(".//p:Dbtr/p:Id/p:PrvtId/p:Othr/p:Id", namespaces=ns)
        if not initiator:
            raise ValueError("Initiator not found in XML")

        # Extract transaction node
        tx = root.find(".//p:CdtTrfTxInf", ns)
        if tx is None:
            raise ValueError("Transaction details not found in XML")

        # Extract recipient (again assuming it's under PrvtId/Othr/Id)
        recipient = tx.findtext(".//p:Cdtr/p:Id/p:PrvtId/p:Othr/p:Id", namespaces=ns)
        if not recipient:
            raise ValueError("Recipient not found in XML")

        # Extract amount
        amount_text = tx.findtext("p:Amt/p:InstdAmt", namespaces=ns)
        if not amount_text:
            raise ValueError("Amount not found in XML")
        amount = float(amount_text)

        # Log extracted values
        print(f"Initiator: {initiator}")
        print(f"Recipient: {recipient}")
        print(f"Amount: {amount}")

        # Prepare payload
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

        # Run classifier
        result = classify_transaction(xml_data)
        print("Classification result:", result)
        return JSONResponse(content=result)

    except ET.ParseError as e:
        print("XML Parse Error:", e)
        raise HTTPException(status_code=400, detail=f"Malformed XML: {e}")
    except Exception as e:
        print("Unexpected Error:", e)
        raise HTTPException(status_code=400, detail=f"XML processing error: {e}")

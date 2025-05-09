# main.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import xml.etree.ElementTree as ET
from MODEL.predict import predict  # your function that takes a dict

app = FastAPI()

@app.post("/predict-iso")
async def predict_iso(request: Request):
    content_type = request.headers.get("content-type", "")
    body = await request.body()
    # 1) parse XML body into our internal dict
    if "xml" in content_type:
        try:
            ns = {"p": "urn:iso:std:iso:20022:tech:xsd:pain.001.001.09"}
            root = ET.fromstring(body)
            # extract one CdtTrfTxInf block (you can loop/multi‑edge)
            tx = root.find(".//p:CdtTrfTxInf", ns)
            amount_el = tx.find("p:Amt/p:InstdAmt", ns)
            e2e = tx.find("p:PmtId/p:EndToEndId", ns)
            dbtr = tx.find("p:Dbtr/ p:Id/p:OrgId/p:Othr/p:Id", ns)
            cdtr = tx.find("p:CdtrAcct/p:Id/p:IBAN", ns)
            sample = {
                "initiator": dbtr.text,
                "recipient": cdtr.text,
                "amount": float(amount_el.text),
                "transactionType": "TRANSFER",
                "oldBalInitiator": 0.0,
                "newBalInitiator": 0.0,
                "oldBalRecipient": 0.0,
                "newBalRecipient": 0.0
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"XML parse error: {e}")
    else:
        # assume JSON
        sample = await request.json()

    # 2) call your predict function
    out = predict(sample)
    return JSONResponse(out)

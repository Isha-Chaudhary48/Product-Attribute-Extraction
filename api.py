import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Load model + vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

app = FastAPI()

class InputText(BaseModel):
    description: str

# Prediction function
def predict(description):
    text = [description.lower()]
    X = vectorizer.transform(text)
    pred = model.predict(X)[0]
    return pred

@app.post("/extract")
def extract(data: InputText):
    pred = predict(data.description)

    try:
        pred = pred.tolist()
    except:
        pred = list(pred)

    attributes = {
        "silhouette": pred[0],
        "fabric": pred[1],
        "neckline": pred[2],
        "sleeve": pred[3],
        "length": pred[4],
        "embellishment": pred[5],
        "color": pred[6],
        "category": pred[7]
    }

    # RULE-BASED FIXES (INSIDE FUNCTION)
    text = data.description.lower()

 
    # LENGTH
    if "maxi" in text or "floor length" in text or "floor-length" in text or "full length" in text:
        attributes["length"] = "long"
    elif "mini" in text or "short" in text:
        attributes["length"] = "short"
    elif "midi" in text or "knee length" in text or "knee-length" in text:
        attributes["length"] = "midi"
    else:
        attributes["length"] = "unknown"

    # CATEGORY
    if "party" in text:
        attributes["category"] = "party"
    elif "wedding" in text or "bridal" in text:
        attributes["category"] = "bridal"
    elif "casual" in text:
        attributes["category"] = "casual"
    elif "night" in text:
        attributes["category"] = "nightwear"
    elif "evening" in text:
        attributes["category"] = "evening"

   
    # EMBELLISHMENT
    if "lace" in text:
        attributes["embellishment"] = "lace"
    elif "sequin" in text:
        attributes["embellishment"] = "sequin"
    elif "embroidery" in text:
        attributes["embellishment"] = "embroidery"
    elif "bead" in text:
        attributes["embellishment"] = "beading"
    elif "feather" in text:
        attributes["embellishment"] = "feather trim"
    elif "glitter" in text:
        attributes["embellishment"] = "glitter"
    else:
        attributes["embellishment"] = "none"
    
    if "cotton" in text:
        attributes["fabric"] = "cotton"
    elif "silk" in text:
        attributes["fabric"] = "silk"
    elif "chiffon" in text:
        attributes["fabric"] = "chiffon"
    elif "denim" in text:
        attributes["fabric"] = "denim"
    elif "satin" in text:
        attributes["fabric"] = "satin"
    elif "tulle" in text:
        attributes["fabric"] = "tulle"
    elif "velvet" in text:
        attributes["fabric"] = "velvet"
    elif "lace" in text:
        attributes["fabric"] = "lace"
    elif "wool" in text:
        attributes["fabric"] = "wool"
    elif "leather" in text:
        attributes["fabric"] = "leather"
    elif "polyester" in text:
        attributes["fabric"] = "polyester"
    elif "jersey" in text:
        attributes["fabric"] = "jersey"
    else:
        attributes["fabric"] = "unknown"
    
    

    return {"attributes": attributes}
# Product Attribute Extraction API

This project extracts useful attributes from a product description using Machine Learning and simple rules.

---

## What it does

Given a text description of a product, it predicts:

- Silhouette
- Fabric
- Neckline
- Sleeve
- Length
- Embellishment
- Color
- Category

---

## Example

### Input:
```json
{
  "description": "Red lace sleeveless mini party dress"
}
```
---

---
# output:

{
  "attributes": {
    "silhouette": "A-line",
    "fabric": "unknown",
    "neckline": "unknown",
    "sleeve": "sleeveless",
    "length": "short",
    "embellishment": "lace",
    "color": "red",
    "category": "party"
  }
}
---

## Evaluation Results

The model was evaluated on multiple attributes:

- Neckline: ~90% accuracy
- Sleeve: ~70% accuracy
- Length: ~70% accuracy
- Color: ~70% accuracy
- Embellishment: ~80% accuracy
- Category: ~50% accuracy

## Observations

- Good performance on frequently occurring labels
- Lower performance on rare categories due to limited data
- Model predicts "unknown" when unsure

## Limitations

- Small dataset
- Class imbalance
- Rare labels are harder to predict

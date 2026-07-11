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
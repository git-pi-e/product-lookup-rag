from typing import Dict


entity_types: Dict[str, str] = {
    "product": "Item detailed type, e.g. 'high waist pants', 'chef kitchen knife'",
    "category": "Item category, e.g. 'home decoration', 'women clothing'",
    "characteristic": "Item characteristics, e.g. 'waterproof', 'adhesive'",
    "measurement": "Dimensions of the item",
    "brand": "Brand of the item",
    "color": "Color of the item",
    "age_group": "Target age group: 'babies', 'children', 'teenagers', 'adults' (pick oldest if multiple)",
}


relation_types: Dict[str, str] = {
    "hasCategory": "item is of this category",
    "hasCharacteristic": "item has this characteristic",
    "hasMeasurement": "item is of this measurement",
    "hasBrand": "item is of this brand",
    "hasColor": "item is of this color",
    "isFor": "item is for this age_group",
}


entity_relationship_match: Dict[str, str] = {
    "category": "hasCategory",
    "characteristic": "hasCharacteristic",
    "measurement": "hasMeasurement",
    "brand": "hasBrand",
    "color": "hasColor",
    "age_group": "isFor",
}

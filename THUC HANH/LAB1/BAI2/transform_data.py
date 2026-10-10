
import math
import pandas as pd

def to_number(value):
    if value is None:
        return None
    
    text = str(value).strip().replace(",", "")
    
    if text == "":
        return None
    
    try:
        number = float(text)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except (ValueError, TypeError):
        return None

def get_card_category(card_type):
    text = str(card_type or "").lower()
    
    if "monster" in text:
        return 0
    elif "spell card" in text:
        return 1
    elif "trap card" in text:
        return 2
    elif "skill" in text:
        return 3
    else:
        return 4

def transform_cards(cards):
    rows = []

    for card in cards:
        card_type = str(card.get("type") or "")
        type_lower = card_type.lower()

        link_markers = card.get("linkmarkers") or []

        prices = {}
        price_list = card.get("card_prices") or []

        if price_list and isinstance(price_list[0], dict):
            prices = price_list[0]

        card_id = to_number(card.get("id"))

        if card_id is not None:
            card_id = int(card_id)

        row = {
            "card_id": card_id,
            "atk": to_number(card.get("atk")),
            "def": to_number(card.get("def")),
            "level_rank": to_number(card.get("level")),
            "linkval": to_number(card.get("linkval")),
            "scale": to_number(card.get("scale")),
            "link_marker_count": len(link_markers),

            "price_cardmarket_eur": to_number(
                prices.get("cardmarket_price")
            ),
            "price_tcgplayer_usd": to_number(
                prices.get("tcgplayer_price")
            ),
            "is_extra_deck": int(
                any(x in type_lower
                    for x in ["fusion", "synchro", "xyz", "link"])
            ),
            "is_pendulum": int("pendulum" in type_lower),
            "is_fusion": int("fusion" in type_lower),
            "is_ritual": int("ritual" in type_lower),
            "is_synchro": int("synchro" in type_lower),
            "is_xyz": int("xyz" in type_lower),
            "is_link": int("link" in type_lower),
            "is_effect": int("effect" in type_lower),
            "is_normal": int("normal" in type_lower),
            "card_category": get_card_category(card_type),
        }

        rows.append(row)

    columns = [
        "card_id",
        "atk",
        "def",
        "level_rank",
        "linkval",
        "scale",
        "link_marker_count",
        "price_cardmarket_eur",
        "price_tcgplayer_usd",
        "is_extra_deck",
        "is_pendulum",
        "is_fusion",
        "is_ritual",
        "is_synchro",
        "is_xyz",
        "is_link",
        "is_effect",
        "is_normal",
        "card_category",
    ]

    data = pd.DataFrame(rows, columns=columns)

    for column in data.columns:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    return data
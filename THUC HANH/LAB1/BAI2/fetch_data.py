import requests

API_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php"

def fetch_all_cards():
    print("Đang tải dữ liệu từ YGOPRODeck API...")
    
    response = requests.get(API_URL)
    response.raise_for_status()
    result = response.json()

    cards = result.get("data")

    print("Đã tải", len(cards), "thẻ")
    return cards

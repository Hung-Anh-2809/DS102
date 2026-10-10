from pathlib import Path
from fetch_data import fetch_all_cards
from transform_data import transform_cards

data_folder = Path(__file__).resolve().parent / "data"
data_folder.mkdir(exist_ok=True)

cards = fetch_all_cards()
data = transform_cards(cards)

output_file = data_folder / "yugioh.csv"
data.to_csv(output_file, index=False, encoding="utf-8")

print("Đã lưu CSV tại:", output_file)
print("Số dòng:", len(data))
print("Số cột:", len(data.columns))
"""Генерация датасета продаж аптечной сети (учебный проект)."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42

REGIONS = {
    "Москва": {"stores": 8},
    "Санкт-Петербург": {"stores": 5},
    "Казань": {"stores": 3},
    "Новосибирск": {"stores": 3},
    "Екатеринбург": {"stores": 3},
}

CATEGORIES = {
    "Лекарства (рецепт)": {
        "base_price": 420,
        "margin": 0.18,
        "daily_qty": 2.8,
        "names": [
            "Амоксициллин 500 мг", "Омепразол 20 мг", "Аторвастатин 10 мг",
            "Лозартан 50 мг", "Метформин 850 мг", "Амлодипин 5 мг",
            "Эналаприл 10 мг", "Бисопролол 5 мг", "Панангин", "Вальсартан 80 мг",
        ],
    },
    "Лекарства (без рецепта)": {
        "base_price": 280,
        "margin": 0.25,
        "daily_qty": 4.5,
        "names": [
            "Нурофен 200 мг", "Цитрамон", "Парацетамол 500 мг", "Ибупрофен",
            "Терафлю", "Смекта", "Мезим", "Активированный уголь", "Но-шпа",
            "Анальгин", "Аспирин", "Ринза", "Гриппферон", "Лоратадин",
        ],
    },
    "Косметика и гигиена": {
        "base_price": 350,
        "margin": 0.32,
        "daily_qty": 3.2,
        "names": [
            "La Roche-Posay крем", "Bioderma Sensibio", "Nivea увлажняющий крем",
            "Colgate Total", "Head & Shoulders 400 мл", "Дезодорант Rexona",
            "Прокладки Always", "Подгузники Pampers", "Зубная паста Splat",
            "Шампунь Gliss Kur", "Мицеллярная вода", "Крем для рук",
        ],
    },
    "БАД и витамины": {
        "base_price": 520,
        "margin": 0.35,
        "daily_qty": 2.1,
        "names": [
            "Омега-3 1000 мг", "Витамин D3 2000 МЕ", "Магний B6",
            "Компливит", "Аевит", "Рыбий жир", "Кальций D3", "Селен",
            "Цинк", "Витамин C 1000 мг", "Глюкозамин", "Коэнзим Q10",
        ],
    },
}

PAYMENT_METHODS = ["Наличные", "Карта", "Онлайн"]
GENDERS = ["Женский", "Мужской"]
AGE_GROUPS = ["18-30", "31-45", "46-60", "60+"]


def generate_products(rng):
    rows = []
    product_id = 1
    for category, params in CATEGORIES.items():
        names = params["names"]
        n_products = rng.integers(max(15, len(names)), max(20, len(names) + 8))
        for i in range(n_products):
            price = max(45, rng.normal(params["base_price"], params["base_price"] * 0.25))
            rows.append({
                "product_id": product_id,
                "product_name": names[i % len(names)],
                "category": category,
                "price": round(price, 2),
                "margin_pct": round(params["margin"] + rng.normal(0, 0.03), 3),
            })
            product_id += 1
    return pd.DataFrame(rows)


def pick_payment(rng, region, age_group):
    # базовые доли: наличные / карта / онлайн
    probs = [0.22, 0.48, 0.30]

    if region == "Москва":
        probs[2] += 0.25
        probs[1] -= 0.12
    if age_group == "18-30":
        probs[2] += 0.18
        probs[0] -= 0.10
    if age_group == "60+":
        probs[0] += 0.12
        probs[2] -= 0.10

    probs = np.clip(probs, 0.05, None)
    probs = probs / sum(probs)
    return rng.choice(PAYMENT_METHODS, p=probs)


def generate_transactions(products, rng, n_days=180):
    start_date = pd.Timestamp("2025-01-01")
    promo_start = start_date + pd.Timedelta(days=90)
    promo_end = promo_start + pd.Timedelta(days=21)

    stores = []
    store_id = 1
    for region, meta in REGIONS.items():
        for _ in range(meta["stores"]):
            stores.append({"store_id": store_id, "region": region})
            store_id += 1
    stores_df = pd.DataFrame(stores)

    rows = []
    transaction_id = 1

    for day_offset in range(n_days):
        current_date = start_date + pd.Timedelta(days=day_offset)
        is_promo = promo_start <= current_date <= promo_end
        is_weekend = current_date.weekday() >= 5

        traffic = 95
        if is_weekend:
            traffic *= 1.12
        if is_promo:
            traffic *= 1.18

        for _ in range(int(rng.poisson(traffic))):
            store = stores_df.sample(1, random_state=rng.integers(0, 1_000_000)).iloc[0]
            product = products.sample(
                1, weights=1 / products["price"],
                random_state=rng.integers(0, 1_000_000),
            ).iloc[0]

            gender = rng.choice(GENDERS, p=[0.58, 0.42])
            age_group = rng.choice(AGE_GROUPS, p=[0.22, 0.31, 0.27, 0.20])

            cat_params = CATEGORIES[product["category"]]
            qty_base = cat_params["daily_qty"]
            # чем дороже позиция, тем меньше штук в среднем берут
            qty = max(1, int(rng.poisson(qty_base * (1.4 - min(product["price"] / 900, 0.9)))))

            # в косметике и БАД чек чуть выше — чаще берут несколько позиций
            if product["category"] in ("Косметика и гигиена", "БАД и витамины") and gender == "Женский":
                qty = max(qty, qty + rng.integers(0, 2))

            discount = 0.0
            if is_promo and rng.random() < 0.35:
                discount = float(rng.choice([0.05, 0.10, 0.15]))

            amount = round(product["price"] * qty * (1 - discount), 2)
            payment = pick_payment(rng, store["region"], age_group)

            rows.append({
                "transaction_id": transaction_id,
                "date": current_date,
                "store_id": int(store["store_id"]),
                "region": store["region"],
                "product_id": int(product["product_id"]),
                "category": product["category"],
                "price": product["price"],
                "quantity": qty,
                "discount_pct": discount,
                "amount": amount,
                "gender": gender,
                "age_group": age_group,
                "payment_method": payment,
                "is_promo_period": int(is_promo),
            })
            transaction_id += 1

    return pd.DataFrame(rows)


def main(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    products = generate_products(rng)
    transactions = generate_transactions(products, rng)

    products.to_csv(output_dir / "products.csv", index=False, encoding="utf-8-sig")
    transactions.to_csv(output_dir / "transactions.csv", index=False, encoding="utf-8-sig")

    daily_revenue = (
        transactions.groupby("date", as_index=False)["amount"]
        .sum()
        .rename(columns={"amount": "daily_revenue"})
    )
    daily_revenue.to_csv(output_dir / "daily_revenue.csv", index=False, encoding="utf-8-sig")

    print(f"Готово: {len(products)} товаров, {len(transactions)} транзакций -> {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    main(args.output)

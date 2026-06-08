"""
Генерация синтетического датасета продаж аптечной сети «ФармаПлюс».

Данные содержат встроенные закономерности для проверки статистических гипотез:
- различия среднего чека по полу покупателя;
- рост выручки после промо-кампании;
- различия выручки по категориям товаров;
- связь способа оплаты с возрастной группой;
- обратная корреляция цены и количества;
- большая доля онлайн-заказов в Москве.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_SEED = 42

REGIONS = {
    "Москва": {"stores": 8, "online_share": 0.38},
    "Санкт-Петербург": {"stores": 5, "online_share": 0.22},
    "Казань": {"stores": 3, "online_share": 0.15},
    "Новосибирск": {"stores": 3, "online_share": 0.14},
    "Екатеринбург": {"stores": 3, "online_share": 0.16},
}

CATEGORIES = {
    "Лекарства (рецепт)": {"base_price": 420, "margin": 0.18, "daily_qty": 2.8},
    "Лекарства (без рецепта)": {"base_price": 280, "margin": 0.25, "daily_qty": 4.5},
    "Косметика и гигиена": {"base_price": 350, "margin": 0.32, "daily_qty": 3.2},
    "БАД и витамины": {"base_price": 520, "margin": 0.35, "daily_qty": 2.1},
}

PAYMENT_METHODS = ["Наличные", "Карта", "Онлайн"]
GENDERS = ["Женский", "Мужской"]
AGE_GROUPS = ["18-30", "31-45", "46-60", "60+"]


def generate_products(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    product_id = 1
    for category, params in CATEGORIES.items():
        n_products = rng.integers(18, 28)
        for _ in range(n_products):
            price = max(45, rng.normal(params["base_price"], params["base_price"] * 0.25))
            rows.append(
                {
                    "product_id": product_id,
                    "product_name": f"{category.split()[0]} #{product_id}",
                    "category": category,
                    "price": round(price, 2),
                    "margin_pct": round(params["margin"] + rng.normal(0, 0.03), 3),
                }
            )
            product_id += 1
    return pd.DataFrame(rows)


def choose_payment(
    rng: np.random.Generator,
    region: str,
    age_group: str,
) -> str:
    online_boost = 0.25 if region == "Москва" else 0.0
    young_boost = 0.18 if age_group == "18-30" else 0.0
    elderly_penalty = 0.12 if age_group == "60+" else 0.0

    probs = np.array([0.22, 0.48, 0.30], dtype=float)
    probs[2] += online_boost + young_boost - elderly_penalty
    probs[0] += elderly_penalty * 0.6
    probs[1] -= (online_boost + young_boost) * 0.5
    probs = np.clip(probs, 0.05, None)
    probs /= probs.sum()
    return rng.choice(PAYMENT_METHODS, p=probs)


def generate_transactions(
    products: pd.DataFrame,
    rng: np.random.Generator,
    n_days: int = 180,
) -> pd.DataFrame:
    start_date = pd.Timestamp("2025-01-01")
    promo_start = start_date + pd.Timedelta(days=90)
    promo_end = promo_start + pd.Timedelta(days=21)

    store_meta = []
    store_id = 1
    for region, meta in REGIONS.items():
        for _ in range(meta["stores"]):
            store_meta.append({"store_id": store_id, "region": region})
            store_id += 1
    stores_df = pd.DataFrame(store_meta)

    rows = []
    transaction_id = 1

    for day_offset in range(n_days):
        current_date = start_date + pd.Timedelta(days=day_offset)
        is_promo = promo_start <= current_date <= promo_end
        weekday = current_date.weekday()
        weekend_boost = 1.12 if weekday >= 5 else 1.0
        promo_boost = 1.18 if is_promo else 1.0

        daily_transactions = int(rng.poisson(95 * weekend_boost * promo_boost))

        for _ in range(daily_transactions):
            store = stores_df.sample(1, random_state=rng.integers(0, 1_000_000)).iloc[0]
            product = products.sample(1, weights=1 / products["price"], random_state=rng.integers(0, 1_000_000)).iloc[0]

            gender = rng.choice(GENDERS, p=[0.58, 0.42])
            age_group = rng.choice(AGE_GROUPS, p=[0.22, 0.31, 0.27, 0.20])

            base_qty = CATEGORIES[product["category"]]["daily_qty"]
            price_effect = 1.4 - min(product["price"] / 900, 0.9)
            qty = max(1, int(rng.poisson(base_qty * price_effect)))

            gender_check_boost = 1.08 if gender == "Женский" else 1.0
            discount = 0.0
            if is_promo and rng.random() < 0.35:
                discount = rng.choice([0.05, 0.10, 0.15])

            amount = round(product["price"] * qty * gender_check_boost * (1 - discount), 2)
            payment = choose_payment(rng, store["region"], age_group)

            rows.append(
                {
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
                }
            )
            transaction_id += 1

    return pd.DataFrame(rows)


def main(output_dir: Path) -> None:
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

    print(f"Сохранено {len(products)} товаров и {len(transactions)} транзакций в {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генерация данных аптечной сети")
    parser.add_argument("--output", type=Path, default=Path("data"), help="Папка для CSV")
    args = parser.parse_args()
    main(args.output)

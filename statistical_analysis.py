# Анализ продаж аптечной сети — финальный проект по статистике

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
ALPHA = 0.05

plt.rcParams["figure.figsize"] = (10, 6)
sns.set_theme(style="whitegrid")


def load_data():
    transactions = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["date"])
    products = pd.read_csv(DATA_DIR / "products.csv")
    daily = pd.read_csv(DATA_DIR / "daily_revenue.csv", parse_dates=["date"])
    return transactions, products, daily


def descriptive_analysis(transactions, daily, output_dir):
    transactions["weekday"] = transactions["date"].dt.day_name()
    transactions["month"] = transactions["date"].dt.to_period("M").astype(str)

    summary = pd.DataFrame({
        "показатель": [
            "Число транзакций",
            "Средний чек, руб.",
            "Медианный чек, руб.",
            "Среднее кол-во единиц в чеке",
            "Средняя дневная выручка, руб.",
            "Станд. откл. дневной выручки",
            "Коэфф. вариации дневной выручки",
        ],
        "значение": [
            len(transactions),
            transactions["amount"].mean(),
            transactions["amount"].median(),
            transactions["quantity"].mean(),
            daily["daily_revenue"].mean(),
            daily["daily_revenue"].std(ddof=1),
            daily["daily_revenue"].std(ddof=1) / daily["daily_revenue"].mean(),
        ],
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sns.histplot(transactions["amount"], bins=40, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Распределение суммы чека")
    axes[0, 0].set_xlabel("Сумма, руб.")

    sns.boxplot(data=transactions, x="category", y="amount", ax=axes[0, 1])
    axes[0, 1].set_title("Чек по категориям")
    axes[0, 1].tick_params(axis="x", rotation=20)

    sns.lineplot(data=daily, x="date", y="daily_revenue", ax=axes[1, 0])
    axes[1, 0].set_title("Дневная выручка по датам")

    region_revenue = transactions.groupby("region")["amount"].sum().sort_values(ascending=False)
    region_revenue.plot(kind="bar", ax=axes[1, 1], color="steelblue")
    axes[1, 1].set_title("Выручка по регионам")
    axes[1, 1].set_ylabel("руб.")
    axes[1, 1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    fig.savefig(output_dir / "01_descriptive_overview.png", dpi=150)
    plt.close(fig)

    summary.to_csv(output_dir / "descriptive_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def run_hypothesis_tests(transactions, daily):
    results = []

    # 1. чек: женщины vs мужчины
    female = transactions.loc[transactions["gender"] == "Женский", "amount"]
    male = transactions.loc[transactions["gender"] == "Мужской", "amount"]
    _, levene_p = stats.levene(female, male)
    equal_var = levene_p >= ALPHA
    t_stat, t_p = stats.ttest_ind(female, male, equal_var=equal_var, alternative="greater")
    results.append({
        "гипотеза": "Средний чек женщин выше, чем у мужчин",
        "критерий": "t-критерий Стьюдента",
        "статистика": round(t_stat, 4),
        "p_value": round(t_p, 6),
        "решение": "Отвергаем H0" if t_p < ALPHA else "Не отвергаем H0",
    })

    # 2. промо vs обычные дни
    promo_days = transactions.loc[transactions["is_promo_period"] == 1, "date"].drop_duplicates()
    regular_days = transactions.loc[transactions["is_promo_period"] == 0, "date"].drop_duplicates()
    n_pairs = min(len(promo_days), len(regular_days))
    promo_sample = promo_days.sample(n_pairs, random_state=42).sort_values()
    regular_sample = regular_days.sample(n_pairs, random_state=42).sort_values()

    promo_rev = transactions[transactions["date"].isin(promo_sample)].groupby("date")["amount"].sum().values
    regular_rev = transactions[transactions["date"].isin(regular_sample)].groupby("date")["amount"].sum().values
    paired_t, paired_p = stats.ttest_rel(promo_rev, regular_rev, alternative="greater")
    results.append({
        "гипотеза": "Дневная выручка в промо-период выше",
        "критерий": "парный t-критерий",
        "статистика": round(paired_t, 4),
        "p_value": round(paired_p, 6),
        "решение": "Отвергаем H0" if paired_p < ALPHA else "Не отвергаем H0",
    })

    # 3. чек по категориям
    groups = [g["amount"].values for _, g in transactions.groupby("category")]
    f_stat, f_p = stats.f_oneway(*groups)
    results.append({
        "гипотеза": "Средний чек различается по категориям",
        "критерий": "ANOVA",
        "статистика": round(f_stat, 4),
        "p_value": round(f_p, 6),
        "решение": "Отвергаем H0" if f_p < ALPHA else "Не отвергаем H0",
    })

    # 4. оплата и возраст
    pay_table = pd.crosstab(transactions["age_group"], transactions["payment_method"])
    chi2, chi_p, dof, _ = stats.chi2_contingency(pay_table)
    results.append({
        "гипотеза": "Способ оплаты зависит от возраста",
        "критерий": "хи-квадрат",
        "статистика": round(chi2, 4),
        "p_value": round(chi_p, 6),
        "решение": "Отвергаем H0" if chi_p < ALPHA else "Не отвергаем H0",
    })

    # 5. цена и количество
    r, r_p = stats.pearsonr(transactions["price"], transactions["quantity"])
    results.append({
        "гипотеза": "Цена отрицательно связана с количеством",
        "критерий": "корреляция Пирсона",
        "статистика": round(r, 4),
        "p_value": round(r_p, 6),
        "решение": "Отвергаем H0" if r_p < ALPHA and r < 0 else "Не отвергаем H0",
    })

    # 6. онлайн-оплата: Москва vs остальные
    moscow = transactions[transactions["region"] == "Москва"]
    other = transactions[transactions["region"] != "Москва"]
    count = np.array([
        (moscow["payment_method"] == "Онлайн").sum(),
        (other["payment_method"] == "Онлайн").sum(),
    ])
    nobs = np.array([len(moscow), len(other)])
    z_stat, z_p = proportions_ztest(count, nobs, alternative="larger")
    results.append({
        "гипотеза": "Доля онлайн-оплаты в Москве выше",
        "критерий": "z-тест для долей",
        "статистика": round(z_stat, 4),
        "p_value": round(z_p, 6),
        "решение": "Отвергаем H0" if z_p < ALPHA else "Не отвергаем H0",
    })

    # 7. выходные vs будни
    daily_ext = daily.copy()
    daily_ext["is_weekend"] = daily_ext["date"].dt.weekday >= 5
    weekend = daily_ext.loc[daily_ext["is_weekend"], "daily_revenue"]
    weekday = daily_ext.loc[~daily_ext["is_weekend"], "daily_revenue"]
    u_stat, u_p = stats.mannwhitneyu(weekend, weekday, alternative="greater")
    results.append({
        "гипотеза": "Выручка в выходные выше",
        "критерий": "Манна-Уитни",
        "статистика": round(u_stat, 4),
        "p_value": round(u_p, 6),
        "решение": "Отвергаем H0" if u_p < ALPHA else "Не отвергаем H0",
    })

    return pd.DataFrame(results)


def build_visualizations(transactions, daily, output_dir):
    sample = transactions.sample(min(2500, len(transactions)), random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=sample, x="price", y="quantity", alpha=0.35, ax=ax)
    ax.set_title("Цена и количество в чеке")
    fig.savefig(output_dir / "02_price_quantity.png", dpi=150)
    plt.close(fig)

    pay_table = pd.crosstab(transactions["age_group"], transactions["payment_method"], normalize="index")
    pay_table.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="Set2")
    plt.title("Способы оплаты по возрасту")
    plt.ylabel("доля")
    plt.tight_layout()
    plt.savefig(output_dir / "03_payment_age.png", dpi=150)
    plt.close()

    promo_daily = transactions.groupby(["date", "is_promo_period"])["amount"].sum().reset_index()
    sns.lineplot(data=promo_daily, x="date", y="amount", hue="is_promo_period")
    plt.title("Выручка: промо и обычные дни")
    plt.tight_layout()
    plt.savefig(output_dir / "04_promo_effect.png", dpi=150)
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transactions, products, daily = load_data()

    summary = descriptive_analysis(transactions, daily, OUTPUT_DIR)
    print("Описательная статистика:")
    print(summary.to_string(index=False))

    tests = run_hypothesis_tests(transactions, daily)
    tests.to_csv(OUTPUT_DIR / "hypothesis_tests.csv", index=False, encoding="utf-8-sig")
    print("\nГипотезы:")
    print(tests.to_string(index=False))
    print(f"\nРезультаты в {OUTPUT_DIR.resolve()}")

    build_visualizations(transactions, daily, OUTPUT_DIR)


if __name__ == "__main__":
    main()

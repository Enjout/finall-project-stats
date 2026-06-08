"""
Статистический анализ продаж аптечной сети «ФармаПлюс».

Запуск:
    python statistical_analysis.py

Результаты сохраняются в папку output/ (графики и сводная таблица тестов).
"""

from __future__ import annotations

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
plt.rcParams["font.family"] = "DejaVu Sans"
sns.set_theme(style="whitegrid")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transactions = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["date"])
    products = pd.read_csv(DATA_DIR / "products.csv")
    daily = pd.read_csv(DATA_DIR / "daily_revenue.csv", parse_dates=["date"])
    return transactions, products, daily


def descriptive_analysis(transactions: pd.DataFrame, daily: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    transactions["weekday"] = transactions["date"].dt.day_name()
    transactions["month"] = transactions["date"].dt.to_period("M").astype(str)

    summary = pd.DataFrame(
        {
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
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    sns.histplot(transactions["amount"], bins=40, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Распределение суммы чека")
    axes[0, 0].set_xlabel("Сумма, руб.")

    sns.boxplot(data=transactions, x="category", y="amount", ax=axes[0, 1])
    axes[0, 1].set_title("Чек по категориям товаров")
    axes[0, 1].tick_params(axis="x", rotation=20)

    daily_plot = daily.copy()
    daily_plot["weekday_num"] = daily_plot["date"].dt.weekday
    sns.lineplot(data=daily_plot, x="date", y="daily_revenue", ax=axes[1, 0])
    axes[1, 0].set_title("Динамика дневной выручки")
    axes[1, 0].set_xlabel("Дата")

    region_revenue = transactions.groupby("region")["amount"].sum().sort_values(ascending=False)
    region_revenue.plot(kind="bar", ax=axes[1, 1], color="steelblue")
    axes[1, 1].set_title("Совокупная выручка по регионам")
    axes[1, 1].set_ylabel("Выручка, руб.")
    axes[1, 1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    fig.savefig(output_dir / "01_descriptive_overview.png", dpi=150)
    plt.close(fig)

    summary.to_csv(output_dir / "descriptive_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def run_hypothesis_tests(transactions: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    results = []

    # H1: средний чек женщин > мужчин (две независимые выборки, t-критерий)
    female_checks = transactions.loc[transactions["gender"] == "Женский", "amount"]
    male_checks = transactions.loc[transactions["gender"] == "Мужской", "amount"]
    levene_stat, levene_p = stats.levene(female_checks, male_checks)
    equal_var = levene_p >= ALPHA
    t_stat, t_p = stats.ttest_ind(female_checks, male_checks, equal_var=equal_var, alternative="greater")
    results.append(
        {
            "гипотеза": "H1: средний чек женщин выше, чем у мужчин",
            "тип_данных": "количественные, 2 независимые группы",
            "критерий": "t-критерий Стьюдента (односторонний)",
            "H0": "μ_жен = μ_муж",
            "статистика": round(t_stat, 4),
            "p_value": round(t_p, 6),
            "решение": "Отвергаем H0" if t_p < ALPHA else "Не отвергаем H0",
            "примечание": f"Levene p={levene_p:.4f}, equal_var={equal_var}",
        }
    )

    # H2: дневная выручка в промо-период выше (парные наблюдения по датам магазинов)
    store_daily = transactions.groupby(["date", "store_id"], as_index=False)["amount"].sum()
    promo_days = transactions.loc[transactions["is_promo_period"] == 1, "date"].drop_duplicates()
    non_promo_days = transactions.loc[transactions["is_promo_period"] == 0, "date"].drop_duplicates()
    n_pairs = min(len(promo_days), len(non_promo_days))
    promo_sample = promo_days.sample(n_pairs, random_state=42).sort_values()
    non_promo_sample = non_promo_days.sample(n_pairs, random_state=42).sort_values()

    promo_rev = (
        store_daily[store_daily["date"].isin(promo_sample)]
        .groupby("date")["amount"]
        .sum()
        .values
    )
    regular_rev = (
        store_daily[store_daily["date"].isin(non_promo_sample)]
        .groupby("date")["amount"]
        .sum()
        .values
    )
    paired_t, paired_p = stats.ttest_rel(promo_rev, regular_rev, alternative="greater")
    results.append(
        {
            "гипотеза": "H2: средняя дневная выручка в промо-период выше обычной",
            "тип_данных": "количественные, парные наблюдения",
            "критерий": "парный t-критерий",
            "H0": "μ_промо = μ_обыч",
            "статистика": round(paired_t, 4),
            "p_value": round(paired_p, 6),
            "решение": "Отвергаем H0" if paired_p < ALPHA else "Не отвергаем H0",
            "примечание": f"Сравнено {n_pairs} пар дней",
        }
    )

    # H3: средний чек различается по категориям (ANOVA)
    groups = [group["amount"].values for _, group in transactions.groupby("category")]
    f_stat, f_p = stats.f_oneway(*groups)
    results.append(
        {
            "гипотеза": "H3: средний чек различается между категориями товаров",
            "тип_данных": "количественные, 4+ группы",
            "критерий": "однофакторный ANOVA",
            "H0": "μ_1 = μ_2 = μ_3 = μ_4",
            "статистика": round(f_stat, 4),
            "p_value": round(f_p, 6),
            "решение": "Отвергаем H0" if f_p < ALPHA else "Не отвергаем H0",
            "примечание": "При отвержении H0 нужен post-hoc (Tukey)",
        }
    )

    # H4: способ оплаты зависит от возрастной группы (хи-квадрат)
    pay_table = pd.crosstab(transactions["age_group"], transactions["payment_method"])
    chi2, chi_p, dof, _ = stats.chi2_contingency(pay_table)
    results.append(
        {
            "гипотеза": "H4: способ оплаты зависит от возрастной группы",
            "тип_данных": "категориальные",
            "критерий": "χ² Пирсона",
            "H0": "способ оплаты и возраст независимы",
            "статистика": round(chi2, 4),
            "p_value": round(chi_p, 6),
            "решение": "Отвергаем H0" if chi_p < ALPHA else "Не отвергаем H0",
            "примечание": f"df={dof}",
        }
    )

    # H5: цена и количество отрицательно коррелируют (количественные)
    r, r_p = stats.pearsonr(transactions["price"], transactions["quantity"])
    results.append(
        {
            "гипотеза": "H5: цена товара отрицательно связана с количеством в чеке",
            "тип_данных": "количественные, связь двух переменных",
            "критерий": "корреляция Пирсона",
            "H0": "ρ = 0",
            "статистика": round(r, 4),
            "p_value": round(r_p, 6),
            "решение": "Отвергаем H0" if r_p < ALPHA and r < 0 else "Не отвергаем H0",
            "примечание": "Отрицательный r подтверждает гипотезу",
        }
    )

    # H6: доля онлайн-оплаты в Москве выше, чем в регионах (пропорции)
    moscow = transactions[transactions["region"] == "Москва"]
    regions = transactions[transactions["region"] != "Москва"]
    count = np.array(
        [
            (moscow["payment_method"] == "Онлайн").sum(),
            (regions["payment_method"] == "Онлайн").sum(),
        ]
    )
    nobs = np.array([len(moscow), len(regions)])
    z_stat, z_p = proportions_ztest(count, nobs, alternative="larger")
    results.append(
        {
            "гипотеза": "H6: доля онлайн-оплаты в Москве выше, чем в других регионах",
            "тип_данных": "бинарные / доли",
            "критерий": "z-критерий для двух пропорций",
            "H0": "p_Москва = p_регионы",
            "статистика": round(z_stat, 4),
            "p_value": round(z_p, 6),
            "решение": "Отвергаем H0" if z_p < ALPHA else "Не отвергаем H0",
            "примечание": f"Онлайн Москва: {count[0]/nobs[0]:.1%}, регионы: {count[1]/nobs[1]:.1%}",
        }
    )

    # H7: дневная выручка в выходные выше будней (Mann-Whitney — непараметрический)
    daily_ext = daily.copy()
    daily_ext["is_weekend"] = daily_ext["date"].dt.weekday >= 5
    weekend_rev = daily_ext.loc[daily_ext["is_weekend"], "daily_revenue"]
    weekday_rev = daily_ext.loc[~daily_ext["is_weekend"], "daily_revenue"]
    u_stat, u_p = stats.mannwhitneyu(weekend_rev, weekday_rev, alternative="greater")
    results.append(
        {
            "гипотеза": "H7: дневная выручка в выходные выше, чем в будни",
            "тип_данных": "количественные, независимые (непараметрические)",
            "критерий": "критерий Манна-Уитни",
            "H0": "распределения выручки одинаковы",
            "статистика": round(u_stat, 4),
            "p_value": round(u_p, 6),
            "решение": "Отвергаем H0" if u_p < ALPHA else "Не отвергаем H0",
            "примечание": "Используем при возможном отклонении от нормальности",
        }
    )

    # H8: нормальность дневной выручки (диагностическая)
    shapiro_stat, shapiro_p = stats.shapiro(daily["daily_revenue"])
    results.append(
        {
            "гипотеза": "H8: дневная выручка распределена нормально",
            "тип_данных": "количественные, проверка распределения",
            "критерий": "Шапиро-Уилка",
            "H0": "данные из нормального распределения",
            "статистика": round(shapiro_stat, 4),
            "p_value": round(shapiro_p, 6),
            "решение": "Отвергаем H0" if shapiro_p < ALPHA else "Не отвергаем H0",
            "примечание": "Влияет на выбор параметрических критериев",
        }
    )

    return pd.DataFrame(results)


def build_visualizations(transactions: pd.DataFrame, daily: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=transactions.sample(min(2500, len(transactions)), random_state=42), x="price", y="quantity", alpha=0.35, ax=ax)
    ax.set_title("Связь цены и количества в чеке")
    fig.savefig(output_dir / "02_price_quantity.png", dpi=150)
    plt.close(fig)

    pay_table = pd.crosstab(transactions["age_group"], transactions["payment_method"], normalize="index")
    pay_table.plot(kind="bar", stacked=True, figsize=(10, 6), colormap="Set2")
    plt.title("Структура способов оплаты по возрастным группам")
    plt.ylabel("Доля")
    plt.xlabel("Возрастная группа")
    plt.legend(title="Оплата", bbox_to_anchor=(1.02, 1))
    plt.tight_layout()
    plt.savefig(output_dir / "03_payment_age.png", dpi=150)
    plt.close()

    promo_daily = transactions.groupby(["date", "is_promo_period"])["amount"].sum().reset_index()
    sns.lineplot(data=promo_daily, x="date", y="amount", hue="is_promo_period", palette="Set1")
    plt.title("Дневная выручка: промо vs обычные дни")
    plt.tight_layout()
    plt.savefig(output_dir / "04_promo_effect.png", dpi=150)
    plt.close()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    transactions, products, daily = load_data()

    print("=== Базовый анализ ===")
    summary = descriptive_analysis(transactions, daily, OUTPUT_DIR)
    print(summary.to_string(index=False))

    print("\n=== Проверка гипотез ===")
    tests = run_hypothesis_tests(transactions, daily)
    tests.to_csv(OUTPUT_DIR / "hypothesis_tests.csv", index=False, encoding="utf-8-sig")
    print(tests[["гипотеза", "критерий", "p_value", "решение"]].to_string(index=False).encode("utf-8", errors="replace").decode("utf-8"))

    build_visualizations(transactions, daily, OUTPUT_DIR)
    print(f"\nГрафики и таблицы сохранены в {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

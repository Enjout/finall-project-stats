# Финальный проект по статистике

Анализ продаж аптечной сети: описательная статистика, проверка гипотез, графики.

**Отчёт:** [ОТЧЕТ_финальный_проект.md](ОТЧЕТ_финальный_проект.md)

## Запуск

```bash
pip install -r requirements.txt
python generate_data.py
python statistical_analysis.py
```

## Файлы

- `generate_data.py` — формирование датасета
- `statistical_analysis.py` — расчёты и тесты
- `data/` — csv с транзакциями
- `output/` — графики и таблицы с результатами

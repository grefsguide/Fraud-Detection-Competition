# Fraud Detection Competition

Этот репозиторий/ноутбук содержит полный ML‑пайплайн для соревнования **Fraud Detection Competition**: от базового решения и EDA до сравнения моделей, экспериментов с поиском аномалий/кластеризацией и AutoML.

## Соревнование

- Постановка: бинарная классификация транзакций на **фрод / не фрод**
- Таргет: `isFraud` (0/1)
- Метрика: **ROC-AUC**
- Формат submission: 2 колонки — `TransactionID`, `isFraud` (вероятность класса *fraud* для каждой транзакции)

Ссылка на соревнование:

```text
https://www.kaggle.com/competitions/fraud-detection-24
```

---

## Данные

Используются стандартные файлы соревнования:

- `train_transaction.csv`
- `train_identity.csv`
- `test_transaction.csv`
- `test_identity.csv`
- `sample_submission.csv`

### Где взять данные

В ноутбуке есть ячейка с загрузкой файлов через `wget`. Если вы используете её, обратите внимание на путь к данным (см. раздел **Запуск**).

---

## Структура проекта

Ожидаемая структура (по умолчанию в ноутбуке):

```text
.
├─ raw_data/
│  ├─ train_transaction.csv
│  ├─ train_identity.csv
│  ├─ test_transaction.csv
│  ├─ test_identity.csv
│  └─ sample_submission.csv
├─ outputs/              # сабмиты из базового/ML/AutoML пайплайна
├─ anomaly_outputs/      # сабмиты из аномалий/пайплайна с аномальными признаками
└─ clusters_outputs/     # сабмиты из пайплайна с кластерными признаками
```

---

## Установка зависимостей

Минимальный набор библиотек:

```bash
pip install -U pip
pip install numpy pandas scipy scikit-learn matplotlib seaborn tqdm \
            catboost lightgbm xgboost flaml
```

Среда: **Python 3.x**, запуск в Jupyter/Colab/Kaggle.

---

## Запуск

1) Скачайте данные соревнования и положите их в `raw_data/`.

2) Откройте ноутбук `fraud_detection_competition.ipynb` и выполните ячейки сверху вниз.

### Если вы скачиваете данные через `wget` в ноутбуке

`wget` сохраняет CSV в текущую директорию. При этом переменная:

```python
INPUT_DIR = "raw_data"
```

Ожидает файлы в папке `raw_data/`.

Варианты:
- либо переместите скачанные CSV в `raw_data/`,
- либо поменяйте `INPUT_DIR` на `"."`.

---

## Что внутри ноутбука

### 0) Базовое решение + пример submission
- Быстрый baseline на **CatBoost**
- Пример формирования файла submission

> Примечание: в baseline важно сохранять именно вероятность класса **fraud** (обычно это `predict_proba(... )[:, 1]`), а не нулевого класса.

### 1) EDA
- Дисбаланс классов
- Распределения числовых признаков
- Связь фичей с таргетом (в т.ч. пример с `dist_sum`)
- Пропуски, корреляции, выбросы
- Черновая очистка: дубли, заполнение NaN, фильтрация явных выбросов по z-score

### 2) Обучение моделей
Единый препроцессинг:
- числовые: `SimpleImputer(fill_value=-999)`
- категориальные: `SimpleImputer(fill_value="-") + OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)`
- `ColumnTransformer` + `Pipeline`

Сравниваются модели:
- **CatBoost**
- **LightGBM**
- **XGBoost**
- **RandomForest**
- **LogisticRegression**

Оценка качества:
- Stratified K-Fold CV (ROC-AUC)
- + фиксация лидерборд‑результатов (как в ноутбуке) для сравнения

### 3) Поиск аномалий
Два подхода:
1) как **standalone anomaly detection** (строим score → ранговая нормализация → сабмит)
2) как **feature engineering**: добавляем аномальные скоры как новые признаки и переобучаем supervised‑модели

Методы:
- IQR (доля выбросов по признакам)
- EllipticEnvelope (+ PCA)
- IsolationForest
- KNN‑скор (по расстояниям до соседей)

### 4) Кластеризация
- DBSCAN на стандартизированных данных
- Добавление признаков: `dbscan_cluster`, `dbscan_is_noise` + простая target‑encoding фича `dbscan_cluster_te`
- Переобучение supervised‑моделей и сравнение качества

### 5) AutoML
- `AutoML` (estimator list: lgbm/xgboost/rf/lr/catboost)
- Ограничение по времени (`time_budget=3600`)
- Генерация сабмита `outputs/submission_automl.csv`

---

## Артефакты и результаты

В ходе выполнения ноутбука формируются CSV‑файлы для отправки на Kaggle:

- `outputs/submission_baseline.csv` — baseline
- `outputs/submission_<model>.csv` — сабмиты для сравниваемых supervised моделей (в разделе 2)
- `anomaly_outputs/submission_anom_*.csv` и `anomaly_outputs/submission_with_anom_*.csv`
- `clusters_outputs/submission_clusters_*.csv`
- `outputs/submission_automl.csv` — AutoML

---

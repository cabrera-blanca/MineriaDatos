# Informe — Entrega 1

## 1. Resumen
Se construyó un pipeline reproducible (Git + DVC + MLflow) que compara 9 configuraciones sobre el histórico de 7.043 clientes y registra un modelo candidato en el Model Registry.

**Candidato: `logreg_C0.1`** (LogisticRegression, C=0.1, threshold 0.18) → registrado como `customer-churn-classifier` v1, alias `candidate`.

## 2. EDA (ver `notebooks/01_eda.ipynb`)
- 7.043 filas × 21 columnas; sin duplicados. Único problema de calidad: 26 nulos en `TotalCharges`, todos con `tenure = 0` → se imputan con 0.
- Target desbalanceado: 26,4 % churn / 73,6 % no churn. Un modelo que siempre predice "No" tendría 73,6 % de accuracy → **accuracy no es suficiente**.
- Churn por Contract: month-to-month 38,7 %, one year 15,0 %, two year 9,0 %. Fibra óptica 40,3 % vs DSL 20,0 %. El churn baja con la antigüedad (41 % en ≤6 meses vs 17 % en >48).
- `gender`, `PaperlessBilling` y `SeniorCitizen` casi no discriminan.

## 3. Comparación de modelos
Métricas de CV (5 folds, solo train) y de test aislado (20 %, 1.409 filas) **usando el threshold ajustado en CV** (baseline: 0.5).

| name | type | threshold | cv_roc_auc | cv_f2 | test_precision | test_recall | test_f1 | test_f2 | test_roc_auc |
|---|---|---|---|---|---|---|---|---|---|
| logreg_C0.1 | logreg | 0.18 | 0.827 | 0.725 | 0.437 | 0.839 | 0.575 | 0.708 | 0.812 |
| logreg_C1 | logreg | 0.17 | 0.827 | 0.725 | 0.437 | 0.844 | 0.576 | 0.712 | 0.812 |
| logreg_C1_balanced | logreg | 0.37 | 0.827 | 0.724 | 0.437 | 0.836 | 0.574 | 0.707 | 0.812 |
| logreg_C0.01 | logreg | 0.22 | 0.825 | 0.716 | 0.446 | 0.831 | 0.580 | 0.708 | 0.810 |
| gb_d3 | gb | 0.16 | 0.821 | 0.724 | 0.418 | 0.847 | 0.560 | 0.703 | 0.808 |
| rf_d10_balanced | rf | 0.41 | 0.818 | 0.710 | 0.439 | 0.796 | 0.566 | 0.685 | 0.802 |
| rf_d10 | rf | 0.23 | 0.818 | 0.707 | 0.453 | 0.785 | 0.575 | 0.685 | 0.805 |
| rf_d5 | rf | 0.23 | 0.814 | 0.717 | 0.423 | 0.860 | 0.567 | 0.713 | 0.801 |
| dummy_prior | dummy | 0.50 | 0.500 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 |

`cv_roc_auc` decide la selección; las columnas `test_*` son solo reporte.

## 4. Selección del candidato
- La regresión logística obtiene el mejor ROC-AUC en CV (0,827) y también en test (0,812). RandomForest y GradientBoosting no la superan (0,814–0,821 en CV): **el modelo más simple e interpretable gana o empata**, y además es rápido y estable.
- Las diferencias entre C=0.1, C=1 y `class_weight=balanced` son de milésimas (prácticamente empate); se aplica el criterio determinista definido de antemano (ROC-AUC, luego F2 en CV) y se toma C=0.1, el más regularizado.
- El baseline (ROC-AUC 0,50, recall 0) confirma que los modelos aprenden señal real.

## 5. Impacto del falso negativo y threshold
Un **falso negativo** es un cliente que iba a abandonar y el modelo lo clasifica como estable: la empresa no le ofrece retención y pierde el ingreso recurrente completo. Un **falso positivo** solo cuesta una oferta de retención innecesaria, en general mucho más barata que perder al cliente. Por eso se prioriza **recall** y se ajusta el threshold con F1.5 en lugar de usar 0.5.

Con threshold 0.18, en el test del candidato: TP=312, FN=60, FP=402, TN=635 → **recall 0,839, precision 0,437, F1 0,575, ROC-AUC 0,812**. Se detecta ~84 % de los que abandonan, a costa de que solo ~44 % de los alertados abandona realmente. El costo relativo real (retención vs. cliente perdido) debería fijar el threshold definitivo con el área comercial; queda parametrizado en `params.yaml`.

## 6. Trazabilidad
Modelo registrado v1 → `source_run_id` (run de MLflow) → tags `git_commit` y `data_md5` (hash del dataset en `data/raw/customer_churn_historical.csv.dvc`). `params.yaml` queda como artefacto del run.

## 7. Limitaciones y próximos pasos
- ROC-AUC ≈ 0,81: margen de mejora limitado con estos datos (posible feature engineering o tuning más amplio).
- Los niveles de riesgo LOW/MEDIUM/HIGH (`params.yaml`: 0.18 / 0.50) son provisorios y se validarán en la Entrega 2.
- `data/production` no se usó (reservado para monitoreo/drift, Entrega final).

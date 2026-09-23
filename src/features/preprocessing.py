"""Preprocesamiento como parte del Pipeline de scikit-learn (mismo código en train e inferencia)."""
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
BINARY_NUM = ["SeniorCitizen"]
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
FEATURES = NUMERIC + BINARY_NUM + CATEGORICAL


def build_preprocessor(scale: bool = True) -> ColumnTransformer:
    """Imputación (0 para numéricas: los faltantes de TotalCharges son clientes con tenure=0, aún sin facturación) -> escalado (opcional, solo modelos lineales) -> OneHot."""
    num_steps = [("imputer", SimpleImputer(strategy="constant", fill_value=0))]
    if scale:
        num_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), NUMERIC),
            ("bin", "passthrough", BINARY_NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), CATEGORICAL),
        ],
        remainder="drop",  # customerID y cualquier otra columna quedan fuera
    )

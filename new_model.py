"""
=============================================================================
PROYECTO: Predicción de Abandono de Clientes en Telecomunicaciones (Churn)
METODOLOGÍA: CRISP-DM
MODELOS: Árbol de Decisión vs. Random Forest
=============================================================================
"""

import subprocess
import sys
import os
import warnings

# ─── INSTALACIÓN AUTOMÁTICA DE DEPENDENCIAS ───────────────────────────────────
def instalar_dependencias():
    paquetes = ["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn", "imbalanced-learn", "tabulate"]
    print("=" * 65)
    print("  Verificando/Instalando dependencias...")
    print("=" * 65)
    for pkg in paquetes:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "-q", "--break-system-packages"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print("  ✓ Todas las dependencias instaladas correctamente.\n")

instalar_dependencias()

# ─── IMPORTACIONES ────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend sin pantalla
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ─── CONFIGURACIÓN GLOBAL ─────────────────────────────────────────────────────
COLORES = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
SALIDA = "outputs/graficas_comparativas"
os.makedirs(SALIDA, exist_ok=True)

sns.set_theme(style="whitegrid", palette=COLORES)
plt.rcParams.update({"figure.dpi": 130, "font.family": "DejaVu Sans"})

def guardar(fig, nombre):
    ruta = os.path.join(SALIDA, nombre)
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Gráfica guardada → {ruta}")

# =============================================================================
# FASES 1 A 3 – NEGOCIO, EDA Y PREPARACIÓN (Mantenemos la estructura)
# =============================================================================
def fase1_comprension_negocio():
    print("\n" + "=" * 65)
    print("  FASE 1 · COMPRENSIÓN DEL NEGOCIO")
    print("=" * 65)
    print("""
  Objetivo Actualizado:
    Construir y comparar un modelo de Árbol de Decisión y un
    Bosque Aleatorio (Random Forest) para predecir la probabilidad
    de abandono (Churn). Seleccionar el mejor modelo según las
    métricas de negocio.
    """)

def fase2_y_3_preparacion(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("  FASES 2 Y 3 · PREPARACIÓN Y LIMPIEZA DE DATOS")
    print("=" * 65)
    
    # 3.1 Eliminar ID
    df.drop(columns=["customerID"], inplace=True)
    
    # 3.2 y 3.3 Tratamiento de numéricas
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    
    # 3.4 Codificación variable objetivo
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    # 3.5 Ingeniería de características (Simplificada)
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12m", "13-24m", "25-48m", "49-72m"],
        include_lowest=True  # <--- CRUCIAL: incluye a los clientes con tenure = 0
    )
    df["HighMonthlyCharge"] = (df["MonthlyCharges"] > df["MonthlyCharges"].median()).astype(int)
    
    # Limpieza preventiva: eliminar cualquier nulo residual (como los 11 de TotalCharges si fallara algo)
    antes = len(df)
    df.dropna(inplace=True)
    despues = len(df)
    if antes != despues:
        print(f"  ⚠ Se eliminaron {antes - despues} filas con valores nulos residuales.")
    
    # 3.6 Label Encoding para compatibilidad con modelos de árboles
    le = LabelEncoder()
    categoricas = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categoricas:
        df[col] = le.fit_transform(df[col].astype(str))
        
    print("  ✓ Datos limpios, nulos imputados y variables categóricas codificadas.")
    return df

# =============================================================================
# FASE 4 – MODELADO (¡CORREGIDO METODOLÓGICAMENTE!)
# =============================================================================
def fase4_modelado(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("  FASE 4 · MODELADO – ÁRBOL VS RANDOM FOREST")
    print("=" * 65)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 1. SPLIT PRIMERO (Evita Fuga de Datos)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"  ✓ Split Realizado | Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    # 2. SMOTE SOLO EN ENTRENAMIENTO
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print("  ✓ SMOTE aplicado al set de entrenamiento.")

    # 3. Entrenamiento: Árbol de Decisión (Baseline)
    dt_model = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=42)
    dt_model.fit(X_train_res, y_train_res)
    print("  ✓ Árbol de Decisión entrenado.")

    # 4. Entrenamiento: Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=10, class_weight="balanced", random_state=42)
    rf_model.fit(X_train_res, y_train_res)
    print("  ✓ Random Forest entrenado (100 árboles).")

    return dt_model, rf_model, X_train_res, X_test, y_train_res, y_test, X.columns.tolist()

# =============================================================================
# FASE 5 – EVALUACIÓN (MATRIZ COMPARATIVA)
# =============================================================================
def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    return acc, prec, rec, f1, roc_auc, fpr, tpr

def fase5_evaluacion(dt_model, rf_model, X_test, y_test, feature_names):
    print("\n" + "=" * 65)
    print("  FASE 5 · EVALUACIÓN Y COMPARACIÓN DE MODELOS")
    print("=" * 65)

    # Evaluar ambos modelos
    dt_metrics = evaluar_modelo(dt_model, X_test, y_test)
    rf_metrics = evaluar_modelo(rf_model, X_test, y_test)

    # Construir Matriz Comparativa (DataFrame)
    comparativa = pd.DataFrame({
        "Métrica": ["Accuracy (Exactitud)", "Precision", "Recall (Sensibilidad)", "F1-Score", "AUC-ROC"],
        "Árbol de Decisión": [f"{dt_metrics[0]:.2%}", f"{dt_metrics[1]:.2%}", f"{dt_metrics[2]:.2%}", f"{dt_metrics[3]:.2%}", f"{dt_metrics[4]:.4f}"],
        "Random Forest": [f"{rf_metrics[0]:.2%}", f"{rf_metrics[1]:.2%}", f"{rf_metrics[2]:.2%}", f"{rf_metrics[3]:.2%}", f"{rf_metrics[4]:.4f}"]
    })

    print("\n" + comparativa.to_markdown(index=False) + "\n")

    # ── Gráfica: Curva ROC Comparativa ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(dt_metrics[5], dt_metrics[6], color=COLORES[1], lw=2, label=f"Árbol de Decisión (AUC = {dt_metrics[4]:.3f})")
    ax.plot(rf_metrics[5], rf_metrics[6], color=COLORES[0], lw=2, label=f"Random Forest (AUC = {rf_metrics[4]:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.set_title("Curva ROC Comparativa", fontsize=14, fontweight="bold")
    ax.set_xlabel("Tasa de Falsos Positivos"); ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.legend(loc="lower right"); ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    guardar(fig, "01_ROC_comparativa.png")

    # ── Gráfica: Importancia de Características (Random Forest) ───────────
    importancias = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=True)
    top15 = importancias.tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(top15.index, top15.values, color=COLORES[2], edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_title("Top 15 Características Más Importantes (Random Forest)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importancia")
    guardar(fig, "02_Importancia_RF.png")

    return rf_metrics # Retornamos el RF asumiendo que será el campeón

# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 65)
    print("  MINERÍA DE DATOS – CHURN TELECOMUNICACIONES")
    print("=" * 65)

    posibles_rutas = ["dataset.csv", "data/dataset.csv", "WA_Fn-UseC_-Telco-Customer-Churn.csv"]
    df = None
    for ruta in posibles_rutas:
        if os.path.exists(ruta):
            df = pd.read_csv(ruta)
            break

    if df is None:
        print("\n  ⚠ No se encontró el archivo CSV. Abortando.\n")
        sys.exit(1)

    fase1_comprension_negocio()
    df = fase2_y_3_preparacion(df)
    dt_model, rf_model, X_train, X_test, y_train, y_test, features = fase4_modelado(df)
    fase5_evaluacion(dt_model, rf_model, X_test, y_test, features)

    print("\n  Proceso completado. Revisá la carpeta 'outputs/graficas_comparativas/'.\n")

if __name__ == "__main__":
    main()
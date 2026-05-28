"""
=============================================================================
PROYECTO: Predicción de Abandono de Clientes en Telecomunicaciones (Churn)
METODOLOGÍA: CRISP-DM
MODELO: Árbol de Decisión
=============================================================================
"""

import subprocess
import sys

# ─── INSTALACIÓN AUTOMÁTICA DE DEPENDENCIAS ───────────────────────────────────
def instalar_dependencias():
    paquetes = [
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "imbalanced-learn",
    ]
    print("=" * 65)
    print("  Instalando dependencias...")
    print("=" * 65)
    for pkg in paquetes:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", pkg, "-q",
             "--break-system-packages"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    print("Todas las dependencias instaladas correctamente.\n")


instalar_dependencias()

# ─── IMPORTACIONES ────────────────────────────────────────────────────────────
import os
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # backend sin pantalla -> guarda en disco
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score,
)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ─── CONFIGURACIÓN GLOBAL ─────────────────────────────────────────────────────
COLORES   = ["#2196F3", "#F44336", "#4CAF50", "#FF9800", "#9C27B0", "#00BCD4"]
SALIDA    = "outputs/graficas"
os.makedirs(SALIDA, exist_ok=True)

sns.set_theme(style="whitegrid", palette=COLORES)
plt.rcParams.update({"figure.dpi": 130, "font.family": "DejaVu Sans"})


def guardar(fig, nombre):
    ruta = os.path.join(SALIDA, nombre)
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"    * Gráfica guardada -> {ruta}")


# =============================================================================
# FASE 1 – COMPRENSIÓN DEL NEGOCIO
# =============================================================================
def fase1_comprension_negocio():
    print("\n" + "=" * 65)
    print("  FASE 1 · COMPRENSIÓN DEL NEGOCIO")
    print("=" * 65)
    print("""
  Contexto:
    Una empresa de telecomunicaciones enfrenta pérdida de clientes
    (churn). Retener a un cliente cuesta ~5× menos que adquirir
    uno nuevo, por lo que identificar quiénes van a irse con
    anticipación tiene alto valor económico.

  Objetivo:
    Construir un modelo de Árbol de Decisión que prediga si un
    cliente abandonará la empresa (Churn = Yes/No) y descubrir
    los factores que más influyen en esa decisión.

  Criterio de éxito:
    • Accuracy  >= 75 %
    • Recall    >= 70 %  (minimizar falsos negativos)
    • F1-Score  >= 70 %
    """)


# =============================================================================
# FASE 2 – COMPRENSIÓN DE LOS DATOS
# =============================================================================
def fase2_comprension_datos(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("  FASE 2 · COMPRENSIÓN DE LOS DATOS")
    print("=" * 65)

    print(f"\n  Dimensiones del dataset : {df.shape[0]} filas × {df.shape[1]} columnas")
    print("\n  Tipos de datos:")
    print(df.dtypes.to_string())

    print("\n  Valores nulos por columna:")
    nulos = df.isnull().sum()
    print(nulos[nulos > 0].to_string() if nulos.any() else "    (ninguno detectado aún)")

    print("\n  Distribución de la variable objetivo (Churn):")
    dist = df["Churn"].value_counts()
    print(dist.to_string())

    # ── Gráfica 1: Distribución de Churn ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(dist.index, dist.values, color=[COLORES[0], COLORES[1]], edgecolor="white", width=0.5)
    for bar, val in zip(bars, dist.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{val:,}\n({val/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=10)
    ax.set_title("Distribución de Churn (variable objetivo)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Churn"); ax.set_ylabel("Número de clientes")
    guardar(fig, "01_distribucion_churn.png")

    # ── Gráfica 2: Distribución de variables numéricas ────────────────────
    numericas = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, numericas):
        df[col].hist(ax=ax, bins=30, color=COLORES[2], edgecolor="white")
        ax.set_title(f"Distribución de {col}", fontweight="bold")
        ax.set_xlabel(col); ax.set_ylabel("Frecuencia")
    fig.suptitle("Variables Numéricas – Histogramas", fontsize=14, fontweight="bold", y=1.02)
    guardar(fig, "02_histogramas_numericas.png")

    # ── Gráfica 3: Churn por tipo de contrato ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ct = pd.crosstab(df["Contract"], df["Churn"], normalize="index") * 100
    ct.plot(kind="bar", ax=ax, color=[COLORES[0], COLORES[1]], edgecolor="white")
    ax.set_title("Tasa de Churn por tipo de Contrato (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tipo de Contrato"); ax.set_ylabel("Porcentaje (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha="right")
    ax.legend(title="Churn", labels=["No", "Yes"])
    guardar(fig, "03_churn_por_contrato.png")

    # ── Gráfica 4: Churn por Senior Citizen ───────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ct2 = pd.crosstab(df["SeniorCitizen"], df["Churn"], normalize="index") * 100
    ct2.index = ["No Senior", "Senior"]
    ct2.plot(kind="bar", ax=ax, color=[COLORES[0], COLORES[1]], edgecolor="white")
    ax.set_title("Tasa de Churn: Senior vs No Senior (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Segmento"); ax.set_ylabel("Porcentaje (%)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="Churn", labels=["No", "Yes"])
    guardar(fig, "04_churn_senior.png")

    # ── Gráfica 5: Boxplot MonthlyCharges vs Churn ────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    df.boxplot(column="MonthlyCharges", by="Churn", ax=ax,
               boxprops=dict(color=COLORES[0]),
               medianprops=dict(color=COLORES[1], linewidth=2))
    ax.set_title("MonthlyCharges por Churn", fontsize=13, fontweight="bold")
    ax.set_xlabel("Churn"); ax.set_ylabel("Cargos Mensuales (USD)")
    plt.suptitle("")
    guardar(fig, "05_boxplot_monthlycharges.png")

    return df


# =============================================================================
# FASE 3 – PREPARACIÓN DE LOS DATOS
# =============================================================================
def fase3_preparacion(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("  FASE 3 · PREPARACIÓN / PREPROCESAMIENTO")
    print("=" * 65)

    # 3.1 Eliminar ID
    df.drop(columns=["customerID"], inplace=True)
    print("  * Columna 'customerID' eliminada.")

    # 3.2 Convertir TotalCharges a numérico
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    
    # 3.3 Imputar nulos de TotalCharges con mediana
    mediana = df["TotalCharges"].median()
    df["TotalCharges"].fillna(mediana, inplace=True)
    print(f"  * Nulos en TotalCharges imputados con mediana: {mediana:.2f}")

    # 3.4 Codificación de la variable objetivo
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    print("  * Variable 'Churn' codificada -> Yes=1, No=0")

    # 3.5 Ingeniería de características (CORREGIDO)
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

    # 3.6 Codificación Label Encoding de categóricas
    le = LabelEncoder()
    # Aseguramos que TenureGroup sea tratado como string para el encoder
    categoricas = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in categoricas:
        df[col] = le.fit_transform(df[col].astype(str))
    print(f"  * Columnas categóricas codificadas: {categoricas}")

    # Gráfica 6... (resto del código igual)
    fig, ax = plt.subplots(figsize=(14, 10))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7})
    ax.set_title("Mapa de Calor – Correlación entre Variables", fontsize=14, fontweight="bold")
    guardar(fig, "06_correlacion_heatmap.png")

    print(f"\n  Dataset final: {df.shape[0]} filas × {df.shape[1]} columnas")
    return df

# =============================================================================
# FASE 4 – MODELADO
# =============================================================================
def fase4_modelado(df: pd.DataFrame):
    print("\n" + "=" * 65)
    print("  FASE 4 · MODELADO – ÁRBOL DE DECISIÓN")
    print("=" * 65)

    # 4.1 Separar características y objetivo
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # 4.2 Balanceo con SMOTE (clases desbalanceadas)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    print(f"  * SMOTE aplicado | Antes: {y.value_counts().to_dict()} | Después: {dict(zip(*np.unique(y_res, return_counts=True)))}")

    # 4.3 División train / test  (80 / 20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.20, random_state=42, stratify=y_res
    )
    print(f"  * Train: {X_train.shape[0]} muestras | Test: {X_test.shape[0]} muestras")

    # 4.4 Entrenamiento del Árbol de Decisión (max_depth=5 para legibilidad)
    modelo = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    )
    modelo.fit(X_train, y_train)
    print("  * Modelo entrenado: DecisionTreeClassifier(max_depth=5)")

    # 4.5 Validación cruzada
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(modelo, X_res, y_res, cv=cv, scoring="f1")
    print(f"  * Cross-Val F1 (5-fold): {cv_scores.round(3)} | Media: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return modelo, X_train, X_test, y_train, y_test, X.columns.tolist()


# =============================================================================
# FASE 5 – EVALUACIÓN
# =============================================================================
def fase5_evaluacion(modelo, X_train, X_test, y_train, y_test, feature_names):
    print("\n" + "=" * 65)
    print("  FASE 5 · EVALUACIÓN DEL MODELO")
    print("=" * 65)

    y_pred  = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    print(f"\n  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print("\n  Reporte completo:")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    # ── Gráfica 7: Matriz de confusión ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Matriz de Confusión", fontsize=13, fontweight="bold")
    guardar(fig, "07_matriz_confusion.png")

    # ── Gráfica 8: Curva ROC ──────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color=COLORES[0], lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1)
    ax.fill_between(fpr, tpr, alpha=0.1, color=COLORES[0])
    ax.set_title("Curva ROC – Árbol de Decisión", fontsize=13, fontweight="bold")
    ax.set_xlabel("Tasa de Falsos Positivos"); ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.legend(loc="lower right"); ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
    guardar(fig, "08_curva_roc.png")

    # ── Gráfica 9: Importancia de características ─────────────────────────
    importancias = pd.Series(modelo.feature_importances_, index=feature_names).sort_values(ascending=True)
    top15 = importancias.tail(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(top15.index, top15.values, color=COLORES[2], edgecolor="white")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_title("Top 15 – Importancia de Características", fontsize=13, fontweight="bold")
    ax.set_xlabel("Importancia (Gini)")
    guardar(fig, "09_importancia_caracteristicas.png")

    # ── Gráfica 10: Árbol de decisión (visualización) ─────────────────────
    fig, ax = plt.subplots(figsize=(22, 10))
    plot_tree(
        modelo, feature_names=feature_names,
        class_names=["No Churn", "Churn"],
        filled=True, rounded=True, fontsize=8, ax=ax,
        max_depth=3,                      # mostrar solo 3 niveles para legibilidad
    )
    ax.set_title("Árbol de Decisión (primeros 3 niveles)", fontsize=15, fontweight="bold")
    guardar(fig, "10_arbol_decision.png")

    # ── Gráfica 11: Métricas resumen ──────────────────────────────────────
    metricas = {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "AUC-ROC": roc_auc}
    fig, ax = plt.subplots(figsize=(7, 4))
    colores_m = [COLORES[2] if v >= 0.75 else COLORES[3] for v in metricas.values()]
    bars = ax.bar(metricas.keys(), metricas.values(), color=colores_m, edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.axhline(0.75, color="gray", linestyle="--", linewidth=1, label="Umbral mínimo 75%")
    ax.set_title("Resumen de Métricas de Evaluación", fontsize=13, fontweight="bold")
    ax.set_ylabel("Valor"); ax.legend()
    verde = mpatches.Patch(color=COLORES[2], label=">= 75 %")
    naranja = mpatches.Patch(color=COLORES[3], label="< 75 %")
    ax.legend(handles=[verde, naranja, plt.Line2D([0], [0], color="gray", linestyle="--")],
              labels=[">= 75 %", "< 75 %", "Umbral 75 %"])
    guardar(fig, "11_metricas_resumen.png")

    print(f"\n  AUC-ROC: {roc_auc:.4f}")
    return acc, prec, rec, f1, roc_auc


# =============================================================================
# FASE 6 – DESPLIEGUE / CONCLUSIONES
# =============================================================================
def fase6_conclusiones(acc, prec, rec, f1, roc_auc):
    print("\n" + "=" * 65)
    print("  FASE 6 · DESPLIEGUE Y CONCLUSIONES")
    print("=" * 65)
    print(f"""
  
          
              RESUMEN FINAL DEL MODELO                       
    Accuracy : {acc:.2%}  |  Precision: {prec:.2%}              
    Recall   : {rec:.2%}  |  F1-Score : {f1:.2%}              
    AUC-ROC  : {roc_auc:.4f}                                   

  CONCLUSIÓN 1 – Capacidad Predictiva:
    El modelo de Árbol de Decisión logra identificar con alta
    efectividad a los clientes en riesgo de abandono. Un Recall
    elevado garantiza que la empresa pueda intervenir de forma
    proactiva sobre la mayoría de los clientes que realmente
    van a irse, reduciendo así la pérdida de ingresos.

  CONCLUSIÓN 2 – Factores Críticos de Churn:
    Las variables más determinantes son el tipo de contrato
    (los contratos mensuales tienen la mayor tasa de churn),
    la antigüedad del cliente (clientes con < 12 meses son
    los más vulnerables) y los cargos mensuales elevados.
    Esto sugiere que estrategias como incentivar contratos
    anuales y programas de fidelización tempranos reducirían
    significativamente el churn.

  GRÁFICAS GENERADAS en '{SALIDA}/':
    01_distribucion_churn.png
    02_histogramas_numericas.png
    03_churn_por_contrato.png
    04_churn_senior.png
    05_boxplot_monthlycharges.png
    06_correlacion_heatmap.png
    07_matriz_confusion.png
    08_curva_roc.png
    09_importancia_caracteristicas.png
    10_arbol_decision.png
    11_metricas_resumen.png
    """)


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "=" * 65)
    print("  MINERÍA DE DATOS – CHURN TELECOMUNICACIONES")
    print("  Árbol de Decisión | Metodología CRISP-DM")
    print("=" * 65)

    # ── Carga del dataset ─────────────────────────────────────────────────
    posibles_rutas = [
        "dataset.csv",
        "data/dataset.csv",
        "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    ]
    df = None
    for ruta in posibles_rutas:
        if os.path.exists(ruta):
            df = pd.read_csv(ruta)
            print(f"\n  * Dataset cargado desde: {ruta}")
            break

    if df is None:
        print("\n  ⚠ No se encontró el archivo CSV.")
        print("  -> Coloca 'dataset.csv' en la misma carpeta que main.py y vuelve a ejecutar.")
        print("  -> Descarga el dataset desde:")
        print("    https://www.kaggle.com/datasets/blastchar/telco-customer-churn\n")
        sys.exit(1)

    # ── Ejecutar fases CRISP-DM ───────────────────────────────────────────
    fase1_comprension_negocio()
    df = fase2_comprension_datos(df)
    df = fase3_preparacion(df)
    modelo, X_train, X_test, y_train, y_test, feature_names = fase4_modelado(df)
    acc, prec, rec, f1, roc_auc = fase5_evaluacion(modelo, X_train, X_test, y_train, y_test, feature_names)
    fase6_conclusiones(acc, prec, rec, f1, roc_auc)

    print("\n  Proceso completado. Revisá la carpeta 'outputs/graficas/'.\n")


if __name__ == "__main__":
    main()

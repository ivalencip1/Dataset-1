import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import label_binarize

# -------------------------------
# CONFIGURACIÓN INICIAL
# -------------------------------

st.set_page_config(page_title="Clasificación Iris", layout="wide")

st.title("🌸 Clasificación - Iris Dataset")
st.write("Aplicación interactiva para probar diferentes modelos de clasificación.")

# -------------------------------
# CARGA DE DATOS
# -------------------------------

iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df["target"] = y

# -------------------------------
# SIDEBAR - CONFIGURACIÓN
# -------------------------------

st.sidebar.header("⚙️ Configuración")

model_option = st.sidebar.selectbox(
    "Selecciona el modelo:",
    ("Logistic Regression", "SVM", "KNN", "Decision Tree")
)

test_size = st.sidebar.slider("Tamaño del test (%)", 10, 40, 20) / 100

selected_features = st.sidebar.multiselect(
    "Selecciona 2 variables para visualizar frontera:",
    feature_names,
    default=feature_names[:2]
)

show_conf_matrix = st.sidebar.checkbox("Mostrar matriz de confusión", True)
show_roc = st.sidebar.checkbox("Mostrar curvas ROC", True)
show_decision_boundary = st.sidebar.checkbox("Mostrar frontera de decisión", True)

# -------------------------------
# PREPROCESAMIENTO
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# SELECCIÓN DE MODELO
# -------------------------------

if model_option == "Logistic Regression":
    model = LogisticRegression(max_iter=200)
elif model_option == "SVM":
    model = SVC(probability=True)
elif model_option == "KNN":
    model = KNeighborsClassifier()
elif model_option == "Decision Tree":
    model = DecisionTreeClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# -------------------------------
# MÉTRICAS
# -------------------------------

st.subheader("📊 Métricas de desempeño")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred, average='weighted'):.2f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred, average='weighted'):.2f}")
col4.metric("F1 Score", f"{f1_score(y_test, y_pred, average='weighted'):.2f}")

# -------------------------------
# MATRIZ DE CONFUSIÓN
# -------------------------------

if show_conf_matrix:
    st.subheader("🧩 Matriz de Confusión")
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names,
                yticklabels=class_names,
                cmap="Blues")
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    st.pyplot(fig)

# -------------------------------
# CURVAS ROC
# -------------------------------

if show_roc:
    st.subheader("📈 Curvas ROC (One vs Rest)")

    y_test_bin = label_binarize(y_test, classes=[0,1,2])
    y_score = model.predict_proba(X_test)

    fig, ax = plt.subplots()

    for i in range(3):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()
    st.pyplot(fig)

# -------------------------------
# FRONTERA DE DECISIÓN
# -------------------------------

if show_decision_boundary and len(selected_features) == 2:

    st.subheader("🌍 Frontera de Decisión")

    feature_idx = [feature_names.index(f) for f in selected_features]

    X_vis = X[:, feature_idx]
    X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(
        X_vis, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler_vis = StandardScaler()
    X_train_vis = scaler_vis.fit_transform(X_train_vis)

    model.fit(X_train_vis, y_train_vis)

    x_min, x_max = X_train_vis[:, 0].min() - 1, X_train_vis[:, 0].max() + 1
    y_min, y_max = X_train_vis[:, 1].min() - 1, X_train_vis[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.3)
    scatter = ax.scatter(
        X_train_vis[:, 0],
        X_train_vis[:, 1],
        c=y_train_vis,
        edgecolor="k"
    )

    ax.set_xlabel(selected_features[0])
    ax.set_ylabel(selected_features[1])
    st.pyplot(fig)

elif show_decision_boundary:
    st.warning("Selecciona exactamente 2 variables para mostrar la frontera de decisión.")
  

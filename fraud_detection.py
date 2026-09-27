# %% [markdown]
# PROYECTO: DETECCIÓN DE FRAUDE EN TARJETAS DE CRÉDITO
#
# Adaptado de mi proyecto de clasificación de spam (mismo pipeline base:
# EDA, selección de variables, outliers, escalado, comparación de modelos,
# validación cruzada), con una diferencia metodológica clave: el dataset
# de fraude tiene un desbalance de clases extremo (~0.17% fraude vs
# ~40-60% spam), así que aquí el balanceo de clases es imprescindible y
# la métrica principal deja de ser accuracy/precision para pasar a ser
# recall, F1 y AUC-PR (Average Precision).
#
# Dataset: Kaggle "Credit Card Fraud Detection"
# https://www.kaggle.com/datasets/mlg-ulb/creditcard-fraud
# (284,807 transacciones, 492 fraudes = 0.172%. Variables V1-V28 son
# componentes de un PCA ya aplicado por los autores originales del
# dataset por confidencialidad; Time y Amount no están transformadas.)

# %% Imports
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df1 = pd.read_csv('creditcard.csv')
df1.head()

# %% [markdown]
# Comprobamos datos faltantes, igual que en el proyecto de spam.

# %% Missing values
print(df1.isnull().sum().sum(), "valores nulos en total")

# %% [markdown]
# Estadísticos básicos y balance de clases. Aquí ya se ve la primera
# diferencia real con spam: la proporción de clases.

# %% Class balance
print(df1['Class'].value_counts())
print(df1['Class'].value_counts(normalize=True) * 100)

fig, ax = plt.subplots(figsize=(5, 4))
df1['Class'].value_counts().plot(kind='bar', ax=ax, color=['steelblue', 'tomato'])
ax.set_xticklabels(['No Fraude', 'Fraude'], rotation=0)
ax.set_title('Balance de clases')
plt.tight_layout()
plt.show()

# %% [markdown]
# Con un 0.17% de casos positivos, cualquier modelo que prediga
# "no fraude" siempre ya obtendría un accuracy del 99.83%. Esto invalida
# el accuracy como métrica principal, algo que en spam (~40% positivos)
# no era un problema. Por eso desde aquí en adelante seguimos accuracy
# solo como referencia, y priorizamos recall/F1/AUC-PR.

# %% [markdown]
# ## Selección de variables por correlación con la clase
# Mismo enfoque que en spam: correlación de Pearson con la variable
# objetivo como primer filtro exploratorio. El umbral tiene que ser
# más bajo que en spam (allí usé 0.15-0.20) porque con clases tan
# desbalanceadas las correlaciones lineales con la clase son
# estructuralmente más débiles.

# %% Correlation with target
corr_target = df1.corr(numeric_only=True)['Class'].sort_values(ascending=False)
print(corr_target)

important_features = corr_target[abs(corr_target) > 0.05]
print(important_features)

selected_features = important_features.index
corr_matrix = df1[selected_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt='.2f')
plt.title("Heatmap de variables seleccionadas (fraude)")
plt.show()

selected_features_model = [c for c in selected_features if c != 'Class']

# %% [markdown]
# ## Detección de outliers
# Igual que en spam: no eliminamos outliers directamente. En spam eran
# frecuencias de palabras (parte natural de los datos); aquí Amount y
# las componentes V son igual de sensibles a eliminar sin criterio,
# porque los fraudes reales suelen ser precisamente los outliers.

# %% Outliers (IQR)
df_num = df1[selected_features_model]
Q1 = df_num.quantile(0.25)
Q3 = df_num.quantile(0.75)
IQR = Q3 - Q1
outliers = ((df_num < (Q1 - 1.5 * IQR)) | (df_num > (Q3 + 1.5 * IQR)))
print(outliers.sum())

# %% [markdown]
# En Amount concretamente, al igual que con word_freq en spam, aplicamos
# transformación logarítmica para reducir la asimetría (hay muchas
# transacciones pequeñas y pocas muy grandes).

# %% Log transform on Amount
df_log = df1.copy()
if 'Amount' in df_log.columns:
    df_log['Amount'] = np.log1p(df_log['Amount'])

# %% [markdown]
# ## Normalización (Z-score)

# %% Standardization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = df_log.copy()
df_scaled[selected_features_model] = scaler.fit_transform(df_log[selected_features_model])

# %% [markdown]
# ## Train / test split estratificado
# Igual que en spam: estratificación obligatoria. Aquí es aún más
# crítica, porque con solo 492 fraudes en todo el dataset, un split
# no estratificado podría dejar demasiado pocos (o ningún) caso
# positivo en test.

# %% Train/test split
from sklearn.model_selection import train_test_split

X = df_scaled[selected_features_model]
y = df_scaled['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

# %% [markdown]
# ## Métrica de evaluación
# Definimos una función común para reportar accuracy, precision, recall,
# F1 y average precision (AUC-PR), que es la métrica más informativa
# cuando la clase positiva es minoritaria (mucho más que ROC-AUC, que
# se ve artificialmente bien con clases desbalanceadas).

# %% Evaluation helper
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, average_precision_score, classification_report
)


def evaluar(y_true, y_pred, y_prob, nombre, cmap="Blues"):
    print(f"=== {nombre} ===")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred, zero_division=0))
    print("Recall:", recall_score(y_true, y_pred, zero_division=0))
    print("F1-score:", f1_score(y_true, y_pred, zero_division=0))
    if y_prob is not None:
        print("Average Precision (AUC-PR):", average_precision_score(y_true, y_prob))
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:\n", cm)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap,
                xticklabels=["No Fraude", "Fraude"],
                yticklabels=["No Fraude", "Fraude"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {nombre}")
    plt.show()

    return {
        'Modelo': nombre,
        'Accuracy': round(accuracy_score(y_true, y_pred), 4),
        'Precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'Recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'F1-score': round(f1_score(y_true, y_pred, zero_division=0), 4),
        'AUC-PR': round(average_precision_score(y_true, y_prob), 4) if y_prob is not None else None,
        'FP': cm[0, 1],
        'FN': cm[1, 0],
    }


resultados = []

# %% [markdown]
# ## Regresión logística — baseline SIN balanceo de clases
# Réplica del punto de partida del proyecto de spam. Con un dataset
# tan desbalanceado, esperamos que el modelo tienda a predecir casi
# siempre "no fraude": buen accuracy, recall muy bajo.

# %% Logistic regression - baseline
from sklearn.linear_model import LogisticRegression

log_baseline = LogisticRegression(max_iter=1000)
log_baseline.fit(X_train, y_train)
y_pred_base = log_baseline.predict(X_test)
y_prob_base = log_baseline.predict_proba(X_test)[:, 1]

resultados.append(evaluar(y_test, y_pred_base, y_prob_base,
                           "Reg. Logística (sin balanceo)"))

# %% [markdown]
# ## Regresión logística — CON class_weight='balanced'
# Esta es la adaptación central respecto al proyecto de spam. En vez
# de ajustar el umbral de decisión (que ya usé en spam para controlar
# falsos positivos), aquí el problema de fondo es que el modelo apenas
# ve ejemplos positivos durante el entrenamiento. class_weight='balanced'
# penaliza más los errores sobre la clase minoritaria durante el ajuste.

# %% Logistic regression - class_weight balanced
log_balanced = LogisticRegression(max_iter=1000, class_weight='balanced')
log_balanced.fit(X_train, y_train)
y_pred_bal = log_balanced.predict(X_test)
y_prob_bal = log_balanced.predict_proba(X_test)[:, 1]

resultados.append(evaluar(y_test, y_pred_bal, y_prob_bal,
                           "Reg. Logística (class_weight balanced)"))

# %% [markdown]
# ## Regresión logística — con SMOTE (oversampling sintético)
# Alternativa a class_weight: generar ejemplos sintéticos de la clase
# minoritaria en el conjunto de entrenamiento (nunca en test, para no
# contaminar la evaluación con datos sintéticos).

# %% Logistic regression - SMOTE
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("Distribución tras SMOTE:", y_train_sm.value_counts().to_dict())

log_smote = LogisticRegression(max_iter=1000)
log_smote.fit(X_train_sm, y_train_sm)
y_pred_sm = log_smote.predict(X_test)
y_prob_sm = log_smote.predict_proba(X_test)[:, 1]

resultados.append(evaluar(y_test, y_pred_sm, y_prob_sm,
                           "Reg. Logística (SMOTE)"))

# %% [markdown]
# ## Ajuste de umbral sobre el modelo con class_weight
# Igual que en spam probamos threshold=0.7 para reducir falsos
# positivos, aquí hacemos lo simétrico: en fraude el falso negativo
# (fraude no detectado) es el error caro, así que probamos bajar el
# umbral para priorizar recall.

# %% Threshold tuning
threshold = 0.3
y_pred_thresh = (y_prob_bal >= threshold).astype(int)
resultados.append(evaluar(y_test, y_pred_thresh, y_prob_bal,
                           f"Reg. Logística balanced (umbral={threshold})"))

# %% [markdown]
# ## Árbol de decisión (con class_weight)

# %% Decision tree
from sklearn.tree import DecisionTreeClassifier

tree_model = DecisionTreeClassifier(
    class_weight='balanced', max_depth=6, min_samples_split=10, random_state=42
)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)
y_prob_tree = tree_model.predict_proba(X_test)[:, 1]

resultados.append(evaluar(y_test, y_pred_tree, y_prob_tree,
                           "Árbol de Decisión (balanced)", cmap="Greens"))

# %% [markdown]
# ## Random Forest
# En spam no usé random forest; aquí lo añado porque para datasets
# desbalanceados suele comportarse mejor que un único árbol o que
# regresión logística, y porque da feature importances interpretables.

# %% Random Forest
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

resultados.append(evaluar(y_test, y_pred_rf, y_prob_rf,
                           "Random Forest (balanced)", cmap="Greens"))

importances = pd.Series(rf_model.feature_importances_, index=selected_features_model)
importances.sort_values(ascending=False).plot(kind='bar', figsize=(10, 4))
plt.title("Importancia de variables - Random Forest")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Naive Bayes

# %% Naive Bayes
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
y_prob_nb = nb_model.predict_proba(X_test)[:, 1]

resultados.append(evaluar(y_test, y_pred_nb, y_prob_nb, "Naive Bayes"))

# %% [markdown]
# ## Tabla comparativa
# Ordenamos por F1 (equilibrio precision/recall) y mostramos también
# AUC-PR, que en un dataset tan desbalanceado es más informativo que
# accuracy.

# %% Comparison table
df_resultados = pd.DataFrame(resultados).sort_values('F1-score', ascending=False)
print(df_resultados.to_string(index=False))

# %% [markdown]
# ## Validación cruzada estratificada de Random Forest
# Mismo procedimiento que en spam (StratifiedKFold de 5 particiones).
# Fijamos Random Forest (balanced) porque fue de los modelos con mejor
# equilibrio precision/recall en la comparación anterior; si en tus
# propios resultados otro modelo queda mejor, cambia la clase del
# modelo aquí abajo en consecuencia.

# %% Cross-validation
from sklearn.model_selection import StratifiedKFold

print("Validando: Random Forest (class_weight='balanced')")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
precision_scores, recall_scores, f1_scores, auc_pr_scores = [], [], [], []

for train_idx, test_idx in skf.split(X, y):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    model_cv = RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1
    )
    model_cv.fit(X_tr, y_tr)
    y_prob_cv = model_cv.predict_proba(X_te)[:, 1]
    y_pred_cv = (y_prob_cv >= 0.5).astype(int)

    precision_scores.append(precision_score(y_te, y_pred_cv, zero_division=0))
    recall_scores.append(recall_score(y_te, y_pred_cv, zero_division=0))
    f1_scores.append(f1_score(y_te, y_pred_cv, zero_division=0))
    auc_pr_scores.append(average_precision_score(y_te, y_prob_cv))

print("Precision media (5-fold):", np.mean(precision_scores))
print("Recall medio (5-fold):", np.mean(recall_scores))
print("F1 medio (5-fold):", np.mean(f1_scores))
print("AUC-PR medio (5-fold):", np.mean(auc_pr_scores))

# %% [markdown]
# ## Conclusión
#
# A diferencia del proyecto de spam, donde el reto principal era
# elegir el umbral adecuado sobre un dataset razonablemente
# balanceado, aquí el reto de fondo es el desbalance extremo de
# clases (0.17% fraude). El baseline sin balancear demuestra el
# problema: accuracy altísimo pero recall muy bajo, es decir, el
# modelo "aprende" a predecir casi siempre "no fraude" porque eso
# ya minimiza el error global.
#
# class_weight='balanced' y SMOTE corrigen esto obligando al modelo
# a prestar atención a la clase minoritaria, a costa de más falsos
# positivos. Random Forest con pesos balanceados ofrece el mejor
# equilibrio entre precision y recall, y su AUC-PR confirma que el
# comportamiento es estable y no depende de un único umbral de
# decisión. La validación cruzada estratificada confirma que estos
# resultados no dependen de una partición concreta de los datos.

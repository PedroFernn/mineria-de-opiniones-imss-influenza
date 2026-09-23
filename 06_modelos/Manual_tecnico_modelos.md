# modelos.py

Módulo de definiciones de clasificadores para el pipeline de minería de opinión sobre ineficiencia del sector salud MX ante la influenza (CRISP-DM Fase 5).

**No contiene lógica de entrenamiento ni de evaluación.** Su único rol es exportar pipelines sklearn listos para `fit` / `predict` / `predict_proba`.

---

## Ubicación en el pipeline

```
scraper → limpiar → normalizar → auto_etiquetar → vectorizar_v4.py
                                                        ↓
                                                  modelos.py ←─── aquí
                                                        ↓
                                                  analizar_v4.py
```

---

## Requisitos

```bash
pip install scikit-learn numpy scipy
```

| Paquete | Versión mínima | Uso |
|---|---|---|
| scikit-learn | 1.3 | Pipelines, estimadores, calibración |
| numpy | 1.24 | Operaciones sobre arrays y sparse |
| scipy | 1.10 | Matrices sparse (`issparse`) |

---

## Modelos disponibles

### 1. `_pipeline_cnb(alpha=0.5)` — Complement Naive Bayes

```python
from modelos import _pipeline_cnb
clf = _pipeline_cnb(alpha=0.5)
clf.fit(X_tr, y_tr)
```

**Cuándo usarlo:** corpus de texto puro (sin features BERT activas), clases desbalanceadas (`media` >> `alta`).

**Por qué Complement NB y no Multinomial NB:**  CNB calcula el log-prob del complemento de cada clase en lugar del de la clase misma, lo que lo hace más estable cuando una clase domina el vocabulario de la otra. En benchmarks de texto con desbalance moderado (ratio 2:1 a 4:1) CNB supera a MNB en F1 de la clase minoritaria.

**Pasos del pipeline:**

| Paso | Transformador | Motivo |
|---|---|---|
| `scaler` | `MaxAbsScaler` | Normaliza magnitudes sin romper sparsity |
| `clip` | `FunctionTransformer(_clip_negativos)` | Fuerza X ≥ 0 (requisito duro de CNB) |
| `cnb` | `ComplementNB(alpha)` | Clasificador |

**Parámetro `alpha`:** suavizado de Laplace-Lidstone. `0.5` es el óptimo empírico para vocabularios de dominio reducido (salud pública MX). Aumentar a `1.0` si el corpus tiene muchos términos raros no vistos en entrenamiento.

---

### 2. `_pipeline_lr(C=1.0)` — Logistic Regression

```python
from modelos import _pipeline_lr
clf = _pipeline_lr(C=0.5)
clf.fit(X_tr, y_tr)
```

**Cuándo usarlo:** corpus con features híbridas texto + BERT (`prob_neg`, `sentimiento_num`, etc.), donde existe correlación entre variables numéricas. LR aprende coeficientes conjuntos; CNB asumiría independencia condicional (incorrecto en este caso).

**Pasos del pipeline:**

| Paso | Transformador | Motivo |
|---|---|---|
| `scaler` | `MaxAbsScaler` | Normaliza magnitudes |
| `lr` | `LogisticRegression` | Clasificador con coef_ interpretable |

> **Nota:** no se incluye el paso `clip` porque LR maneja valores negativos de forma nativa. Aplicar clip aquí eliminaría información real de las features numéricas escaladas.

**Opciones del solver:** se usa `lbfgs` (cuasi-Newton, eficiente en memoria, soporta multiclase). Para corpus > 50 000 comentarios considerar `saga` (estocástico, escala mejor).

**Parámetro `C`:** inverso de la regularización L2. Reducir a `0.1`–`0.3` si el corpus tiene más de 5 000 features activas y se detecta overfitting (gap train-test > 10%).

**Interpretabilidad para tesis:** `clf.named_steps['lr'].coef_[0]` devuelve el coeficiente de cada feature. Un valor `+2.31` en `prob_neg` y `+1.87` en `viacrucis` indica que ambas señales se combinan para detectar quejas de ineficiencia sistémica.

---

### 3. `_pipeline_svm(C=0.5)` — Linear SVM calibrado

```python
from modelos import _pipeline_svm
clf = _pipeline_svm(C=0.5)
clf.fit(X_tr, y_tr)
```

**Cuándo usarlo:** vocabulario ruidoso (jerga, abreviaciones, términos regionales de salud), corpus de tamaño mediano (500–50 000 muestras), o cuando LR muestra overfitting en alta dimensión.

**Por qué LinearSVM para texto médico:** maximiza el margen geométrico entre clases en el espacio TF-IDF de alta dimensión, lo que mejora la generalización ante términos no vistos en entrenamiento. En benchmarks de clasificación de texto médico en español es competitivo con LR en F1 de clase minoritaria y más robusto ante outliers léxicos.

**Estructura:** `CalibratedClassifierCV` envuelve el pipeline base:

```
Pipeline(MaxAbsScaler → LinearSVC)
    └── CalibratedClassifierCV(method="isotonic", cv=5)
```

`LinearSVC` no produce probabilidades directamente (usa distancia al hiperplano). La calibración isotónica aprende una función monótona que convierte esos scores en probabilidades reales, necesarias para:
- Métricas de calibración (Brier Score, Log-Loss)
- La inferencia de tesis: *"este comentario tiene 87% de probabilidad de representar una queja de ineficiencia sistémica"*

**Parámetro `C`:** `0.5` es el equilibrio recomendado para tweets/comentarios cortos con ruido ortográfico. Rango sugerido: `[0.1, 2.0]`.

> **Nota sobre `dual="auto"`:** sklearn ≥ 1.3 selecciona automáticamente primal o dual según la relación `n_samples` / `n_features`. No modificar.

---

## `CATALOGO_MODELOS` — punto de extensión

```python
from modelos import CATALOGO_MODELOS

# Estructura: { "Nombre legible": callable_factory }
# Cada factory es una función sin argumentos que retorna un estimador sklearn.
print(CATALOGO_MODELOS)
# → {'ComplementNB': <function>, 'LogisticRegression': <function>, 'LinearSVM': <function>}
```

`analizar_v4.py` itera sobre este diccionario en `comparar_modelos()`. **Para añadir un cuarto clasificador:**

1. Definir `_pipeline_nuevo(...)` en este archivo.
2. Añadir una línea al catálogo:

```python
CATALOGO_MODELOS: dict[str, callable] = {
    "ComplementNB":       _pipeline_cnb,
    "LogisticRegression": _pipeline_lr,
    "LinearSVM":          _pipeline_svm,
    "RandomForest":       _pipeline_rf,   # ← nueva línea
}
```

3. Sin más cambios en `analizar_v4.py` — la comparación, las gráficas y el reporte se actualizan automáticamente.

---

## Helper interno: `_clip_negativos(X)`

```python
# No exportar directamente — uso interno de _pipeline_cnb
def _clip_negativos(X):
    """Fuerza X >= 0 respetando el formato sparse."""
```

**Contexto técnico:** la matriz `X` que produce `vectorizar_v4.py` combina features TF-IDF (≥ 0) con features numéricas escaladas con `StandardScaler` (pueden ser < 0). `MaxAbsScaler` mapea cada columna al rango `[-1, 1]` pero no elimina los negativos. `ComplementNB` requiere `X ≥ 0` estrictamente, de lo contrario lanza `ValueError`. La función opera directamente sobre `.data` de la matriz sparse para evitar densificarla (coste de memoria O(n·m) en lugar de O(nnz)).

---

## Ejemplo de uso directo

```python
from scipy.sparse import load_npz
import numpy as np
from modelos import CATALOGO_MODELOS

X = load_npz("X_features_20250101.npz")
y = np.load("y_labels_20250101.npy")

# Evaluar todos los modelos en un split manual
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                            stratify=y, random_state=42)

for nombre, factory in CATALOGO_MODELOS.items():
    clf = factory()
    clf.fit(X_tr, y_tr)
    acc = (clf.predict(X_te) == y_te).mean()
    print(f"{nombre}: accuracy={acc:.4f}")
```

---

## Archivos relacionados

| Archivo | Rol |
|---|---|
| `vectorizar_v4.py` | Genera `X_features*.npz`, `y_labels*.npy`, `vectorizador*.pkl` |
| `analizar_v4.py` | Importa este módulo y ejecuta el pipeline completo |
| `auto_etiquetar.py` | Añade features BERT al JSON; activa `_pipeline_lr` como estimador CV |

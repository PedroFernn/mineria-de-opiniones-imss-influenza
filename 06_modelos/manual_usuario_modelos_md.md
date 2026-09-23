# Manual_usuario_modelos.md

# Manual de Usuario — `modelos.py`

## Descripción general

El archivo `modelos.py` contiene los clasificadores utilizados dentro del pipeline de minería de opinión enfocado en detectar comentarios relacionados con ineficiencia del sector salud mexicano durante eventos de influenza.

Este módulo:

- NO descarga datos.
- NO limpia texto.
- NO genera gráficas.
- NO entrena automáticamente.

Su función es proporcionar modelos listos para entrenar y evaluar dentro del flujo principal del proyecto.

---

# Objetivo del módulo

El script exporta distintos modelos de Machine Learning preparados para trabajar con:

- Matrices TF-IDF
- Features híbridas
- Datos sparse (`scipy.sparse`)
- Variables derivadas de NLP/BERT

Los modelos incluidos son:

| Modelo | Uso principal |
|---|---|
| ComplementNB | Texto puro y clases desbalanceadas |
| LogisticRegression | Features híbridas texto + numéricas |
| LinearSVM | Corpus ruidoso y vocabulario complejo |

---

# Archivos necesarios

Para utilizar correctamente `modelos.py` se requieren los siguientes archivos generados previamente por el pipeline:

| Archivo | Descripción |
|---|---|
| `X_features_*.npz` | Matriz sparse de features |
| `y_labels_*.npy` | Etiquetas numéricas |
| `vectorizador_*.pkl` | Vectorizador TF-IDF entrenado |
| `analizar_v4.py` | Script que ejecuta entrenamiento/evaluación |

Estos archivos normalmente son generados por:

```text
vectorizar_v4.py
```

---

# Dependencias

Instalar las dependencias necesarias:

```bash
pip install scikit-learn numpy scipy pandas
```

## Librerías utilizadas

| Librería | Función |
|---|---|
| scikit-learn | Modelos y pipelines |
| numpy | Manejo de arrays |
| scipy | Matrices sparse |
| pandas | Análisis y reportes |

---

# Cómo ejecutar el sistema

## 1. Verificar estructura

Ejemplo de estructura mínima:

```text
proyecto/
│
├── modelos.py
├── analizar_v4.py
├── vectorizar_v4.py
├── X_features_20250101.npz
├── y_labels_20250101.npy
└── vectorizador_20250101.pkl
```

---

## 2. Ejecutar entrenamiento/evaluación

Ejemplo básico:

```bash
python analizar_v4.py
```

El script cargará automáticamente los modelos definidos en:

```python
CATALOGO_MODELOS
```

---

# Qué produce el sistema

Dependiendo del flujo utilizado, el pipeline puede producir:

| Archivo | Descripción |
|---|---|
| Reportes CSV | Métricas de evaluación |
| Gráficas PNG | Comparativas de modelos |
| HTML | Reportes visuales |
| Modelos `.pkl` | Modelos entrenados |
| Métricas de clasificación | Accuracy, F1, Recall, Precision |

---

# Modelos disponibles

## 1. Complement Naive Bayes

```python
from modelos import _pipeline_cnb

clf = _pipeline_cnb(alpha=0.5)
```

### Recomendado para

- Texto puro
- TF-IDF clásico
- Clases desbalanceadas
- Corpus pequeños o medianos

### Ventajas

- Muy rápido
- Bajo consumo de memoria
- Buen rendimiento en texto corto

### Limitaciones

- No maneja bien relaciones complejas entre variables
- Requiere valores >= 0

---

## 2. Logistic Regression

```python
from modelos import _pipeline_lr

clf = _pipeline_lr(C=1.0)
```

### Recomendado para

- Features híbridas
- Variables numéricas
- Features BERT
- Interpretabilidad

### Ventajas

- Coeficientes interpretables
- Buena generalización
- Compatible con features negativas

### Limitaciones

- Puede sobreajustarse con demasiadas features

---

## 3. Linear SVM

```python
from modelos import _pipeline_svm

clf = _pipeline_svm(C=0.5)
```

### Recomendado para

- Texto ruidoso
- Comentarios reales
- Jerga y abreviaciones
- Alta dimensionalidad

### Ventajas

- Excelente generalización
- Robusto ante ruido
- Buen rendimiento en NLP

### Limitaciones

- Más lento que Naive Bayes
- Requiere calibración para probabilidades

---

# Cómo agregar un nuevo modelo

Agregar una nueva función en `modelos.py`:

```python

def _pipeline_rf():
    ...
```

Después registrar el modelo:

```python
CATALOGO_MODELOS = {
    "ComplementNB": _pipeline_cnb,
    "LogisticRegression": _pipeline_lr,
    "LinearSVM": _pipeline_svm,
    "RandomForest": _pipeline_rf
}
```

---

# Advertencias de uso

## 1. ComplementNB requiere valores positivos

Si se usan features negativas:

```text
ValueError: Negative values in data passed to ComplementNB
```

El pipeline ya corrige esto automáticamente mediante:

```python
_clip_negativos()
```

---

## 2. No modificar matrices sparse manualmente

Convertir matrices sparse a densas puede consumir demasiada RAM:

```python
X.toarray()
```

Evitar esta operación con datasets grandes.

---

## 3. Mantener compatibilidad de versiones

Se recomienda:

| Librería | Versión |
|---|---|
| scikit-learn | >= 1.3 |
| numpy | >= 1.24 |
| scipy | >= 1.10 |

---

## 4. No usar datasets vacíos

Verificar que:

```python
len(y) > 0
```

Antes de entrenar.

---

# Ejemplo rápido de entrenamiento

```python
from scipy.sparse import load_npz
import numpy as np
from modelos import CATALOGO_MODELOS
from sklearn.model_selection import train_test_split

X = load_npz("X_features_20250101.npz")
y = np.load("y_labels_20250101.npy")

X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

for nombre, factory in CATALOGO_MODELOS.items():
    clf = factory()
    clf.fit(X_tr, y_tr)

    acc = (clf.predict(X_te) == y_te).mean()

    print(f"{nombre}: {acc:.4f}")
```

---

# Ejemplos rápidos de análisis con pandas

## 1. Cargar métricas CSV

```python
import pandas as pd

metricas = pd.read_csv("metricas_modelos.csv")

print(metricas.head())
```

---

## 2. Ordenar modelos por accuracy

```python
import pandas as pd

metricas = pd.read_csv("metricas_modelos.csv")

ranking = metricas.sort_values(
    by="accuracy",
    ascending=False
)

print(ranking)
```

---

## 3. Filtrar modelos con F1 alto

```python
import pandas as pd

metricas = pd.read_csv("metricas_modelos.csv")

buenos = metricas[
    metricas["f1"] > 0.80
]

print(buenos)
```

---

## 4. Exportar resultados

```python
metricas.to_csv(
    "ranking_final.csv",
    index=False
)
```

---

# Flujo recomendado de trabajo

```text
1. Scraping de comentarios
2. Limpieza y normalización
3. Vectorización TF-IDF
4. Generación de features
5. Entrenamiento de modelos
6. Evaluación comparativa
7. Exportación de resultados
```

---

# Buenas prácticas

- Mantener backups de modelos entrenados.
- Validar datasets antes del entrenamiento.
- Usar `train_test_split(... stratify=y)`.
- Guardar métricas por experimento.
- No mezclar datasets incompatibles.
- Versionar resultados importantes.

---

# Solución de problemas

## Error: `ModuleNotFoundError`

Instalar dependencias:

```bash
pip install -r requirements.txt
```

---

## Error: memoria insuficiente

Reducir:

- Tamaño del vocabulario TF-IDF
- Número de features
- Tamaño del corpus

---

## Accuracy demasiado alta

Posible overfitting.

Revisar:

- Leakage de datos
- Features duplicadas
- División train/test incorrecta

---

# Conclusión

`modelos.py` centraliza los clasificadores utilizados en el pipeline de minería de opinión y permite:

- Comparar múltiples modelos fácilmente
- Mantener un diseño modular
- Reutilizar pipelines sklearn
- Automatizar experimentos
- Integrarse con análisis estadístico y NLP

El módulo está diseñado para integrarse directamente con el resto del pipeline de análisis de texto orientado a salud pública y minería de opinión.


# analizar_v4.py

Script principal de evaluación del pipeline de minería de opinión sobre ineficiencia del sector salud MX ante la influenza (CRISP-DM Fase 5 — Evaluación).

Consume los artefactos que genera `vectorizar_v4.py` (`.npz`, `.npy`, `.pkl`) e importa los clasificadores desde `modelos.py`. Produce un reporte JSON estructurado, un resumen narrativo en texto y, opcionalmente, gráficas PNG de curva de aprendizaje y matrices de confusión.

---

## Ubicación en el pipeline

```
vectorizar_v4.py
  ├── X_features*.npz
  ├── y_labels*.npy
  └── vectorizador*.pkl
          ↓
    analizar_v4.py  ←── modelos.py (CNB · LR · LinearSVM)
          ↓
  ├── reporte_v4_*.json
  ├── resumen_narrativo_v4_*.txt
  ├── errores_criticos_*.json
  ├── curva_<modelo>_*.png        (si matplotlib instalado)
  └── confusion_<modelo>_*.png    (si matplotlib instalado)
```

---

## Requisitos

```bash
pip install scikit-learn numpy scipy
pip install matplotlib          # opcional — activa gráficas PNG
```

| Paquete | Versión mínima | Uso |
|---|---|---|
| scikit-learn | 1.3 | Modelos, métricas, validación cruzada |
| numpy | 1.24 | Álgebra y arrays |
| scipy | 1.10 | Matrices sparse, Spearman, χ² |
| matplotlib | 3.7 | Curvas de aprendizaje y matrices de confusión (opcional) |

`modelos.py` debe estar en el mismo directorio que este script.

---

## Uso

```bash
python3 analizar_v4.py
```

El script detecta automáticamente los artefactos disponibles en su directorio. Si hay más de un archivo de cada tipo presenta un menú numerado para elegir.

**Archivos que debe encontrar antes de ejecutar:**

```
X_features*.npz        ← matriz de features (salida de vectorizar_v4.py)
y_labels*.npy          ← etiquetas binarias  (0=media, 1=alta)
vectorizador*.pkl      ← artefactos del vectorizador (TF-IDF, selector, cols_num)
dataset_*.json         ← opcional: activa análisis Spearman, χ² y errores BERT
```

---

## Qué hace paso a paso

### ① Detección automática de features BERT

Al cargar `X`, el script busca la columna `prob_neg` en `features_numericas` del `.pkl` y comprueba si al menos el 10 % de las filas tiene valor distinto de cero.

- **BERT activo** → se usa `LogisticRegression` como estimador principal para validación cruzada y calibración (maneja correlación entre `prob_neg` y `sentimiento_num`).
- **BERT inactivo** → se usa `ComplementNB` (corpus procesado sin `auto_etiquetar.py`).

Este selector no afecta la comparación de modelos, que siempre evalúa los tres clasificadores del catálogo.

---

### ② Validación cruzada robusta (10-fold estratificado)

```python
cv_res = validacion_cruzada_robusta(X, y, bert_activo=bert_activo)
```

Métricas calculadas con IC 95% (`x̄ ± 1.96 · σ / √k`) citables en tesis:

| Campo | Descripción |
|---|---|
| `f1_weighted.media` | F1 ponderado promedio entre folds |
| `f1_weighted.ic95_pm` | Semiancho del intervalo de confianza 95% |
| `f1_alta.media` | F1 de la clase positiva (ineficiencia alta) |
| `roc_auc.media` | AUC promedio |
| `overfitting.gap` | Diferencia train F1 − test F1 |
| `overfitting.detectado` | `true` si gap > 10% |
| `veredicto` | Texto interpretable: "muy estable", "inestable", etc. |

---

### ③ Calibración isotónica

```python
clf_cal, probs_cal, y_pred_cal, cal_res = entrenar_con_calibracion(
    X_tr, y_tr, X_te, y_te, bert_activo=bert_activo)
```

Entrena el estimador principal dos veces: sin calibrar (línea base) y con `CalibratedClassifierCV(method="isotonic", cv=5)`. Compara:

- **Brier Score:** mide la exactitud de las probabilidades predichas (menor = mejor). Un modelo sin calibrar infla las probabilidades en los extremos.
- **Log-Loss:** penaliza más las predicciones incorrectas de alta confianza.
- **`mejora_brier_pct`:** si es positivo, la calibración mejoró la calidad probabilística.

El modelo calibrado (`clf_cal`) se usa en los pasos siguientes para el análisis de errores y la extracción de features.

---

### ④ Análisis de errores críticos

```python
errores_res = analizar_errores_criticos(
    X_te, y_te, y_pred_cal, probs_cal, registros, idx_te)
```

Identifica casos donde el modelo calibrado se equivocó **con alta confianza** (umbral: 0.80).

**Tipos de error:**

| Tipo | Definición | Coste en salud pública |
|---|---|---|
| FN confiado | Modelo dijo MEDIA con ≥ 80% confianza, era ALTA | Alto — queja real ignorada |
| FP confiado | Modelo dijo ALTA con ≥ 80% confianza, era MEDIA | Medio — falsa alarma |

Para cada error se reporta: texto original, tokens principales, diagnóstico automático, y análisis de desacuerdo BERT vs TF-IDF (si el dataset pasó por `auto_etiquetar.py`).

**Diagnóstico de desacuerdo BERT:**

| Situación | Interpretación para tesis |
|---|---|
| BERT correcto, TF-IDF incorrecto | Léxico de dominio insuficiente — ampliar `TERMINOS_DOMINIO` |
| Ambos incorrectos | Caso genuinamente difícil: sarcasmo, queja implícita o ambigüedad |
| BERT incorrecto, TF-IDF correcto | Contexto ambiguo que confunde al modelo de sentimiento |

---

### ⑤ Curva de aprendizaje (genérica)

```python
# Función genérica — acepta cualquier pipeline sklearn
curva_res = curva_aprendizaje(pipeline, X, y, nombre="MiModelo")
```

Calcula F1-weighted en 8 tamaños de entrenamiento (10%–100%) con validación cruzada de 5 folds. Diagnóstico automático:

| Condición | Diagnóstico | Acción recomendada |
|---|---|---|
| `gap_final > 0.12` | Overfitting moderado | Reducir vocabulario o aumentar `alpha` |
| `pendiente_test > 0.02` | El modelo sigue mejorando | Continuar scraping |
| `gap_final < 0.03` | Modelo saturado | Mejorar features, no añadir datos |
| Resto | Equilibrado | Ninguna |

**Gráfica PNG:**

```python
graficar_curva_aprendizaje(pipeline, X, y, nombre, carpeta, ts)
```

Guarda `curva_<nombre>_<timestamp>.png`. Funciona con cualquier estimador sklearn. Si `matplotlib` no está instalado, la función retorna `None` sin lanzar error.

---

### ⑥ Comparación de modelos — tabla Precision / Recall / F1

```python
comparacion = comparar_modelos(X_tr, y_tr, X_te, y_te, carpeta, ts)
```

Itera sobre `CATALOGO_MODELOS` de `modelos.py` y evalúa cada clasificador en el mismo split 80/20. Reporta por cada modelo:

| Métrica | Descripción |
|---|---|
| `precision_alta` | De las predicciones ALTA, cuántas eran correctas |
| `recall_alta` | De las quejas reales ALTA, cuántas detectó el modelo |
| `f1_alta` | Media armónica de Precision y Recall para clase ALTA |
| `f1_weighted` | F1 ponderado por soporte de clase (métrica global) |
| `roc_auc` | Área bajo la curva ROC |
| `accuracy` | Porcentaje de predicciones correctas |
| `confusion_matrix` | Matriz 2×2 cruda |

**Por qué Recall es la métrica prioritaria en salud pública:**  
Un Falso Negativo (queja real no detectada) tiene coste mayor que un Falso Positivo (falsa alarma). El sistema debería preferir un modelo con Recall alto, incluso si eso reduce un poco la Precision. La tabla lo señala explícitamente en el veredicto.

**Tabla impresa en consola (ejemplo):**

```
  ──────────────────────────────────────────────────────────────────
  TABLA COMPARATIVA — CLASE ALTA (ineficiencia detectada)
  Prec↑ = Precision(alta) | Recall↑ = Recall(alta) | F1↑ = F1(alta)
  ──────────────────────────────────────────────────────────────────
  Modelo                   Prec↑  Recall↑     F1↑    F1-W     AUC     Acc
  ──────────────────────────────────────────────────────────────────
  LinearSVM               0.8312   0.7941  0.8122  0.8247  0.8901  0.8375 ★
  LogisticRegression      0.8104   0.7823  0.7961  0.8089  0.8834  0.8250
  ComplementNB            0.7891   0.7412  0.7644  0.7823  0.8601  0.7975
  ──────────────────────────────────────────────────────────────────

  → Mejor para detectar INEFICIENCIAS: LinearSVM
     Precision=0.8312 | Recall=0.7941 | F1=0.8122
```

**Gráfica PNG de confusión:**

```python
graficar_matriz_confusion(y_te, y_pred, nombre, carpeta, ts)
```

Guarda `confusion_<nombre>_<timestamp>.png` con conteos absolutos y normalizada. Función genérica — acepta cualquier par `(y_te, y_pred)`.

---

### ⑦ Top features e interpretación

```python
top_terms     = extraer_top_terms(clf_cal, vec, selector)
tabla_hibrida = top_features_hibrido(clf_cal, vec, selector, art, n=20)
```

`extraer_top_terms` soporta CNB (usa `feature_log_prob_`) y LR/SVM (usa `coef_`). Cada término se enriquece con su dimensión narrativa (`falta_insumos`, `tiempo_espera`, etc.).

`top_features_hibrido` genera la tabla mixta texto + BERT con coeficiente, dirección (`→ alta` / `→ media`) y tipo (`texto_tfidf`, `BERT`, `enriquecimiento`). Solo funciona con estimadores que tienen `coef_` (LR y SVM calibrado, no CNB).

---

## Salidas generadas

Todos los archivos se guardan en el mismo directorio del script con timestamp `YYYYMMDD_HHMMSS`.

| Archivo | Contenido |
|---|---|
| `reporte_v4_<ts>.json` | Todas las métricas estructuradas en JSON |
| `resumen_narrativo_v4_<ts>.txt` | Texto interpretable listo para copiar en tesis |
| `errores_criticos_<ts>.json` | Lista de FN/FP con diagnóstico y contexto |
| `curva_<modelo>_<ts>.png` | Curva de aprendizaje por modelo |
| `confusion_<modelo>_<ts>.png` | Matriz de confusión por modelo |

### Estructura del JSON de reporte

```json
{
  "meta": {
    "version": "4.0",
    "estimador_cv": "LR_calibrado",
    "bert_activo": true,
    "mejor_modelo": "LinearSVM",
    "catalogo": ["ComplementNB", "LogisticRegression", "LinearSVM"]
  },
  "validacion_cruzada": { ... },
  "calibracion": { ... },
  "comparacion_modelos": [ { "modelo": "...", "precision_alta": ..., ... } ],
  "top_terms_por_clase": { "alta": [...], "media": [...] },
  "top_features_hibrido": [ { "rank": 1, "nombre": "prob_neg", "coef": 2.31, ... } ],
  "dimensiones_narrativas": { "falta_insumos": { "peso_total": ..., "terminos": [...] } },
  "correlaciones_spearman": [ { "feature": "ifb_score", "rho": 0.42, ... } ],
  "chi2_categorias": [ { "categoria": "desabasto_medicamentos", "chi2": ..., ... } ],
  "curva_aprendizaje": { "modelo": "...", "puntos": [...], "diagnostico": "..." }
}
```

---

## Cómo añadir un modelo nuevo

1. Abrir `modelos.py` y definir `_pipeline_nuevo(...)`.
2. Añadirlo a `CATALOGO_MODELOS`:

```python
CATALOGO_MODELOS = {
    "ComplementNB":       _pipeline_cnb,
    "LogisticRegression": _pipeline_lr,
    "LinearSVM":          _pipeline_svm,
    "RandomForest":       _pipeline_rf,   # ← nuevo
}
```

3. Ejecutar `analizar_v4.py` sin cambios — el nuevo modelo aparece automáticamente en la tabla comparativa, en las gráficas de confusión y en el reporte JSON.

---

## Configuración principal

Todas las constantes están al inicio del script:

| Constante | Default | Descripción |
|---|---|---|
| `TEST_SIZE` | `0.20` | Fracción del corpus para test |
| `RANDOM_STATE` | `42` | Semilla de reproducibilidad |
| `N_FOLDS` | `10` | Folds para validación cruzada |
| `IC_Z` | `1.96` | Z para IC 95% |
| `METODO_CALIBRACION` | `"isotonic"` | Método de calibración de probabilidades |
| `UMBRAL_CONFIANZA` | `0.80` | Umbral para clasificar error como "confiado" |
| `N_ERRORES_EXPORTAR` | `20` | Máximo de FN/FP exportados al JSON |
| `CURVA_PUNTOS` | `8` | Puntos en la curva de aprendizaje |
| `UMBRAL_BERT_ACTIVO` | `0.10` | Fracción mínima de filas con `prob_neg != 0` para activar LR |
| `TOP_N_TERMS` | `25` | Features a mostrar por clase |

---

## Archivos relacionados

| Archivo | Rol |
|---|---|
| `modelos.py` | Define los tres pipelines y `CATALOGO_MODELOS` |
| `vectorizar_v4.py` | Genera los artefactos de entrada (`.npz`, `.npy`, `.pkl`) |
| `auto_etiquetar.py` | Añade features BERT al dataset JSON (activa modo híbrido) |

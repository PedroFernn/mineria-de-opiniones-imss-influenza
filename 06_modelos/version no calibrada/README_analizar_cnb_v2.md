# 🔬 Análisis CNB — Validación Robusta + Calibración + Errores `v2`

Quinta y última etapa del pipeline de modelado (CRISP-DM Fase 5: Evaluación). Toma los artefactos producidos por el vectorizador y ejecuta un ciclo completo de entrenamiento, validación estadística, calibración de probabilidades, diagnóstico de errores y comparación de modelos. Produce los reportes finales listos para citar en una tesis académica.

---

## 📋 Descripción

El script entrena un clasificador **Complement Naive Bayes (CNB)** sobre la matriz X/y generada por `vectorizar_v3.py` y lo somete a cinco análisis de robustez que van más allá del accuracy puntual: validación cruzada con intervalos de confianza, calibración isotónica de probabilidades, análisis forense de los errores más confiados, curva de aprendizaje y comparación directa con LogisticRegression y LinearSVC. Todos los resultados se exportan en tres formatos: JSON estructurado, texto narrativo listo para copiar en un informe y JSON de errores críticos para diagnóstico.

---

## ✨ Mejoras implementadas en v2

| # | Mejora | Valor para tesis académica |
|---|---|---|
| ① | **Validación cruzada StratifiedKFold 10-fold con IC 95%** | Permite afirmar "F1 = 82% ± 1.2% (IC 95%)" en lugar de un dato puntual no replicable |
| ② | **Calibración isotónica de probabilidades** | Convierte el score de CNB (inflado por independencia condicional) en una probabilidad real citeable: "85% de probabilidad de queja clínicamente relevante" |
| ③ | **Análisis de errores críticos por confianza** | Exporta los FN/FP donde el modelo era muy seguro y se equivocó — diagnóstico directo de qué subtemas fallan |
| ④ | **Curva de aprendizaje** | Responde: ¿el modelo necesita más datos o mejores features? |
| ⑤ | **Comparación CNB vs LogisticRegression vs LinearSVC** | Justifica la elección del clasificador con evidencia comparativa en el mismo split |

**Heredadas de v1:**

| # | Función | Descripción |
|---|---|---|
| ⑥ | `extraer_top_terms()` | Top-N términos predictivos por clase extraídos de `feature_log_prob_` del CNB |
| ⑦ | `correlacion_spearman()` | Correlación ρ de Spearman entre features numéricas (IFB, polaridad, impacto) y la variable objetivo |
| ⑧ | `prueba_chi2_categorias()` | Prueba χ² entre categorías semánticas del enriquecedor y la clase de relevancia |
| ⑨ | `analizar_dimensiones()` | Agrupa los top términos en las 6 dimensiones narrativas de la "burocracia del dolor" |
| ⑩ | `generar_narrativa()` | Reporte en texto plano con hallazgos listos para copiar en un informe o tesis |

---

## 📂 Archivos de entrada y salida

### Entrada (los cuatro artefactos del vectorizador)

```
X_features_v2_*.npz          ← Matriz X sparse (docs × features)
y_labels_v2_*.npy             ← Vector y de etiquetas (0/1)
vectorizador_v2_*.pkl         ← Objeto vectorizador + scaler + selector
dataset_normalizado_*.json    ← Dataset original para texto en análisis de errores
  (o dataset_enriquecido_*.json)
```

### Salida (tres artefactos de análisis)

```
reporte_cnb_v2_YYYYMMDD_HHMMSS.json        ← Reporte estructurado completo
resumen_narrativo_v2_YYYYMMDD_HHMMSS.txt   ← Texto listo para tesis
errores_criticos_YYYYMMDD_HHMMSS.json      ← FN/FP de alta confianza para diagnóstico
```

---

## ⚙️ Requisitos

**Python 3.10+**

```bash
pip install scikit-learn numpy scipy
```

El script verifica e instala estas dependencias automáticamente al arrancar si no están presentes.

---

## 🚀 Uso

```bash
python3 analizar_cnb_v2.py
```

El script detecta automáticamente los archivos disponibles en la carpeta. Si hay más de uno de cada tipo (varias ejecuciones del vectorizador), muestra un menú de selección. Si solo hay uno de cada tipo, lo carga directamente sin preguntar.

El dataset JSON es **opcional**: sin él el análisis de errores críticos no puede mostrar el texto original de los comentarios fallidos, pero el resto del pipeline funciona con normalidad.

### 🕐 Qué esperar durante la ejecución

| Lo que ves en consola | Causa | Duración aprox. |
|---|---|---|
| `Ejecutando 10-fold estratificado...` | Validación cruzada completa | 10–60 s según corpus |
| `Calibrando con método 'isotonic' (cv=5)...` | Entrenamiento del calibrador | 5–20 s |
| `Analizando errores críticos...` | Clasificación de FN/FP | Inmediato |
| `Calculando curva de aprendizaje (8 puntos)...` | 8 tamaños × 5-fold | 30–120 s |
| `Comparando CNB vs LR vs LinearSVC...` | Tres modelos en el mismo split | 10–40 s |

---

## 🔧 Configuración principal

| Constante | Valor por defecto | Descripción |
|---|---|---|
| `N_FOLDS` | `10` | Número de folds para validación cruzada estratificada |
| `IC_Z` | `1.96` | z-score para IC 95% (usar `2.576` para IC 99%) |
| `TEST_SIZE` | `0.20` | Fracción del corpus reservada para test en el split único |
| `RANDOM_STATE` | `42` | Semilla de aleatoriedad para reproducibilidad |
| `METODO_CALIBRACION` | `"isotonic"` | Método de calibración: `"isotonic"` o `"sigmoid"` |
| `N_ERRORES_EXPORTAR` | `20` | Número máximo de errores críticos exportados por tipo |
| `UMBRAL_CONFIANZA` | `0.80` | Probabilidad mínima para considerar un error "crítico" |
| `CURVA_PUNTOS` | `8` | Puntos de tamaño de entrenamiento en la curva de aprendizaje |
| `TOP_N_TERMS` | `25` | Número de términos predictivos extraídos por clase |

---

## 🧠 Arquitectura técnica avanzada

---

### 1. Por qué Complement Naive Bayes y no Multinomial NB

Complement Naive Bayes es una variante diseñada específicamente para **corpus desbalanceados**, que es exactamente el caso de este corpus: la clase `alta` (quejas explícitas) suele ser más frecuente que `media` en un dataset limpiado agresivamente, o puede estar subrepresentada si el filtro OR/AND fue muy estricto.

La diferencia clave está en cómo calcula los pesos:

| | Multinomial NB | Complement NB |
|---|---|---|
| **Calcula** | P(feature \| clase) para cada clase | P(feature \| complemento de la clase) |
| **Mejor con** | Clases balanceadas | Clases desbalanceadas |
| **Problema** | Sobreestima clases mayoritarias | Más robusto ante desbalance |

CNB entrena el clasificador de la clase A estimando la distribución del **complemento** (todo lo que no es A), lo que produce estimaciones más estables cuando una clase tiene pocos ejemplos. En este corpus, captura mejor los textos de alta relevancia aunque sean minoría.

#### El pipeline sparse-compatible

CNB requiere `X ≥ 0`. El vectorizador concatena TF-IDF (siempre ≥ 0) con features numéricas escaladas con `StandardScaler` (que puede producir negativos). El pipeline resuelve esto en tres pasos encadenados:

```python
Pipeline([
    ("scaler", MaxAbsScaler()),     # divide cada columna por su máximo absoluto → [-1, 1]
    ("clip",   FunctionTransformer( # fuerza X ≥ 0 respetando el formato sparse
                  _clip_negativos)),
    ("cnb",    ComplementNB(alpha=0.5)),
])
```

El clip opera directamente sobre `X.data` (solo los valores no-cero almacenados en la matriz CSR) para no destruir la estructura sparse ni convertir la matriz a densa, lo que consumiría órdenes de magnitud más memoria.

---

### 2. Validación cruzada robusta — por qué 10-fold y cómo leer el IC

#### Por qué 10-fold para una tesis

| Criterio | 5-fold | 10-fold |
|---|---|---|
| Tamaño de cada conjunto test (n≈2500) | ~500 muestras | ~250 muestras |
| Varianza del estimador de métrica | Alta | ~30% menor |
| Costo computacional | Bajo | Moderado |
| Citabilidad en literatura académica | Aceptable | Estándar |

Con 250 muestras por fold se calculan métricas estadísticamente estables. Con 5-fold y 125 muestras, un solo fold con distribución atípica puede distorsionar la media significativamente.

**La estratificación es obligatoria** para corpus desbalanceados: garantiza que cada fold mantenga la misma proporción de clases alta/media que el dataset completo. Sin estratificación, un fold podría tener el 90% de muestras `alta` y el siguiente el 40%, produciendo varianzas artificialmente altas.

#### Cómo interpretar el intervalo de confianza

El IC se calcula con la fórmula estándar del error de la media sobre las distribuciones empíricas de los K folds:

$$IC_{95\%} = \bar{x} \pm z \cdot \frac{\sigma}{\sqrt{K}}$$

donde $\bar{x}$ es la media de los 10 scores, $\sigma$ su desviación estándar, $K=10$ y $z=1.96$.

```
Ejemplo de lectura:
  F1 weighted = 0.8243 ± 0.0118  (IC 95%)
  IC: [0.8125, 0.8361]

Interpretación correcta para tesis:
  "En el 95% de posibles muestras aleatorias del mismo corpus, el modelo
  obtendría un F1 entre 0.8125 y 0.8361."

Interpretación incorrecta:
  "El modelo tiene 95% de probabilidad de obtener F1 entre esos valores"
  (eso es un intervalo bayesiano, no frecuentista)
```

#### Detección automática de overfitting

Si el F1 en entrenamiento supera al de test por más del 10%, el campo `overfitting.detectado` es `true`. Un gap de 0–5% es normal (bias de entrenamiento), 5–10% es aceptable, >10% indica que el modelo memoriza el corpus de entrenamiento en vez de generalizar.

---

### 3. Calibración de probabilidades — por qué CNB sin calibrar no es citeable

CNB asume **independencia condicional entre features**: dado que el texto es relevante, la presencia de "desabasto" es independiente de la presencia de "IMSS". Esta asunción es falsa en texto natural (las palabras co-ocurren), lo que produce probabilidades extremas:

```
CNB sin calibrar → P(alta | texto con 3 términos de queja) = 0.997
Probabilidad real → P(alta | mismo texto) ≈ 0.82

CNB sin calibrar → P(alta | texto ambiguo) = 0.03
Probabilidad real → P(alta | mismo texto) ≈ 0.30
```

Para una tesis de salud pública, afirmar que "un comentario tiene 99.7% de probabilidad de representar una queja clínica" cuando la probabilidad real es 82% invalida cualquier análisis cuantitativo basado en esas probabilidades.

#### Calibración isotónica vs. sigmoid (Platt scaling)

| | Isotónica | Sigmoid (Platt) |
|---|---|---|
| **Modelo** | Función monótona no paramétrica | Regresión logística sobre scores |
| **Flexibilidad** | Alta — aprende cualquier forma | Baja — solo transforma linealmente |
| **Requisito de datos** | ≥ 1,000 muestras de calibración | Funciona con <500 |
| **Riesgo** | Sobreajuste con corpus pequeños | Infraajuste con distribuciones no lineales |

`CalibratedClassifierCV` con `cv=5` divide el conjunto de entrenamiento en 5 sub-folds, entrena el CNB en 4 y calibra en el quinto, rotando. Esto evita que el calibrador vea los mismos datos con los que entrenó el clasificador base, previniendo filtración de datos.

#### Métricas de calidad de calibración

**Brier Score** — error cuadrático medio entre la probabilidad predicha $\hat{p}$ y la etiqueta real $y$:

$$BS = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i)^2$$

Rango: 0 (perfecto) → 1 (pésimo). Para referencia, un clasificador que siempre predice 0.5 obtiene BS = 0.25. Un BS < 0.15 en este corpus indica calibración aceptable para uso académico.

**Log-loss** — penaliza más severamente las predicciones confiadas y equivocadas que el Brier Score. Útil para detectar si el modelo tiene muchos casos con alta confianza incorrecta.

---

### 4. Análisis de errores críticos — los Falsos Negativos son los más costosos

Un **error crítico** es un caso donde el modelo estaba seguro (probabilidad ≥ `UMBRAL_CONFIANZA = 0.80`) pero se equivocó. La confianza se define como:

```
Si pred = alta  → confianza = P(alta)
Si pred = media → confianza = P(media) = 1 - P(alta)
```

#### Falsos Negativos vs. Falsos Positivos en este dominio

| Tipo | Definición | Impacto en investigación |
|---|---|---|
| **FN confiado** | Modelo dijo MEDIA con 90% de confianza pero era ALTA | ⚠️ El sistema habría ignorado una señal de alarma real de ineficiencia clínica |
| **FP confiado** | Modelo dijo ALTA con 90% de confianza pero era MEDIA | ✅ Menos grave — sobreestima el volumen de quejas pero no pierde señales reales |

Para un análisis de vigilancia epidemiológica social, el costo asimétrico entre FN y FP es relevante: **perder una queja real es más costoso que incluir un falso positivo**. El script captura este sesgo exportando los FN primero y generando recomendaciones accionables sobre cómo reducirlos.

#### El sistema de diagnóstico automático

Para cada error crítico, `_diagnosticar_error()` inspecciona los tokens del texto e intenta identificar el patrón causante:

| Patrón detectado | Diagnóstico | Recomendación generada |
|---|---|---|
| Tokens de pago privado en FN | El modelo no asocia `"particular"` con queja alta | Añadir al diccionario del normalizador |
| Tokens de consecuencia clínica en FN | Palabras de impacto no detectadas (`"fallecer"`, `"morir"`) | Ampliar TERMINOS_DOMINIO |
| Texto muy corto (<8 tokens) en FN | Insuficiente señal léxica | Bajar MIN_CHARS en el limpiador |
| Institución sin problema en FP | Mención de IMSS sin indicador de queja | Añadir bigramas contextuales al vectorizador |
| Vacunación informativa en FP | Comentario positivo sobre vacunas clasificado como queja | Añadir señales de contexto positivo |

Las recomendaciones son **directamente accionables sobre otros scripts del pipeline**, cerrando el ciclo CRISP-DM con retroalimentación concreta.

---

### 5. Curva de aprendizaje — ¿más datos o mejores features?

La curva evalúa el F1 del modelo en 8 tamaños de entrenamiento crecientes (de 10% a 100% del corpus), usando 5-fold CV en cada punto. El diagnóstico automático interpreta la forma resultante:

```
Caso A — Las dos curvas convergen con pocos datos y el gap es pequeño:
  gap_final < 0.03  →  "modelo saturado — mejorar features es más efectivo"
  Recomendación: ampliar diccionarios, añadir BERT, ajustar SelectKBest

Caso B — La curva de test sigue subiendo al añadir datos:
  pendiente_test > 0.02 en la segunda mitad  →  "más datos mejorarían el modelo"
  Recomendación: continuar scraping con el scraper, ampliar subreddits

Caso C — La curva de entrenamiento es mucho más alta que la de test:
  gap_final > 0.12  →  "overfitting moderado"
  Recomendación: aumentar alpha en CNB, reducir MAX_FEATURES, más regularización
```

La curva se calcula sobre el `_pipeline_cnb` completo (MaxAbsScaler → clip → CNB) para que cada punto refleje el comportamiento real del modelo incluyendo el preprocesamiento, no solo el clasificador.

---

### 6. Comparación de modelos en el mismo split

Los tres modelos se evalúan sobre **exactamente el mismo split estratificado 80/20**, lo que garantiza que las diferencias de métricas son reales y no artefactos de distintos conjuntos de test:

```
Split único (RANDOM_STATE=42, stratify=y)
    ├── X_tr (80%) ─────────────────────────────────────────────────┐
    │                 ┌── ComplementNB (pipeline MaxAbs+clip+CNB)   │
    │                 ├── LogisticRegression (pipeline MaxAbs+LR)   │
    │                 └── LinearSVC calibrado (pipeline MaxAbs+SVC) │
    └── X_te (20%) ←─────── evaluación de los tres modelos ─────────┘
```

**LinearSVC** no tiene `predict_proba` nativo, por lo que se envuelve en `CalibratedClassifierCV` para poder calcular AUC-ROC. Esto añade un overhead de entrenamiento pero lo hace comparable con los otros dos modelos.

Los resultados se ordenan por F1 weighted descendente. El mejor modelo se cita explícitamente en el reporte narrativo con justificación.

---

### 7. Extracción de términos predictivos desde `feature_log_prob_`

CNB almacena en `feature_log_prob_` la log-probabilidad de cada feature dado el complemento de cada clase. Para extraer los términos más predictivos de la clase `alta`, el script toma los pesos de la fila correspondiente y los ordena de mayor a menor:

```python
pesos = -log_probs[idx_clase]   # negación: mayor peso = más predictivo
top_idx = np.argsort(pesos)[-n:][::-1]
```

La negación convierte las log-probabilidades (negativas) en scores positivos ordenables.

#### Navegación recursiva del estimador anidado

`CalibratedClassifierCV(Pipeline([...]))` crea una estructura de tres niveles de anidamiento. La función `_extraer_cnb()` navega recursivamente hasta encontrar el objeto con `feature_log_prob_`, manejando cuatro casos posibles:

```
CalibratedClassifierCV
    └── calibrated_classifiers_[0]
           └── estimator  (el Pipeline)
                  └── named_steps["cnb"]  ← aquí está feature_log_prob_
```

---

### 8. Pruebas estadísticas heredadas de v1

#### Correlación de Spearman (features numéricas ↔ y)

Spearman es más adecuado que Pearson para este corpus porque:
- Las features numéricas (IFB, polaridad, impacto) no siguen distribución normal
- La relación entre, por ejemplo, IFB y relevancia puede ser monótona pero no lineal

Un ρ = 0.45 (p < 0.001) en `ifb_score` significa que los comentarios con mayor fricción burocrática tienden a ser de relevancia alta, con un efecto de magnitud "moderada".

#### Prueba χ² con corrección de Yates + Odds Ratio

Para cada categoría semántica del enriquecedor (desabasto, burocracia, corrupción, etc.) se construye una tabla de contingencia 2×2 y se calcula:

- **χ²** con corrección de Yates (para celdas pequeñas): prueba si la categoría y la clase son independientes
- **Odds Ratio**: cuántas veces más probable es ser `alta` si el comentario contiene esa categoría vs. si no la contiene. OR > 1.5 → asociado con alta; OR < 0.67 → asociado con media

La corrección de Yates reduce el riesgo de falsos positivos cuando alguna celda tiene pocos casos (mínimo requerido: ≥ 3 menciones de la categoría).

---

## 📊 Estructura de los archivos de salida

### `reporte_cnb_v2_*.json`

```json
{
  "meta": {
    "generado": "2025-04-01T18:00:00",
    "version": "2.0",
    "modelo": "CalibratedCNB",
    "corpus": "X_features_v2_*.npz"
  },
  "validacion_cruzada": {
    "n_folds": 10,
    "ic_nivel": "95%",
    "f1_weighted": {
      "media": 0.8243, "std": 0.0118,
      "ic95_pm": 0.0073,
      "ic95_lo": 0.8170, "ic95_hi": 0.8316,
      "valores_por_fold": [0.81, 0.83, 0.82, ...],
      "estable": true
    },
    "overfitting": {
      "f1_train_media": 0.8901,
      "f1_test_media":  0.8243,
      "gap": 0.0658,
      "detectado": false
    },
    "veredicto": "estable (σ < 4%)"
  },
  "calibracion": {
    "sin_calibrar": { "brier_score": 0.12430, "log_loss": 0.41230 },
    "calibrado":    { "brier_score": 0.09871, "mejora_brier_pct": 20.6 },
    "interpretacion": "mejorada — usar probabilidades calibradas para inferencia"
  },
  "comparacion_modelos": [
    { "modelo": "LogisticRegression", "f1_weighted": 0.8512, "f1_alta": 0.8801, "roc_auc": 0.9203 },
    { "modelo": "ComplementNB",       "f1_weighted": 0.8243, "f1_alta": 0.8612, "roc_auc": 0.8994 },
    { "modelo": "LinearSVC",          "f1_weighted": 0.8198, "f1_alta": 0.8543, "roc_auc": 0.8821 }
  ],
  "curva_aprendizaje": {
    "puntos": [
      {"n_train": 380, "f1_train_media": 0.91, "f1_test_media": 0.79, "gap": 0.12},
      ...
      {"n_train": 3832, "f1_train_media": 0.89, "f1_test_media": 0.82, "gap": 0.07}
    ],
    "diagnostico": "equilibrado — modelo estable"
  }
}
```

### `errores_criticos_*.json`

```json
{
  "umbral_confianza": 0.80,
  "total_fn_confiados": 23,
  "total_fp_confiados": 14,
  "fn_exportados": [
    {
      "indice_original":   1482,
      "etiqueta_real":     "alta",
      "prediccion":        "media",
      "prob_clase_alta":   0.14,
      "confianza_erronea": 0.86,
      "texto_original":    "Tuve que ir al particular porque el IMSS...",
      "ifb_score":         4.5,
      "polaridad_score":   6.2,
      "contextos":         ["costos_financiamiento"],
      "tipo_error":        "FN",
      "diagnostico":       "posible FN por pago_privado: el modelo no asocia 'particular' con queja alta"
    }
  ],
  "recomendaciones": [
    "ACCIÓN FN: Añadir 'particular', 'médico_privado', 'de_mi_bolsillo' a TERMINOS_DOMINIO..."
  ]
}
```

### `resumen_narrativo_v2_*.txt`

Texto listo para copiar en sección de resultados de una tesis, con:
- Hallazgo principal con los top-3 términos predictivos
- Sección de validación cruzada con IC y veredicto de estabilidad
- Sección de calibración con mejora en Brier Score
- Top-5 errores más confiados con diagnóstico
- Tabla comparativa de modelos
- Correlaciones Spearman y χ² más significativas
- Top-15 términos predictivos por clase con barras visuales

---

## 📈 Flujo de ejecución

```
pedir_archivos()  →  selección de X, y, pkl, JSON (opcional)
      │
      ▼
cargar_todo()  →  load_npz(X)  |  np.load(y)  |  pickle.load(pkl)
      │
      ▼
train_test_split(stratify=y, test_size=0.20)  →  split único reproducible
      │
      ▼
┌─── ① validacion_cruzada_robusta(X, y) ────────────────────────────────┐
│    StratifiedKFold(n_splits=10)                                        │
│    cross_validate(Pipeline[MaxAbs+clip+CNB], scoring={f1,auc,acc})    │
│    → media, std, IC 95%, veredicto de estabilidad, gap overfitting     │
└───────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─── ② entrenar_con_calibracion(X_tr, y_tr, X_te, y_te) ───────────────┐
│    CNB raw → brier_score_loss, log_loss (línea base)                   │
│    CalibratedClassifierCV(Pipeline, method="isotonic", cv=5)           │
│    → probs_cal calibradas  |  Brier Score  |  Log-loss  |  F1         │
└───────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─── ③ analizar_errores_criticos(X_te, y_te, probs_cal) ───────────────┐
│    Filtra errores con confianza ≥ 0.80                                 │
│    Separa FN (más graves) de FP                                        │
│    _diagnosticar_error() → patrón causante                            │
│    _generar_recomendaciones() → acciones sobre otros scripts           │
└───────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─── ④ curva_aprendizaje(X, y) ─────────────────────────────────────────┐
│    learning_curve(Pipeline, train_sizes=linspace(0.1, 1.0, 8))        │
│    → 8 puntos  |  diagnóstico automático: saturado/overfitting/crecer  │
└───────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─── ⑤ comparar_modelos(X_tr, y_tr, X_te, y_te) ───────────────────────┐
│    CNB | LogisticRegression | LinearSVC (calibrado)                   │
│    → F1-weighted  |  F1-alta  |  AUC-ROC  |  Accuracy                 │
│    → ordenados por F1-weighted descendente                             │
└───────────────────────────────────────────────────────────────────────┘
      │
      ▼
extraer_top_terms()     → navegación recursiva hasta feature_log_prob_
correlacion_spearman()  → ρ Spearman features numéricas ↔ y
prueba_chi2_categorias()→ χ² + Odds Ratio por categoría semántica
analizar_dimensiones()  → agrupa top_terms en 6 dimensiones narrativas
      │
      ▼
generar_narrativa()  →  texto .txt con hallazgos citables
      │
      ▼
guardar_todo()
  ├── reporte_cnb_v2_*.json
  ├── resumen_narrativo_v2_*.txt
  └── errores_criticos_*.json
```

---

## 🔬 Cómo usar los resultados en una tesis

### Citar la validación cruzada

```
"El clasificador CNB calibrado obtuvo un F1-weighted de X.XX ± Y.YY
(IC 95%, 10-fold StratifiedKFold), lo que indica un modelo [estable/inestable]
con AUC-ROC de X.XX ± Y.YY."
```

### Citar la calibración

```
"Las probabilidades brutas de CNB presentaron un Brier Score de X.XXXXX,
reducido a Y.YYYYY tras calibración isotónica (mejora del Z%), permitiendo
la interpretación probabilística directa de los scores de clasificación."
```

### Citar los términos predictivos

```
"Los términos con mayor peso predictivo para la clase de alta relevancia
fueron [término_1], [término_2] y [término_3], pertenecientes principalmente
a la dimensión narrativa de [dimensión_dominante], lo que confirma que el
ciudadano percibe la ineficiencia del sistema principalmente a través de
[descripción de la dimensión]."
```

### Interpretar el análisis de errores

Los FN de alta confianza son la contribución metodológica más valiosa del análisis: identifican directamente los subtemas del corpus que el modelo no cubre bien y generan recomendaciones concretas para mejorar el pipeline en iteraciones futuras, lo que es un argumento sólido para la sección de "limitaciones y trabajo futuro" de una tesis.

---

## 🔗 Posición en el pipeline completo

```
[1] scraper_influenza_stealth_v3.py   →  dataset crudo
[2] limpiar_dataset_v3.py             →  corpus limpio
[3] enriquecer_dataset_v3.py          →  corpus con IFB, polaridad, impacto
[4a] normalizar_nlp_v2.py             →  texto lematizado + peso_reddit
[4b] vectorizar_v3.py                 →  X_features.npz | y_labels.npy | pkl
                                              │
                                              ▼
[5] analizar_cnb_v2.py                ←  aquí estamos
     │
     ▼
reporte_cnb_v2.json       ← resultados para análisis estadístico
resumen_narrativo_v2.txt  ← texto para tesis / informe
errores_criticos.json     ← retroalimentación para iteración del pipeline
```

Los `errores_criticos.json` cierran el ciclo CRISP-DM: sus recomendaciones apuntan directamente a parámetros modificables en `limpiar_dataset_v3.py`, `enriquecer_dataset_v3.py` y `vectorizar_v3.py`, permitiendo una segunda iteración del pipeline con datos más limpios y features más discriminativas.

---

## ⚠️ Consideraciones metodológicas

- El split único (80/20) se usa para calibración, análisis de errores y comparación de modelos. La validación cruzada usa el corpus completo. Ambos resultados son complementarios: la CV reporta generalización, el split único permite análisis de casos individuales.
- Si `N_FOLDS = 10` y el corpus tiene menos de 200 registros, cada fold de test tendrá menos de 20 muestras — las métricas serán inestables. Usar `N_FOLDS = 5` para corpus pequeños.
- `METODO_CALIBRACION = "isotonic"` requiere ≥ 1,000 muestras en el conjunto de entrenamiento para ser estable. Con corpus más pequeños, cambiar a `"sigmoid"`.
- El reporte narrativo usa los top-5 términos predictivos para construir la interpretación. Si los primeros términos son tokens técnicos del normalizador (ej. `entidad_salud_imss`), reemplazarlos por su forma original en la cita de la tesis para legibilidad.

---

## 📝 Licencia

Uso académico / investigación. Consultar con el equipo responsable del proyecto antes de redistribuir o publicar datos recolectados.

# 🔬 Analizador CNB v3 — Clasificación y Validación Robusta `v3`

Paso **3** del pipeline de modelado (CRISP-DM Fase 5). Recibe los artefactos matriciales producidos por `vectorizar_v3.py` y ejecuta un pipeline completo de clasificación supervisada con selección automática de estimador, validación cruzada 10-fold con intervalos de confianza, calibración de probabilidades, análisis de errores críticos y reporte narrativo citable en tesis.

---

## 📋 Descripción

El analizador no reentrrena ni modifica el vectorizador: su propósito es **evaluar y diagnosticar el rendimiento del modelo de clasificación** sobre el corpus etiquetado. Detecta automáticamente si la matriz de entrada contiene features BERT (provenientes de `auto_etiquetar.py`) y elige entre `ComplementNB` o `LogisticRegression` según el tipo de datos — CNB para texto puro, LR para features híbridas correlacionadas. Los resultados incluyen métricas académicamente citables, diagnóstico de errores por tipo y dimensión narrativa, y un análisis de desacuerdo entre BERT y el clasificador para identificar qué "cerebro" contribuyó a cada error crítico.

---

## ✨ Novedades en v3

| # | Mejora | Descripción |
|---|---|---|
| ① | **Selector automático de estimador** | Detecta si `X` contiene features BERT (`prob_neg`, `sentimiento_num`…) y elige `LogisticRegression` en lugar de `ComplementNB` — LR maneja mejor datos híbridos correlacionados donde CNB viola la independencia condicional |
| ② | **Top-features híbrido texto + BERT** | Muestra nombres reales de variables numéricas junto a palabras TF-IDF (ej. `"prob_neg [BERT]"` y `"negligencia [texto]"`) con dirección del coeficiente y tipo de feature — citable en tesis |
| ③ | **Análisis de desacuerdo BERT vs clasificador** | Detecta casos donde robertuito dijo NEG pero el modelo clasificó media (y viceversa); distingue tres tipos de error: BERT correcto/TF-IDF incorrecto, BERT incorrecto/TF-IDF correcto, y ambos incorrectos |

**Heredadas de v2:**

| # | Función | Descripción |
|---|---|---|
| ④ | `validacion_cruzada_robusta()` | 10-fold estratificado con IC 95%, veredicto de estabilidad y detección de overfitting train-test |
| ⑤ | `entrenar_con_calibracion()` | Calibración isotónica de probabilidades con Brier Score antes/después y Log-Loss |
| ⑥ | `analizar_errores_criticos()` | FN/FP de alta confianza con diagnóstico por patrón léxico y recomendaciones accionables |
| ⑦ | `curva_aprendizaje()` | 8 puntos de curva train vs test con diagnóstico automático (overfitting / saturado / más datos) |
| ⑧ | `comparar_modelos()` | Benchmark simultáneo de CNB, LogisticRegression y LinearSVC en el mismo split |
| ⑨ | `correlacion_spearman()` + `prueba_chi2_categorias()` + `analizar_dimensiones()` + `generar_narrativa()` | Estadísticas de asociación, análisis por dimensión temática y reporte narrativo listo para copiar en tesis |

---

## 🔄 Pipeline de análisis

El script ejecuta **7 etapas secuenciales** tras la carga de artefactos. El orden importa: la validación cruzada debe ocurrir antes que la calibración, y la extracción de top-terms antes que la narrativa.

```
[1]  Detectar features BERT activas
     ├── Buscar "prob_neg" en cols_num del .pkl
     ├── Verificar que ≥ 10% de filas tienen valor ≠ 0
     └── Elegir estimador: LogisticRegression (BERT) | ComplementNB (texto puro)
 │
[2]  Validación cruzada robusta  (10-fold estratificado)
     ├── F1 weighted, F1 clase alta, AUC-ROC, Accuracy
     ├── IC 95%: x̄ ± z·(σ/√k)  con z=1.96
     ├── Detección de overfitting: gap train-test > 10%
     └── Veredicto: muy estable | estable | aceptable | inestable
 │
[3]  Calibración de probabilidades  (isotónico, cv=5)
     ├── Brier Score antes / después de calibrar
     ├── Mejora porcentual y Log-Loss
     └── Interpretación automática de la calidad de calibración
 │
[4]  Análisis de errores críticos  (umbral confianza ≥ 0.80)
     ├── FN confiados: quejas reales ignoradas por el modelo
     ├── FP confiados: falsas alarmas con alta certeza errónea
     ├── Diagnóstico léxico por patrón  (pago_privado, impacto_clínico, etc.)
     ├── Desacuerdo BERT vs clasificador  (nuevo v3)
     └── Recomendaciones accionables sobre el pipeline
 │
[5]  Curva de aprendizaje  (8 puntos, 10% → 100% del corpus)
     └── Diagnóstico: overfitting | saturado | más datos mejoran
 │
[6]  Comparación de modelos  (CNB vs LR vs LinearSVC)
     └── F1 weighted, F1 alta, AUC-ROC, Accuracy en el mismo split 80/20
 │
[7]  Top-features + Spearman + χ² + Dimensiones narrativas + Narrativa
     ├── Top-25 features predictivas por clase (texto + BERT)
     ├── Correlaciones Spearman con features numéricas del corpus
     ├── Prueba χ² por categoría temática con Odds Ratio
     ├── Dimensiones narrativas dominantes del corpus
     └── Reporte narrativo en texto plano (citable directamente en tesis)
```

---

## 📂 Estructura del proyecto

```
analizar_cnb_v3.py                              ← Script principal
X_features_YYYYMMDD_HHMMSS.npz                 ← Matriz sparse entrada (de vectorizar_v3.py)
y_labels_YYYYMMDD_HHMMSS.npy                   ← Vector de etiquetas binarias
vectorizador_YYYYMMDD_HHMMSS.pkl               ← Artefacto vectorizador + metadatos
dataset_etiquetado_YYYYMMDD_HHMMSS.json        ← Dataset opcional (de auto_etiquetar.py)
reporte_cnb_v3_YYYYMMDD_HHMMSS.json            ← Salida: métricas completas
resumen_narrativo_v2_YYYYMMDD_HHMMSS.txt       ← Salida: reporte citable en tesis
errores_criticos_YYYYMMDD_HHMMSS.json          ← Salida: FN/FP de alta confianza
```

---

## ⚙️ Requisitos

**Python 3.10+**

### Dependencias — instaladas automáticamente si no están presentes

```bash
pip install scikit-learn numpy scipy
```

> **¿Por qué `ComplementNB` y no `MultinomialNB` como estimador base?**
> CNB invierte la lógica de entrenamiento: en lugar de modelar P(feature|clase), modela P(feature|¬clase). Esto lo hace más robusto ante desequilibrio de clases — un problema frecuente en corpus de Reddit donde los comentarios de relevancia media superan en número a los de alta. Sin embargo, CNB asume independencia condicional entre features, lo cual falla cuando se añaden features BERT correlacionadas (`prob_neg ↔ sentimiento_num`). Por eso v3 cambia a `LogisticRegression` cuando detecta un corpus etiquetado con `auto_etiquetar.py`.

---

## 🚀 Uso

```bash
python3 analizar_cnb_v3.py
```

El script es completamente interactivo. Busca automáticamente los artefactos disponibles (`X_features*.npz`, `y_labels*.npy`, `vectorizador*.pkl`) en la misma carpeta y solicita selección si hay más de uno. El JSON del dataset es opcional pero necesario para el análisis de errores críticos con desacuerdo BERT.

### ⚠️ Advertencias de uso

| Situación | Consecuencia |
|---|---|
| Sin artefactos `.npz` / `.npy` / `.pkl` en la carpeta | ❌ El script termina — ejecutar `vectorizar_v3.py` primero |
| Pasar el JSON de `dataset_normalizado` (sin BERT) | ✅ Funciona; el análisis de desacuerdo BERT mostrará "sin datos BERT" en cada error |
| Pasar el JSON de `dataset_etiquetado` (con BERT) | ✅ Uso óptimo — activa el análisis de desacuerdo BERT v3 |
| Omitir el JSON completamente | ⚠️ Spearman, χ² y desacuerdo BERT quedan vacíos; métricas del modelo permanecen intactas |
| OOM durante validación cruzada | ⚠️ Reducir `N_FOLDS` a `5` o subir `TEST_SIZE` a `0.30` |

---

## 🔧 Configuración principal

| Constante | Valor | Descripción |
|---|---|---|
| `TOP_N_TERMS` | `25` | Número de features predictivas a extraer por clase |
| `N_TERMINOS_CITA` | `5` | Términos incluidos en el párrafo citable del reporte narrativo |
| `TEST_SIZE` | `0.20` | Fracción del corpus reservada para el split de test |
| `RANDOM_STATE` | `42` | Semilla de aleatoriedad para reproducibilidad |
| `N_FOLDS` | `10` | Número de folds en la validación cruzada estratificada |
| `IC_Z` | `1.96` | Factor Z para el intervalo de confianza del 95% |
| `METODO_CALIBRACION` | `"isotonic"` | Método de calibración: `"isotonic"` (flexible, ≥1000 muestras) o `"sigmoid"` (estable con menos datos) |
| `N_ERRORES_EXPORTAR` | `20` | Máximo de FN/FP de alta confianza incluidos en `errores_criticos_*.json` |
| `UMBRAL_CONFIANZA` | `0.80` | Probabilidad mínima para considerar un error como "confiado" y exportarlo |
| `CURVA_PUNTOS` | `8` | Puntos de la curva de aprendizaje (10% a 100% del corpus) |
| `FEATURES_BERT` | `["prob_neg", "prob_neu", "prob_pos", "sentimiento_num"]` | Nombres de features BERT a detectar en el artefacto `.pkl` |
| `UMBRAL_BERT_ACTIVO` | `0.10` | Fracción mínima de filas con `prob_neg ≠ 0` para activar LogisticRegression |

---

## 🧠 Arquitectura técnica avanzada

Esta sección documenta las decisiones de diseño de cada componente. Es la información clave para justificar las elecciones metodológicas en publicaciones académicas.

---

### 1. Detección automática de estimador (v3)

El selector automático resuelve un problema de compatibilidad entre versiones del pipeline: si el corpus pasó por `auto_etiquetar.py`, la matriz `X` contiene features BERT correlacionadas entre sí (`prob_neg ↔ sentimiento_num ↔ prob_pos` suman 1.0). CNB viola la independencia condicional en ese caso, produciendo probabilidades infladas.

#### Lógica de detección

```
1. Buscar "prob_neg" en cols_num del artefacto .pkl
2. Calcular el índice de la columna en la matriz X combinada:
       col_idx = n_features_tfidf + posición_de_prob_neg_en_cols_num
3. Contar fracción de filas con valor ≠ 0 en esa columna
4. Si fracción ≥ UMBRAL_BERT_ACTIVO (0.10) → BERT activo → usar LR
   Si fracción < 0.10  → BERT inactivo (dataset sin auto_etiquetar) → usar CNB
```

**¿Por qué LR y no un árbol o una SVM directamente?**

LR aprende coeficientes conjuntos para todas las features, manejando la correlación entre `prob_neg` y `sentimiento_num` sin penalizar artificialmente ninguna. LinearSVC también funciona bien, pero no produce probabilidades calibradas sin una segunda capa — y las probabilidades son necesarias para el análisis de errores críticos.

---

### 2. Validación cruzada 10-fold estratificada

#### ¿Por qué 10-fold y no 5-fold para una tesis?

| Aspecto | 5-fold | 10-fold |
|---|---|---|
| Muestras de test por fold (n≈2500) | ~125 | ~250 |
| Varianza del estimador | mayor | ~30% menor |
| Estabilidad del IC 95% | limitada | citeable en literatura académica |
| Tiempo de cómputo | menor | moderado |

El IC 95% se calcula con la distribución empírica de los 10 scores: `x̄ ± 1.96 · (σ / √k)`. Un F1 de `0.8412 ± 0.0183` significa que en el 95% de posibles muestras del mismo corpus, el modelo se comportaría dentro de ese rango — afirmación directamente citable.

#### Veredictos de estabilidad

| σ del F1 | Veredicto |
|---|---|
| `< 0.02` | muy estable |
| `< 0.04` | estable |
| `< 0.07` | aceptable |
| `≥ 0.07` | inestable — revisar datos |

---

### 3. Calibración isotónica de probabilidades

#### El problema de CNB sin calibrar

CNB asume independencia condicional entre features. En la práctica, un comentario con 3 términos de queja puede recibir `P(alta) = 0.997` cuando la probabilidad real estimada por calibración sería `~0.82`. Para afirmar en una tesis "este comentario tiene 85% de probabilidad de representar una queja relevante", ese 85% debe ser real.

#### Isotónico vs Sigmoid

| Método | Ventaja | Cuándo usar |
|---|---|---|
| `isotonic` | Función monótona no paramétrica, más flexible | ≥ 1000 muestras de calibración |
| `sigmoid` | Más estable con pocos datos | corpus pequeños, splits reducidos |

La calidad se mide con **Brier Score** (MSE entre probabilidad predicha y etiqueta real): `0.0 = perfecto`, `0.25 = modelo aleatorio`. Una mejora del 15% en Brier tras calibración es suficiente para justificar el paso en la metodología.

---

### 4. Análisis de errores críticos y desacuerdo BERT (v3)

El análisis exporta únicamente los errores "confiados": casos donde el modelo asignó probabilidad ≥ `UMBRAL_CONFIANZA` a la clase incorrecta. Son los más costosos para la investigación porque el modelo estaba seguro de una predicción falsa.

#### Tipos de desacuerdo BERT vs clasificador

| Situación | Diagnóstico | Acción sugerida |
|---|---|---|
| BERT NEG + modelo predijo media (FN) | BERT correcto, TF-IDF incorrecto | Ampliar `TERMINOS_DOMINIO` en el normalizador |
| BERT POS/NEU + modelo predijo alta (FP) | BERT correcto, TF-IDF incorrecto | Añadir bigramas neutrales como señal de media |
| BERT NEG + modelo predijo alta (FP) | Ambos sobreestimaron negatividad | Revisar la etiqueta ground truth del registro |
| BERT POS/NEU + modelo predijo media (FN) | Caso difícil: sarcasmo o queja implícita | Revisión manual; posible límite del corpus |

#### Patrones de diagnóstico léxico

| Patrón detectado | Tipo de error | Diagnóstico generado |
|---|---|---|
| `particular`, `de_mi_bolsillo`, `tuve_que_pagar` | FN | pago_privado no asociado con queja alta |
| `fallecer`, `morir`, `intubado`, `uci` | FN | impacto_clínico no detectado |
| Texto con `< 8` tokens | FN | señal léxica insuficiente — texto muy corto |
| `vacuna_info_neutral`, `vacuna_gratuita` | FP | token neutral de vacunación no filtrado |
| `imss`, `issste`, `hospital` sin indicador | FP | mención institucional sin queja real |

---

### 5. Curva de aprendizaje

Responde la pregunta clave para la tesis: **¿el modelo mejora si recolectamos más datos con el scraper?**

| Diagnóstico automático | Condición | Implicación |
|---|---|---|
| `"overfitting moderado"` | gap final `> 0.12` | Reducir vocabulario o aumentar alpha en CNB |
| `"más datos mejorarían"` | pendiente test `> 0.02` en segunda mitad | Continuar scraping — el modelo no está saturado |
| `"modelo saturado"` | gap final `< 0.03` | Mejorar features es más efectivo que añadir datos |
| `"equilibrado"` | ninguna condición anterior | Modelo estable, rendimiento confiable |

---

### 6. Comparación de modelos

Los tres modelos se evalúan en el mismo split 80/20 con las mismas métricas para garantizar comparabilidad:

| Modelo | Cuándo es mejor |
|---|---|
| `ComplementNB` | Texto puro, corpus desbalanceado, sin features BERT |
| `LogisticRegression` | Features híbridas (texto + BERT), corpus equilibrado |
| `LinearSVC` | Corpus grande (> 5000), alta dimensionalidad, sin necesidad de probabilidades |

El resultado se ordena por F1 weighted descendente y se reporta el modelo ganador en el reporte narrativo.

---

## 📦 Estructura de los archivos de salida

### `reporte_cnb_v3_*.json`

```json
{
  "meta": {
    "generado":    "2024-04-01T18:30:00",
    "version":     "3.0",
    "modelo":      "LR_calibrado",
    "bert_activo": true,
    "corpus":      "X_features_20240401_180000.npz"
  },
  "validacion_cruzada": {
    "n_folds":     10,
    "ic_nivel":    "95%",
    "f1_weighted": { "media": 0.8412, "std": 0.0183, "ic95_pm": 0.0113,
                     "ic95_lo": 0.8299, "ic95_hi": 0.8525,
                     "valores_por_fold": [...], "estable": true },
    "roc_auc":     { "media": 0.8731, ... },
    "overfitting": { "gap": 0.045, "detectado": false },
    "veredicto":   "estable (σ < 4%)"
  },
  "calibracion": {
    "sin_calibrar": { "brier_score": 0.14230 },
    "calibrado":    { "brier_score": 0.11847, "mejora_brier_pct": 16.7 },
    "interpretacion": "calibración isotónica exitosa"
  },
  "comparacion_modelos": [
    { "modelo": "LogisticRegression", "f1_weighted": 0.8412, "roc_auc": 0.8731 },
    { "modelo": "ComplementNB",       "f1_weighted": 0.8201, "roc_auc": 0.8520 },
    { "modelo": "LinearSVC",          "f1_weighted": 0.8389, "roc_auc": 0.8695 }
  ],
  "top_features_hibrido": [
    { "rank": 1, "nombre": "negligencia [texto]", "coef": 1.8432,
      "direccion": "→ alta",  "tipo": "texto" },
    { "rank": 2, "nombre": "prob_neg [BERT]",     "coef": 1.6201,
      "direccion": "→ alta",  "tipo": "BERT" },
    ...
  ],
  "correlaciones_spearman": [ ... ],
  "chi2_categorias":        [ ... ],
  "curva_aprendizaje":      { "puntos": [...], "diagnostico": "equilibrado" }
}
```

### `errores_criticos_*.json`

```json
{
  "umbral_confianza":   0.80,
  "total_fn_confiados": 12,
  "total_fp_confiados": 7,
  "fn_exportados": [
    {
      "texto_original":     "fui al particular porque en el imss no había...",
      "tokens_principales": ["particular", "imss", "no_hay"],
      "confianza_erronea":  0.914,
      "tipo_error":         "FN",
      "diagnostico":        "posible FN por pago_privado: el modelo no asocia...",
      "bert_sentimiento":   "NEG",
      "bert_prob_neg":      0.873,
      "desacuerdo_bert":    "BERT correcto (NEG p=0.87), TF-IDF incorrecto → señal léxica insuficiente"
    }
  ],
  "recomendaciones": [
    "ACCIÓN FN: Añadir 'particular', 'médico_privado', 'de_mi_bolsillo' a TERMINOS_DOMINIO..."
  ]
}
```

---

## 📈 Flujo de ejecución

```
pedir_archivos()  →  selección interactiva de X.npz / y.npy / .pkl / .json
      │
      ▼
cargar_todo()  →  load_npz + np.load + pickle.load + json.load
      │
      ▼
detectar_bert_activo()  →  busca prob_neg en cols_num del pkl
      │                     ≥10% filas activas → LR | < 10% → CNB
      ▼
train_test_split()  →  80% train / 20% test (estratificado, seed=42)
      │
      ▼
┌─── Etapas de análisis ─────────────────────────────────────────────────┐
│                                                                        │
│  ① validacion_cruzada_robusta()                                        │
│  ┌─ 10-fold estratificado  ──────────────────────────────────────────┐ │
│  │  F1 weighted / F1 alta / AUC-ROC / Accuracy por fold             │ │
│  │  IC 95%: x̄ ± 1.96·(σ/√10)  |  detección overfitting gap > 10% │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ② entrenar_con_calibracion()                                          │
│  ┌─ CalibratedClassifierCV(method=isotonic, cv=5)  ─────────────────┐ │
│  │  Brier Score antes/después  |  Log-Loss  |  F1 calibrado        │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ③ analizar_errores_criticos()                                         │
│  ┌─ FN/FP con confianza ≥ 0.80  ─────────────────────────────────────┐ │
│  │  diagnóstico léxico  |  desacuerdo BERT vs modelo  (v3)          │ │
│  │  patrones por tipo   |  recomendaciones accionables              │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  ④ curva_aprendizaje()  →  8 puntos, diagnóstico automático           │
│  ⑤ comparar_modelos()   →  CNB vs LR vs LinearSVC, mismo split        │
│  ⑥ extraer_top_terms() + top_features_hibrido()  →  texto + BERT      │
│  ⑦ correlacion_spearman() + prueba_chi2_categorias()                  │
│  ⑧ analizar_dimensiones() + generar_narrativa()                       │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
      │
      ▼
guardar_todo()  →  reporte_cnb_v3_*.json
                   resumen_narrativo_v2_*.txt
                   errores_criticos_*.json
```

---

## 🔬 Primeros pasos con los resultados (análisis con Python)

```python
import json
import pandas as pd

# ── Cargar reporte principal ───────────────────────────────────────────────────
with open("reporte_cnb_v3_YYYYMMDD_HHMMSS.json") as f:
    rep = json.load(f)

# 1. Métricas de validación cruzada con IC 95%
cv = rep["validacion_cruzada"]
print(f"F1 weighted: {cv['f1_weighted']['media']:.4f} ± {cv['f1_weighted']['ic95_pm']:.4f}")
print(f"AUC-ROC    : {cv['roc_auc']['media']:.4f} ± {cv['roc_auc']['ic95_pm']:.4f}")
print(f"Veredicto  : {cv['veredicto']}")

# 2. Impacto de la calibración
cal = rep["calibracion"]
print(f"Brier sin calibrar : {cal['sin_calibrar']['brier_score']:.5f}")
print(f"Brier calibrado    : {cal['calibrado']['brier_score']:.5f}  "
      f"(mejora: {cal['calibrado']['mejora_brier_pct']:+.1f}%)")

# 3. Comparación de modelos — ¿cuál ganó?
df_comp = pd.DataFrame(rep["comparacion_modelos"])
print(df_comp[["modelo","f1_weighted","f1_alta","roc_auc"]].to_string(index=False))

# 4. Top features híbridas — texto + BERT
df_feat = pd.DataFrame(rep["top_features_hibrido"])
print(df_feat[["rank","nombre","coef","tipo"]].head(10).to_string(index=False))

# 5. Correlaciones Spearman con features numéricas
df_cor = pd.DataFrame(rep["correlaciones_spearman"])
sig = df_cor[df_cor["significativa"]]
print(sig[["feature","rho","p_valor","magnitud"]].to_string(index=False))

# 6. χ² por categoría — ¿qué temas están más asociados con queja alta?
df_chi = pd.DataFrame(rep["chi2_categorias"])
print(df_chi[["categoria","odds_ratio","interpretacion"]].to_string(index=False))

# ── Cargar errores críticos ────────────────────────────────────────────────────
with open("errores_criticos_YYYYMMDD_HHMMSS.json") as f:
    err = json.load(f)

# 7. Diagnóstico de FN con desacuerdo BERT
df_fn = pd.DataFrame(err["fn_exportados"])
print(df_fn[["confianza_erronea","diagnostico","desacuerdo_bert"]].head(5))

# 8. Recomendaciones del analizador
for r in err["recomendaciones"]:
    print(f"→ {r}")
```

---

## 🔗 Posición en el pipeline completo

Este script es el **paso 3 del pipeline de modelado**, tras vectorización y (opcionalmente) etiquetado BERT:

```
[1] scraper_influenza_stealth_v3.py
     │  Recolecta posts y comentarios de Reddit MX
     ▼
dataset_influenza_crudo_final.json
     │
     ▼
[2] limpiar_dataset_v3.py / enriquecer_dataset_v3.py
     │  Filtra, normaliza y enriquece con IFB, polaridad localizada e impacto
     ▼
dataset_normalizado_YYYYMMDD.json
     │
     ▼
[2.5] auto_etiquetar.py  ← opcional pero recomendado
     │  Añade sentimiento BERT (robertuito), actualiza relevancia con lógica híbrida
     ▼
dataset_etiquetado_YYYYMMDD.json
     │
     ▼
[2.7] vectorizar_v3.py
     │  Genera matriz TF-IDF + features numéricas (incluye sentimiento_num si hay BERT)
     ▼
X_features.npz  |  y_labels.npy  |  vectorizador.pkl
     │
     ▼
[3] analizar_cnb_v3.py   ← aquí estamos
     │  Evalúa, compara, calibra y diagnostica el modelo de clasificación
     ▼
reporte_cnb_v3_*.json
resumen_narrativo_v2_*.txt
errores_criticos_*.json
     │
     ▼
Publicación académica / Tesis / Visualización / Exportación a SPSS o R
```

Los archivos de salida son directamente utilizables en:
- Tablas de resultados con IC 95% en metodología de tesis (sección validación cruzada)
- Comparativa de modelos como justificación de la elección del estimador final
- Análisis de errores como evidencia de las limitaciones del modelo y propuestas de mejora
- El `resumen_narrativo_*.txt` puede copiarse directamente como párrafo de resultados

---

## ⚠️ Consideraciones éticas y de validez del corpus

- El clasificador opera sobre opiniones ciudadanas reales. Las métricas de rendimiento deben interpretarse en el contexto del corpus de Reddit MX, no como generalizables a toda la población que experimenta problemas con el sistema de salud.
- El campo `relevancia` usada como ground truth fue asignada por una combinación de reglas léxicas y, opcionalmente, un modelo BERT. Para publicaciones académicas se recomienda validar una muestra estratificada con etiquetado humano independiente.
- Los Falsos Negativos (quejas reales ignoradas por el modelo) tienen mayor coste de investigación que los Falsos Positivos — una queja omitida sesga la prevalencia estimada de ineficiencia sistémica hacia abajo. Esto debe discutirse explícitamente en la sección de limitaciones de la tesis.
- El análisis de desacuerdo BERT vs clasificador es exploratorio y no debe interpretarse como una afirmación definitiva sobre cuál modelo es "correcto" — ambos pueden equivocarse en casos de sarcasmo, ironia o lenguaje eufemístico.
- Los registros analizados contienen datos de usuarios reales de Reddit. Cualquier publicación debe agregar o anonimizar los textos citados en el análisis de errores críticos para no exponer a usuarios individuales.

---

## 📝 Licencia

Uso académico / investigación. Consultar con el equipo responsable del proyecto antes de redistribuir o publicar datos recolectados.

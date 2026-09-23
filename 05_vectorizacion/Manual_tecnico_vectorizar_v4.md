# 📐 Vectorizador de Features — Salud Pública MX / Influenza `v2`

Cuarta etapa del pipeline de investigación (CRISP-DM Fase 4: Modelado). Transforma el corpus enriquecido en matrices numéricas listas para clasificadores de machine learning. Produce los artefactos `X`, `y` y el objeto vectorizador serializado que permite reproducir la misma transformación sobre texto nuevo durante inferencia.

---

## 📋 Descripción

El script ofrece cinco estrategias de vectorización seleccionables de forma interactiva, desde TF-IDF clásico hasta embeddings contextuales BERT, con una capa de features numéricas del pipeline anterior que siempre se concatena al bloque de texto. Incluye cuatro mejoras que atacan problemas específicos del corpus en español mexicano de salud pública: manejo de siglas institucionales, selección estadística de features, lexicon de quejas como feature garantizada y calibración adaptativa del umbral de frecuencia máxima.

---

## ✨ Mejoras implementadas

| # | Mejora | Problema que resuelve |
|---|---|---|
| ① | **BERT sentence embeddings** | TF-IDF no captura negación ni sarcasmo; BERT entiende contexto gramatical completo |
| ② | **Lexicon de quejas MX como feature numérica** | TF-IDF puede dar bajo peso a "viacrucis" si es poco frecuente; el lexicon garantiza su peso siempre |
| ③ | **SelectKBest χ²** | Elimina features de texto que son ruido estadístico respecto a la etiqueta de clase |
| ④A | **Normalización de siglas** | `IMSS`, `H1N1`, `CDMX` quedan fuera del token pattern estándar; se convierten a tokens en minúsculas antes de vectorizar |
| ④B | **`max_df` dinámico** | Un umbral fijo puede conservar términos hiper-frecuentes en el dominio (ej. "salud" en 95% de posts); se calcula el umbral óptimo por corpus |

---

## 🎯 Cinco estrategias de vectorización

```
[A] TF-IDF estándar          → rápido, interpretable, recomendado para baseline
[B] TF-IDF ponderado Reddit  → multiplica cada fila por peso_reddit del comentario
[C] CountVectorizer          → frecuencias brutas, línea base para Naive Bayes
[D] BERT embeddings          → 768 dims contextuales, captura negación y sarcasmo
[E] BERT + TF-IDF híbrido    → concatena D + A: máxima cobertura semántica
```

Todas las estrategias concatenan al final el bloque de **features numéricas** del corpus enriquecido (polaridad, IFB, impacto, palabras, peso Reddit y score de quejas).

---

## 📂 Artefactos de salida

```
vectorizar_v3.py                                ← Script principal
dataset_normalizado_*.json                      ← Entrada (salida de normalizar_nlp_v2.py)

X_features_v2_YYYYMMDD_HHMMSS.npz              ← Matriz X sparse (docs × features)
y_labels_v2_YYYYMMDD_HHMMSS.npy                ← Vector y de etiquetas (0/1)
vectorizador_v2_YYYYMMDD_HHMMSS.pkl            ← Artefactos serializados para inferencia
reporte_vectorizacion_v2_YYYYMMDD_HHMMSS.json  ← Reporte con top features y estadísticas
```

### Contenido del `.pkl`

El archivo `vectorizador_v2_*.pkl` es un diccionario con todo lo necesario para reproducir la transformación sobre datos nuevos sin re-entrenar:

| Clave | Tipo | Descripción |
|---|---|---|
| `vectorizador` | `TfidfVectorizer` / `CountVectorizer` / `None` | Objeto scikit-learn ajustado al corpus |
| `scaler` | `StandardScaler` | Escalador ajustado sobre las features numéricas |
| `selector` | `SelectKBest` / `None` | Selector χ² ajustado; `None` si no se aplicó |
| `estrategia` | `str` | `"A"` / `"B"` / `"C"` / `"D"` / `"E"` |
| `features_numericas` | `list[str]` | Nombres de columnas del bloque numérico |
| `mapa_etiqueta` | `dict` | `{"alta": 1, "media": 0}` |
| `token_pattern` | `str` | Patrón regex usado para tokenización |
| `siglas_normalizacion` | `dict` | Mapa de siglas → tokens usados en preprocesamiento |

---

## ⚙️ Requisitos

**Python 3.10+**

### Dependencias base (todas las estrategias)

```bash
pip install scikit-learn numpy scipy
```

### Dependencias BERT (solo estrategias D y E)

```bash
pip install sentence-transformers torch
```

> **Nota sobre BERT:** la primera ejecución descarga el modelo desde HuggingFace Hub (~400 MB). Las ejecuciones posteriores usan la caché local. El script **no instala estas dependencias automáticamente** porque la descarga puede tardar varios minutos dependiendo de la conexión. Instálalas manualmente antes de elegir la estrategia D o E.
>
> Las dependencias base (`scikit-learn`, `numpy`, `scipy`) sí se ofrecen para instalación automática al arrancar si no están presentes.

---

## 🚀 Uso

```bash
python3 vectorizar_v3.py
```

El script es completamente interactivo. Solicita en orden:

1. El número del archivo JSON a vectorizar (debe ser el output del normalizador NLP)
2. La estrategia de vectorización (A / B / C / D / E)

Al finalizar imprime el reporte completo en consola y guarda los cuatro artefactos en la misma carpeta.

### 🕐 Qué esperar durante la ejecución

| Lo que ves en consola | Causa | Duración aprox. |
|---|---|---|
| `max_df dinámico : 0.XXX (percentil 85 de N términos únicos)` | Cálculo adaptativo del umbral | Inmediato |
| `SelectKBest χ² : N → K features (M eliminadas)` | Selección estadística de features | Segundos |
| `Cargando modelo BERT: ...` | Primera carga o caché fría | 10–60 s |
| `(primera ejecución descarga ~400 MB — solo una vez)` | Descarga del modelo desde HuggingFace | 1–5 min (una sola vez) |
| Barra de progreso `Batches: N%` | Generación de embeddings BERT | 1–20 min según corpus y hardware |

> **GPU vs CPU:** BERT en CPU es significativamente más lento. Para corpus de más de 5,000 registros se recomienda GPU o usar la estrategia A/B como alternativa eficiente.

---

## 🔧 Configuración principal

| Constante | Valor por defecto | Descripción |
|---|---|---|
| `MAX_FEATURES` | `5_000` | Vocabulario máximo para TF-IDF / Count |
| `NGRAM_RANGE` | `(1, 2)` | Unigramas y bigramas |
| `MIN_DF` | `2` | Frecuencia mínima de documento para incluir un término |
| `MAX_DF_FALLBACK` | `0.85` | Umbral de frecuencia máxima si el cálculo dinámico falla |
| `SUBLINEAR_TF` | `True` | Aplica `log(tf)` en vez de `tf` crudo — reduce el peso de términos muy frecuentes |
| `APLICAR_SELECCION` | `True` | Activa/desactiva SelectKBest χ² |
| `K_FEATURES` | `3_000` | Features a conservar tras χ² |
| `MODELO_BERT` | `hiiamsid/sentence_similarity_spanish_es` | Modelo HuggingFace para embeddings |
| `BERT_BATCH` | `32` | Comentarios por lote en la generación de embeddings |
| `COLUMNA_ETIQUETA` | `"relevancia"` | Campo del JSON usado como variable objetivo |
| `MAPA_ETIQUETA` | `{"alta": 1, "media": 0}` | Codificación numérica de las clases |

### Modelos BERT alternativos

| Modelo | Dims | Velocidad | Especialización |
|---|---|---|---|
| `hiiamsid/sentence_similarity_spanish_es` *(default)* | 768 | Media | Similitud de oraciones en español |
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Rápida | Multilingüe 50+ idiomas, ideal para clustering |
| `dccuchile/bert-base-spanish-wwm-cased` (BETO) | 768 | Lenta | Español general, ampliamente validado |
| `PlanTL-GOB-ES/roberta-base-bne` | 768 | Lenta | RoBERTa español, estado del arte |

---

## 🧠 Arquitectura técnica avanzada

---

### 1. El problema de la negación en TF-IDF y por qué BERT lo resuelve

TF-IDF trata cada texto como una **bolsa de palabras**: el orden y el contexto gramatical desaparecen. Esto produce un punto ciego crítico para este corpus:

```
Texto A: "No hay medicinas en el IMSS"
Texto B: "Medicinas hay en el IMSS, no faltan"
```

Ambos textos comparten exactamente los mismos tokens (`no`, `hay`, `medicinas`, `imss`, `faltan`). Sus vectores TF-IDF son prácticamente idénticos aunque expresen lo opuesto. Un clasificador entrenado con TF-IDF tendrá dificultades para distinguirlos.

BERT (Bidirectional Encoder Representations from Transformers) procesa cada token en el contexto de todos los demás tokens del texto. En el Texto A, `"hay"` asiste al token `"No"` que lo precede y actualiza su representación para reflejar la negación. Los vectores resultantes son distintos y correctamente orientados en el espacio semántico.

**Por eso BERT opera sobre `comentario` (texto con stopwords), no sobre `texto_normalizado`:** las stopwords (`no`, `hay`, `me`, `a`) son el andamiaje gramatical que BERT necesita para entender la negación, la modalidad y el sarcasmo. Eliminarlas antes de BERT destruye exactamente la información que lo hace valioso.

---

### 2. Lexicon de quejas como feature numérica garantizada

El lexicon `LEXICON_QUEJAS_SALUD` (~55 términos con pesos 1.5–3.0) se suma como columna numérica **adicional** a las features de texto, no como sustituto. Su valor técnico se entiende con este escenario:

Un corpus de 4,000 comentarios podría tener solo 12 menciones de `"viacrucis"`. TF-IDF calculará:

```
IDF("viacrucis") = log(4000 / (12 + 1)) ≈ 5.73
```

Es un IDF alto (buena señal discriminativa), pero si ese término no aparece en muchos documentos de entrenamiento, el modelo puede no aprender su importancia con solidez estadística. En un corpus pequeño, puede quedar fuera de los `MAX_FEATURES = 5000` más frecuentes.

El lexicon garantiza que `"viacrucis"` **siempre aporte peso 3.0** al vector de ese documento, independientemente de su frecuencia en el corpus. Es una forma de inyectar conocimiento de dominio que el TF-IDF estadístico no puede inferir solo.

La función `calcular_score_queja()` opera sobre `texto_normalizado` (ya lematizado y con n-gramas como tokens), lo que maximiza el matching con los términos del lexicon.

---

### 3. Normalización de siglas — Mejora ④A

#### El problema

El token pattern estándar de scikit-learn (`r"(?u)\b\w\w+\b"`) convierte todo a minúsculas después de tokenizar, pero el problema ocurre antes: `IMSS`, `H1N1` y `CDMX` contienen mayúsculas o dígitos que el pattern `r"[a-záéíóúüñ_]{2,}"` (solo letras minúsculas y guión bajo) **excluye por completo**. Resultado: las instituciones más importantes del corpus desaparecen del vocabulario TF-IDF.

#### La solución en dos pasos

**Paso 1 — Pre-normalización:** antes de pasar el corpus al vectorizador, cada sigla conocida se reemplaza por un token en minúsculas con semántica descriptiva:

```
"Fui al IMSS y no había nada"
    → "Fui al entidad_salud_imss y no había nada"
```

El guión bajo une las palabras del token compuesto para que el vectorizador lo trate como una sola feature `"entidad_salud_imss"` en vez de tres palabras separadas.

**Paso 2 — Token pattern ampliado:** se configura el vectorizador con:
```python
TOKEN_PATTERN = r"[a-záéíóúüñ][a-záéíóúüñ0-9_]{1,}"
```
Este patrón acepta letras, dígitos y guión bajo, lo que permite que tokens como `virus_h1n1` o `entidad_salud_imss` sean reconocidos correctamente.

**Beneficio adicional:** el modelo aprende un concepto único (`entidad_salud_imss`) en vez de variantes ortográficas separadas (`IMSS`, `imss`, `Imss`, `el IMSS`), lo que mejora la generalización.

**Importante:** la normalización de siglas se aplica al corpus para TF-IDF pero **no** al corpus para BERT. BERT fue entrenado con texto natural que incluye siglas en mayúsculas y sabe interpretarlas correctamente en su contexto.

---

### 4. `max_df` dinámico — Mejora ④B

#### El problema

`max_df=0.90` fijo significa "excluir términos que aparecen en más del 90% de los documentos". En un corpus genérico es razonable. En un corpus de dominio específico como éste, palabras como `"salud"`, `"médico"` o `"imss"` pueden aparecer en el 95% de los comentarios siendo al mismo tiempo **altamente discriminativas** entre clases. Un umbral fijo eliminaría señales valiosas.

#### La metodología

```python
# Para cada término único del corpus, contar en cuántos documentos aparece
frecuencias_relativas = [n_docs_con_termino / n_docs_total
                         for termino in vocabulario]

# Tomar el percentil 85 de esa distribución
umbral_raw = np.percentile(frecuencias_relativas, 85)

# Aplicar piso y techo para evitar extremos
max_df = max(0.60, min(0.95, umbral_raw))
```

El percentil 85 identifica la "rodilla" de la curva de frecuencias: el punto donde los términos dejan de aportar discriminación y empiezan a ser ruido omnipresente. El piso en 0.60 evita que corpus muy pequeños eliminen términos con una frecuencia razonable; el techo en 0.95 evita que términos extremadamente comunes contaminen el espacio de features.

---

### 5. Selección de features con χ² — Mejora ③

Tras vectorizar, la matriz TF-IDF tiene hasta `MAX_FEATURES = 5,000` columnas. No todas son igualmente útiles para la tarea de clasificación: muchos términos tienen distribución similar entre clases y solo añaden ruido.

`SelectKBest` con la prueba χ² (chi-cuadrado) mide la **dependencia estadística** entre cada feature (término) y la variable objetivo (relevancia alta vs. media):

$$\chi^2 = \sum_{i,j} \frac{(O_{ij} - E_{ij})^2}{E_{ij}}$$

Un χ² alto indica que la presencia/ausencia del término se correlaciona con la clase, es decir, que es una feature discriminativa. Un χ² bajo indica distribución independiente de la clase: ruido que se puede eliminar.

El script conserva las `K_FEATURES = 3,000` features con mayor χ², reduciendo la dimensionalidad de 5,000 a 3,000 sin sacrificar poder discriminativo.

**Restricción importante:** χ² requiere `X ≥ 0`. TF-IDF y CountVectorizer lo garantizan porque sus valores son frecuencias o pesos siempre no-negativos. Por esto SelectKBest se aplica **solo sobre el bloque de texto**, nunca sobre el bloque de features numéricas (que incluye el IFB y la polaridad, valores ya validados por construcción) ni sobre embeddings BERT (que pueden ser negativos).

---

### 6. Ensamblado de la matriz final

Todas las estrategias producen el mismo formato de salida: una **matriz sparse CSR** que concatena horizontalmente los bloques disponibles:

```
                    ┌──────────────────────────────────────────────────────────┐
                    │              MATRIZ X FINAL (docs × features)            │
                    ├──────────────────┬──────────────┬────────────────────────┤
Estrategia A/B/C →  │ TF-IDF / Count   │              │                        │
                    │ (hasta 3,000 f.) │              │   Features numéricas   │
Estrategia D →      │                  │ BERT (768 f.)│   (6-7 columnas)       │
                    │                  │              │   polaridad_score       │
Estrategia E →      │ TF-IDF (3,000 f.)│ BERT (768 f.)│   ifb_score             │
                    │                  │              │   impacto_escala        │
                    │                  │              │   num_palabras          │
                    │                  │              │   peso_reddit           │
                    │                  │              │   score_queja_salud ✅  │
                    └──────────────────┴──────────────┴────────────────────────┘
```

Los bloques densos (BERT, numéricas) se convierten a `csr_matrix` antes del `hstack` para que el resultado final sea uniforme y compatible con todos los clasificadores de scikit-learn. La sparsity resultante varía: estrategia A con K=3,000 produce ~97% sparsity; estrategia E con BERT concatenado baja a ~60-70%.

---

## 📊 Reporte de vectorización

Al finalizar, el script guarda un JSON con:

```json
{
  "meta_vectorizacion": {
    "generado":          "2025-04-01T17:00:00",
    "version_script":    "2.0",
    "estrategia":        "A",
    "dimensiones_X":     [4790, 3007],
    "clases_y":          {"alta": 3089, "media": 1701},
    "mejoras_activas": {
      "bert_embeddings":      false,
      "lexicon_quejas_salud": true,
      "select_k_best_chi2":   true,
      "siglas_normalizadas":  true,
      "max_df_dinamico":      true
    },
    "config": {
      "max_features":    5000,
      "ngram_range":     [1, 2],
      "min_df":          2,
      "sublinear_tf":    true,
      "k_features_chi2": 3000,
      "bert_modelo":     null,
      "features_num":    ["polaridad_score","ifb_score","impacto_escala",
                          "num_palabras","peso_reddit","score_queja_salud"]
    }
  },
  "top_features_por_clase": {
    "alta":  [{"termino": "desabasto_medicamento", "peso_medio": 0.04821}, ...],
    "media": [{"termino": "influenza_gripe",       "peso_medio": 0.03104}, ...]
  },
  "score_queja_salud_por_clase": {
    "alta":  {"media": 8.43, "mediana": 7.5,  "max": 28.1, "pct_cero": 4.2},
    "media": {"media": 1.82, "mediana": 0.0,  "max": 12.4, "pct_cero": 61.3}
  }
}
```

El campo `score_queja_salud_por_clase` es una validación inmediata de la calidad del corpus: si el lexicon está bien calibrado, la clase `alta` debe tener un score_queja medio significativamente mayor que `media`, y un `pct_cero` mucho menor.

---

## 📈 Flujo de ejecución

```
pedir_archivo()  →  selección interactiva del JSON normalizado
      │
      ▼
elegir_estrategia()  →  A | B | C | D | E
      │
      ▼
cargar_registros()  →  extrae "datos" y "meta_normalizacion"
      │
      ▼
extraer_corpus()
  ├── corpus_norm   ← texto_normalizado  (para TF-IDF)
  ├── corpus_orig   ← comentario limpio  (para BERT, con stopwords)
  ├── etiquetas     ← MAPA_ETIQUETA[relevancia]  →  {alta:1, media:0}
  ├── pesos         ← peso_reddit por registro
  └── feat_num_raw  ← {polaridad_score, ifb_score, ...} + score_queja_salud ②
      │
      ▼
calcular_max_df_dinamico()  ④B
  └── percentil 85 de frecuencias relativas → umbral entre [0.60, 0.95]
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BLOQUE DE TEXTO (estrategias A / B / C / E)                        │
│                                                                     │
│  normalizar_siglas()  ④A                                            │
│    IMSS → entidad_salud_imss | H1N1 → virus_h1n1 | ...             │
│                │                                                    │
│  construir_vectorizador(estrategia, max_df)                         │
│    A/B → TfidfVectorizer(sublinear_tf=True, norm=l2)               │
│    C   → CountVectorizer                                            │
│                │                                                    │
│  fit_transform(corpus_norm_con_siglas_normalizadas)                 │
│                │                                                    │
│  [estrategia B] diags(pesos).dot(X_texto)  ← pondera por reddit    │
│                │                                                    │
│  aplicar_seleccion_features(X_texto, y, K=3000)  ③                 │
│    χ²(feature, clase) → conserva top K features                    │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BLOQUE BERT (estrategias D / E)                                    │
│                                                                     │
│  generar_embeddings_bert(corpus_orig)  ①                            │
│    SentenceTransformer(MODELO_BERT)                                 │
│    .encode(batch_size=32, normalize_embeddings=True)                │
│    → array (n_docs × 768)  normalize L2 → cosine ≡ dot product     │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  BLOQUE NUMÉRICO (todas las estrategias)                            │
│                                                                     │
│  escalar_features_numericas()                                       │
│    cols = [polaridad_score, ifb_score, impacto_escala,              │
│             num_palabras, peso_reddit, score_queja_salud]           │
│    StandardScaler().fit_transform(mat)  → media=0, σ=1             │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
scipy.sparse.hstack([X_texto, X_bert, X_num], format="csr")
      │
      ▼
X_final: shape (n_docs × n_features_total)
      │
      ▼
guardar_artefactos()
  ├── X_features_v2_*.npz      ← save_npz(X_final)
  ├── y_labels_v2_*.npy        ← np.save(y, dtype=int8)
  ├── vectorizador_v2_*.pkl    ← pickle {vec, scaler, selector, ...}
  └── reporte_vectorizacion_*.json
```

---

## 🔬 Cómo cargar y usar los artefactos

```python
from scipy.sparse import load_npz
import numpy as np
import pickle

# ── Cargar artefactos ─────────────────────────────────────────────────────────
X = load_npz("X_features_v2_YYYYMMDD_HHMMSS.npz")
y = np.load("y_labels_v2_YYYYMMDD_HHMMSS.npy")

with open("vectorizador_v2_YYYYMMDD_HHMMSS.pkl", "rb") as f:
    art = pickle.load(f)

print(f"Dimensiones: {X.shape}")   # (n_docs, n_features)
print(f"Clases: {np.unique(y, return_counts=True)}")

# ── División estratificada ────────────────────────────────────────────────────
from sklearn.model_selection import train_test_split

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Clasificadores recomendados según estrategia ──────────────────────────────
# Estrategias A / B  → modelos lineales (rápidos, interpretables)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
clf.fit(X_tr, y_tr)
print(f"Accuracy: {clf.score(X_te, y_te):.3f}")

# Estrategia C       → Naive Bayes (solo con CountVectorizer, X ≥ 0)
from sklearn.naive_bayes import ComplementNB
clf_nb = ComplementNB()
clf_nb.fit(X_tr, y_tr)

# Estrategias D / E  → modelos no-lineales para aprovechar el espacio BERT
from sklearn.neural_network import MLPClassifier
clf_mlp = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=200,
                        random_state=42)
clf_mlp.fit(X_tr, y_tr)

# ── Inferencia sobre texto nuevo ──────────────────────────────────────────────
# Para reproducir exactamente la misma transformación:
from vectorizar_v3 import normalizar_siglas, calcular_score_queja

texto_nuevo   = "el imss no tenía vacuna y tuve que ir al particular"
texto_siglas  = normalizar_siglas(texto_nuevo)      # ④A aplicada
X_texto_nuevo = art["vectorizador"].transform([texto_siglas])

# Aplicar selector si existe
if art["selector"]:
    X_texto_nuevo = art["selector"].transform(X_texto_nuevo)

# Features numéricas (ejemplo con valores conocidos o estimados)
import numpy as np
nums = np.array([[2.5, 4.0, 2.0, 12.0, 1.0,         # polaridad, ifb, impacto, palabras, peso
                  calcular_score_queja(texto_nuevo)]], # ② score_queja
                dtype=np.float32)
X_num_nuevo = art["scaler"].transform(nums)

# Combinar y predecir
from scipy.sparse import hstack, csr_matrix
X_nuevo = hstack([X_texto_nuevo, csr_matrix(X_num_nuevo)], format="csr")
pred    = clf.predict(X_nuevo)
print("Alta relevancia" if pred[0] == 1 else "Media relevancia")
```

---

## 🔗 Posición en el pipeline completo

```
[1] scraper_influenza_stealth_v3.py
     │  Recolecta posts y comentarios de Reddit MX
     ▼
dataset_influenza_crudo_final.json
     │
     ▼
[2] limpiar_dataset_v3.py
     │  Filtra, normaliza y clasifica semánticamente
     ▼
dataset_limpio_YYYYMMDD.json
     │
     ▼
[3] enriquecer_dataset_v3.py
     │  Añade IFB, polaridad localizada, nivel de impacto
     ▼
dataset_enriquecido_YYYYMMDD.json
     │
     ▼
[4a] normalizar_nlp_v2.py          ← paso previo implícito
     │  Lematiza, elimina stopwords, genera texto_normalizado y peso_reddit
     ▼
dataset_normalizado_YYYYMMDD.json
     │
     ▼
[4b] vectorizar_v3.py              ← aquí estamos
     │  Construye X, y y artefactos de transformación
     ▼
X_features.npz  |  y_labels.npy  |  vectorizador.pkl
     │
     ▼
[5] Clasificación supervisada
    LogisticRegression / LinearSVC / MLPClassifier / XGBoost
    → modelo entrenado para inferencia en producción
```

---

## ⚠️ Consideraciones metodológicas

- La estrategia **B (TF-IDF ponderado)** usa `peso_reddit` del normalizador. Si ese campo no existe en el JSON de entrada, todos los pesos son `1.0` y el resultado es idéntico a la estrategia A.
- El campo `texto_normalizado` del normalizador es la entrada principal. Si un registro no tiene ese campo, se descarta con advertencia en consola.
- La **sparsity de la matriz X** es un indicador de salud: valores por debajo del 85% en estrategias A/B/C pueden indicar que `MAX_FEATURES` es demasiado grande para el corpus disponible, lo que introduce ruido.
- Los artefactos `.pkl`, `.npz` y `.npy` deben mantenerse juntos y con los mismos timestamps para garantizar coherencia entre `X`, `y` y el vectorizador durante el entrenamiento y la inferencia.
- Para publicaciones académicas, reportar la estrategia, `MAX_FEATURES`, `NGRAM_RANGE`, `K_FEATURES` y el modelo BERT usado es suficiente para reproducibilidad.

---

## 📝 Licencia

Uso académico / investigación. Consultar con el equipo responsable del proyecto antes de redistribuir o publicar datos recolectados.

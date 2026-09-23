# 🔤 Normalizador NLP — Salud Pública MX / Influenza `v2`

Cuarta etapa del pipeline de investigación, primera del bloque de modelado (CRISP-DM Fase 3→4). Recibe el dataset enriquecido y produce el `texto_normalizado` que el vectorizador consume directamente: texto lematizado, con stopwords eliminadas selectivamente, n-gramas del dominio consolidados como tokens únicos, entidades nombradas extraídas y un peso de ponderación Reddit calculado por registro.

---

## 📋 Descripción

El script usa **spaCy** (`es_core_news_sm`) para tokenizar y lematizar cada comentario, pero no delega en spaCy la decisión final sobre qué conservar y qué eliminar. Esa lógica está controlada por cuatro capas de reglas del dominio que se aplican en orden estricto: protección de negaciones y causales, consolidación de n-gramas antes de la tokenización, tabla de excepciones de lematización, y búsqueda de respaldo para entidades nombradas que el modelo NER puede perder en texto coloquial. Todo el procesamiento ocurre en lotes via `nlp.pipe()` para rendimiento.

---

## ✨ Mejoras implementadas en v2

| # | Mejora | Problema que resuelve |
|---|---|---|
| ① | **Stopwords refinadas — negaciones y causales siempre protegidas** | `"no hay medicina"` sin protección → `"medicina"`, invirtiendo el sentido semántico |
| ② | **N-gramas de dominio predefinidos** | `"no hay sistema"` tokenizado como tres palabras pierde la unidad semántica; como `no_hay_sistema` el vectorizador lo trata como concepto único |
| ③ | **Lematización con preservación de sentimiento** | spaCy puede neutralizar `"pésimo"` → `"malo"` o `"falleció"` → forma genérica, perdiendo intensidad emocional y temporalidad clínica |
| ④ | **NER — Extracción de Entidades Nombradas** | Permite filtrar y comparar quejas por institución (IMSS vs ISSSTE) o región (CDMX vs Monterrey) sin trabajo manual posterior |
| ⑤ | **Peso Reddit por registro** | Los comentarios muy votados expresan opinión validada por la comunidad; el peso permite dar más influencia a esas filas en TF-IDF |
| ⑥ | **Robustez: carga segura + `nlp.pipe()` en lotes** | Un solo error de carga del modelo no aborta el pipeline; `nlp.pipe()` es 3–5× más rápido que llamar `nlp(texto)` en bucle |

**Heredadas de v1:**

| # | Función | Descripción |
|---|---|---|
| ⑦ | Tokenización + lematización base | Pipeline spaCy con morphologizer, lemmatizer, ner |
| ⑧ | Conservación de términos de dominio | Lista `TERMINOS_DOMINIO` protege vocabulario clave aunque spaCy lo marque como stopword |
| ⑨ | Estadísticas de vocabulario y top términos | Vocabulario único, top-30 términos, top-15 n-gramas, top-10 instituciones NER |
| ⑩ | JSON listo para TF-IDF / BoW / embeddings | `texto_normalizado` con n-gramas como tokens únicos, compatible con FastText y sentence-transformers |

---

## 🔄 Pipeline de normalización por registro

El procesamiento de cada comentario sigue **8 pasos secuenciales** dentro de `procesar_batch()`. El orden es crítico: los n-gramas deben consolidarse antes de que spaCy tokenice, y las excepciones de lema deben aplicarse antes de la decisión de filtrado.

```
[1]  Conversión :emoji: → texto plano
      ":cara_enojada:" → "cara enojada"
 │
[2]  Aplicar n-gramas del dominio (ANTES de spaCy)
      "no hay medicamento" → "no_hay_medicamento"
      "seguro social"      → "seguro_social"
 │
[3]  spaCy tokeniza y analiza morfológicamente
      (tokenizer + morphologizer + lemmatizer + ner activos)
 │
[4]  Por cada token — filtros de ruido:
      ├── ¿es puntuación o espacio?  → descartar
      ├── ¿es número?               → descartar
      └── ¿longitud < MIN_LEN_TOKEN (2)?  → descartar
 │
[5]  Resolver lema con tabla de excepciones [MEJORA 3]
      "pésimo" → "pésimo"   (NO → "malo")
      "falleció" → "fallecer"  (preserva, usa infinitivo)
      "no_hay_sistema" → "no_hay_sistema"  (conservar intacto)
 │
[6]  Decisión de filtrado stopwords [MEJORA 1]
      ├── ¿lema está en TERMINOS_DOMINIO?  → conservar siempre
      ├── ¿lema está en SIEMPRE_CONSERVAR (negaciones/causales)?  → conservar siempre
      ├── ¿es stopword spaCy / STOPWORDS_EXTRA?
      │     Y ¿NO está en SIEMPRE_CONSERVAR?  → descartar
      └── ¿contiene solo letras o guión bajo?  → conservar
 │
[7]  Extraer entidades NER del doc [MEJORA 4]
      ORG → instituciones normalizadas  |  LOC/GPE → lugares
      + búsqueda manual de respaldo con INSTITUCION_CANONICA
 │
[8]  Calcular peso Reddit [MEJORA 5]
      log1p(score_comentario)×0.7 + log1p(score_post)×0.3
      × penalización por profundidad
```

---

## 📂 Estructura del proyecto

```
normalizar_nlp_v3.py                           ← Script principal
dataset_enriquecido_YYYYMMDD.json              ← Entrada (output del enriquecedor)
dataset_normalizado_YYYYMMDD_HHMMSS.json       ← Salida: corpus listo para vectorizar
```

---

## ⚙️ Requisitos

**Python 3.10+**

### Dependencia principal

```bash
pip install spacy
python -m spacy download es_core_news_sm
```

> El script detecta e instala `spaCy` automáticamente al arrancar si no está disponible, y descarga `es_core_news_sm` (~12 MB) si el modelo no está en el entorno. Si la descarga automática falla por permisos, el script imprime el comando manual exacto y termina limpiamente en vez de fallar con un traceback.

### Modelos alternativos de spaCy (mayor precisión, mayor tamaño)

| Modelo | Tamaño | Vectores GloVe | Mejor para |
|---|---|---|---|
| `es_core_news_sm` *(default)* | ~12 MB | No | Velocidad en corpus grandes |
| `es_core_news_md` | ~43 MB | Sí (20k) | Balance velocidad/calidad |
| `es_core_news_lg` | ~566 MB | Sí (685k) | Máxima calidad NER y lematización |

Para cambiar el modelo: editar `MODELO_SPACY = "es_core_news_lg"` en la configuración.

---

## 🚀 Uso

```bash
python3 normalizar_nlp_v3.py
```

El script es completamente interactivo. Solicita en orden:

1. El número del JSON a normalizar (debe ser el output del enriquecedor)
2. El nombre/ruta del archivo de salida (o Enter para usar el nombre sugerido)

El progreso se muestra con un contador de filas actualizado en la misma línea (`→ 1280/4790 (26%)...`) para no saturar la terminal.

### 🕐 Qué esperar durante la ejecución

| Lo que ves en consola | Causa | Duración aprox. |
|---|---|---|
| `Instalando spaCy...` | Primera ejecución sin spaCy | 30–60 s |
| `Descargando modelo es_core_news_sm (~12 MB)...` | Primera ejecución sin el modelo | 10–30 s |
| `ℹ️ Pipes activos: tokenizer + morphologizer, lemmatizer, ner` | Confirmación de pipeline cargado | Inmediato |
| `→ XXXX/YYYY (ZZ%)...` actualizándose | Procesamiento en lotes de 64 | 1–10 min según corpus |

> **Performance:** `nlp.pipe()` con `BATCH_SIZE=64` es 3–5× más rápido que procesar texto a texto en un bucle. Para un corpus de 4,800 registros se esperan 2–4 minutos en CPU. Con `es_core_news_lg` el tiempo se multiplica por ~4.

---

## 🔧 Configuración principal

| Constante | Valor por defecto | Descripción |
|---|---|---|
| `MIN_LEN_TOKEN` | `2` | Longitud mínima de token para conservar (filtra letras sueltas) |
| `BATCH_SIZE` | `64` | Registros por lote en `nlp.pipe()` |
| `MODELO_SPACY` | `es_core_news_sm` | Modelo de spaCy a cargar |
| `NEGACIONES_PROTEGIDAS` | 13 términos | Nunca se eliminan por filtro stopword |
| `CONECTORES_CAUSALES` | ~18 términos | Nunca se eliminan (detectan causalidad ineficiencia → daño) |
| `TERMINOS_DOMINIO` | ~50 términos | Vocabulario clave del dominio, protegido aunque spaCy lo marque stopword |
| `STOPWORDS_EXTRA` | ~25 términos | Stopwords específicas de Reddit MX sin valor temático ni de sentimiento |
| `NGRAMAS_DOMINIO` | 33 patrones | Bigramas y trigramas del dominio consolidados pre-tokenización |
| `EXCEPCIONES_LEMA` | ~45 entradas | Tabla de lemas correctos para términos que spaCy neutralizaría |
| `INSTITUCION_CANONICA` | ~16 entradas | Mapa de variantes a nombre canónico para normalización NER |

---

## 🧠 Arquitectura técnica avanzada

---

### 1. Por qué la negación es el punto crítico del preprocesamiento NLP

En análisis de sentimientos estándar, las stopwords se eliminan por completo porque palabras como "de", "la", "que" no aportan información temática. Sin embargo, en el corpus de quejas de salud pública mexicana, **la negación es la señal semántica más importante**:

```
Sin protección de negaciones:
  "no hay medicamento"    → tokens: ["medicamento"]
  "hay medicamento"       → tokens: ["medicamento"]
  → vectores idénticos → el modelo no puede distinguirlos

Con NEGACIONES_PROTEGIDAS:
  "no hay medicamento"    → tokens: ["no_hay_medicamento"]  (n-grama)
  "hay medicamento"       → tokens: ["medicamento"]
  → vectores distintos → el modelo aprende la diferencia
```

Lo mismo aplica a los conectores causales: `"empeoró porque no atendieron"` sin `"porque"` se convierte en `"empeorar atender"`, perdiendo la relación causal entre la ineficiencia del sistema y el daño clínico. Esa relación es precisamente el objeto de investigación.

**Lista de términos siempre conservados:**

- *Negaciones:* `no`, `ni`, `tampoco`, `sin`, `nunca`, `jamás`, `nada`, `nadie`, `ningún`, `ninguno`, `ninguna`
- *Causales:* `porque`, `pues`, `ya que`, `dado que`, `debido`, `por eso`, `en consecuencia`, `pero`, `aunque`, `sin embargo`, `a pesar`

---

### 2. N-gramas del dominio: por qué predefinidos y no automáticos

La extracción automática de n-gramas (PMI, frecuencia mínima) es el enfoque estándar en NLP general. Para este corpus se eligió un diccionario predefinido por tres razones:

**Razón 1 — Garantía semántica:** los algoritmos automáticos pueden capturar `"me dijo"` o `"fue al"` como bigramas frecuentes que no aportan nada. Los 33 n-gramas definidos son conceptos unitarios verificados en el dominio.

**Razón 2 — Preservación de negaciones compuestas:** `"no hay sistema"`, `"no me atendieron"`, `"no hay vacuna"` son frases de queja completas. Divididas en tokens, el modelo puede no asociar el `"no"` con el concepto que niega. Como token único `no_hay_sistema`, el vectorizador les asigna un peso IDF propio.

**Razón 3 — Resistencia a corpus pequeños:** la extracción automática necesita corpus grandes para estimar PMI de forma estable. Con 4,000–5,000 registros, un bigrama como `"vuelva mañana"` puede no superar el umbral de frecuencia mínima aunque sea semánticamente crítico.

#### Implementación: los trigramas se aplican primero

```python
NGRAMAS_DOMINIO = [
    # Trigramas primero (orden crítico)
    ("no hay sistema",       "no_hay_sistema"),
    ("vuelva usted mañana",  "vuelva_manana"),
    # Bigramas después
    ("vuelva mañana",        "vuelva_manana"),
    ("no hay",               "no_hay"),        # bigrama residual
]
```

Si los bigramas se aplicaran primero, `"no hay sistema"` se convertiría en `"no_hay sistema"` y el trigrama nunca podría matchear. El orden garantiza que las frases más largas se consolidan antes que sus sub-secuencias.

Los patrones se compilan una sola vez con `re.compile(re.escape(original), re.IGNORECASE)` antes del bucle principal, evitando recompilar miles de veces.

---

### 3. Tabla de excepciones de lematización: cuándo spaCy se equivoca

spaCy lematiza con reglas morfológicas del español general. En el contexto de quejas de salud pública, produce errores sistemáticos en dos categorías:

#### Adjetivos extremos neutralizados

spaCy puede mapear adjetivos superlativos o intensos a su raíz morfológica más neutra. En análisis de sentimientos, esto destruye la señal:

| Forma original | Lema spaCy | Lema correcto (EXCEPCIONES_LEMA) | Por qué importa |
|---|---|---|---|
| `"pésimo"` / `"pésima"` | `"malo"` | `"pésimo"` | El TF-IDF debe distinguir "malo" de "pésimo" como niveles distintos de negatividad |
| `"terrible"` | `"mal"` | `"terrible"` | Misma razón |
| `"negligente"` | `"descuidado"` | `"negligente"` | Término legal/médico específico del dominio |
| `"incompetente"` | `"incapaz"` | `"incompetente"` | Semántica institucional distinta |

#### Verbos de impacto clínico con pérdida de temporalidad

La lematización al infinitivo puede hacer que el modelo no aprenda el peso específico de los eventos pasados:

```
"falleció" → spaCy → "fallecer"  (aceptable, mantenemos infinitivo)
"murió"    → spaCy → "morir"     (aceptable)
"empeoró"  → spaCy → "empeorar"  (aceptable)
```

Los verbos clínicos se mapean al infinitivo, no a formas conjugadas — esto es correcto para TF-IDF porque uniformiza las variantes conjugadas (`"murió"`, `"murieron"`, `"se murió"`) bajo el mismo token `"morir"`.

#### N-gramas con guión bajo: nunca lematizar internamente

Todos los n-gramas consolidados (`no_hay_sistema`, `seguro_social`) tienen entrada en EXCEPCIONES_LEMA que retorna el token tal cual. Sin esta protección, spaCy podría intentar lematizar el token compuesto como si fuera una sola palabra, produciendo resultados incorrectos.

---

### 4. NER con doble estrategia: modelo + búsqueda de respaldo

El NER de `es_core_news_sm` está entrenado principalmente en texto periodístico formal. En texto coloquial de Reddit (abreviaciones, mayúsculas variables, apodos institucionales), su recall es limitado. El script usa dos estrategias en paralelo:

**Estrategia 1 — NER del modelo:** detecta entidades `ORG`, `PRODUCT`, `LOC`, `GPE` en el doc procesado por spaCy. Las entidades `ORG`/`PRODUCT` se normalizan con `INSTITUCION_CANONICA` si hay coincidencia exacta; si no, se incluyen si contienen palabras clave de salud (`"hospital"`, `"clínica"`, `"imss"`, `"farmacia"`...).

**Estrategia 2 — Búsqueda de respaldo manual:** independientemente de lo que NER detecte, el script recorre `INSTITUCION_CANONICA` y añade al set de instituciones cualquier término que aparezca en el texto del doc. Esto garantiza que `"el seguro"` (apodo coloquial del IMSS) sea siempre detectado como `"IMSS"` aunque el modelo NER no lo reconozca como entidad.

```python
# Estrategia 1: NER del modelo
for ent in doc.ents:
    if ent.label_ in ("ORG", "PRODUCT"):
        canonico = INSTITUCION_CANONICA.get(ent.text.lower())
        ...

# Estrategia 2: búsqueda de respaldo
for termino, canonico in INSTITUCION_CANONICA.items():
    if termino in doc.text.lower():
        instituciones_set.add(canonico)   # siempre añade aunque NER no lo vio
```

El resultado es un set de instituciones normalizado a nombres canónicos (`"IMSS"`, `"ISSSTE"`, `"Hospital_La_Raza"`) que permite filtrado y comparación regional-institucional en análisis posteriores.

---

### 5. Fórmula del peso Reddit — por qué log y no lineal

Los scores de Reddit siguen una distribución de ley de potencia: la mayoría de comentarios tiene 0–5 upvotes y unos pocos tienen cientos. Usar el score crudo como peso produciría que esos outliers dominen completamente la matriz TF-IDF, dando un peso 200× mayor a un comentario viral que a uno con 5 upvotes.

La transformación logarítmica comprime la distribución manteniendo el orden relativo:

$$\text{peso} = \log(1 + score\_cmt) \times 0.7 + \log(1 + score\_post) \times 0.3$$

Los factores `0.7` y `0.3` reflejan que el score del comentario es más informativo que el del hilo padre: el hilo puede ser popular por razones ajenas a la queja específica.

**Penalización por profundidad:**

| Profundidad | Factor | Justificación |
|---|---|---|
| 0–1 (top-level) | ×1.0 | Mayor visibilidad, más votos relevantes |
| 2 | ×0.85 | Segundo nivel: visible pero menos alcanzado |
| ≥3 | ×0.70 | Respuestas anidadas: audiencia muy reducida |

**Floor en 0.1:** los comentarios sin upvotes (nuevos o ignorados) reciben peso mínimo 0.1 en vez de 0, evitando que queden anulados en la multiplicación de la matriz TF-IDF.

---

### 6. Procesamiento en lotes con `nlp.pipe()`

El método `nlp(texto)` crea un nuevo doc para cada llamada, incluyendo inicialización del pipeline interno. `nlp.pipe(lista_textos, batch_size=64)` procesa los textos como un stream continuo, reutilizando la infraestructura interna y aprovechando las optimizaciones de batch de cython:

```
nlp(texto) × 4800 llamadas ≈ 4800 inicializaciones   → lento
nlp.pipe(4800 textos, batch_size=64) ≈ 75 lotes      → 3–5× más rápido
```

El `BATCH_SIZE=64` es el punto de equilibrio entre memoria y velocidad para `es_core_news_sm`. Con el modelo `lg` y GPU disponible, valores de 128–256 pueden ser más eficientes.

---

## 📊 Esquema del dataset de salida

El JSON de salida tiene dos claves de primer nivel: `meta_normalizacion` y `datos`.

### `meta_normalizacion` — Reporte de la ejecución

```json
{
  "meta_normalizacion": {
    "generado":               "2025-04-01T19:00:00",
    "version_script":         "2.0",
    "modelo_spacy":           "es_core_news_sm",
    "batch_size":             64,
    "registros_entrada":      4790,
    "registros_salida":       4771,
    "registros_vaciados":     19,
    "tokens_raw_total":       312480,
    "tokens_final_total":     187420,
    "ratio_reduccion_promedio": 0.401,
    "vocabulario_unico":      8234,
    "registros_con_ngramas":  3102,
    "registros_con_ner_inst": 4209,
    "top_30_terminos": [
      {"termino": "imss",          "frecuencia": 3241},
      {"termino": "no_hay",        "frecuencia": 2198},
      {"termino": "medicamento",   "frecuencia": 1876}
    ],
    "top_ngramas_detectados": [
      {"ngrama": "no_hay",              "frecuencia": 2198},
      {"ngrama": "sin_medicamento",     "frecuencia": 1043},
      {"ngrama": "seguro_social",       "frecuencia": 987}
    ],
    "instituciones_ner_top10": [
      {"institucion": "IMSS",    "menciones": 4102},
      {"institucion": "ISSSTE",  "menciones": 892}
    ]
  }
}
```

### `datos` — Registros normalizados

Cada registro hereda todos los campos del enriquecedor y añade:

```json
{
  "...campos heredados del enriquecedor...",

  "tokens":            ["no_hay_medicamento", "imss", "fallecer", "espera", "horas"],
  "texto_normalizado": "no_hay_medicamento imss fallecer espera horas",
  "tokens_raw":        87,
  "tokens_final":      31,
  "ratio_reduccion":   0.644,

  "entidades_ner": {
    "instituciones": ["IMSS"],
    "lugares":       ["CDMX"],
    "raw_ner": [
      {"texto": "IMSS",  "tipo": "ORG"},
      {"texto": "CDMX",  "tipo": "GPE"}
    ]
  },

  "peso_reddit": 2.3147
}
```

> **`texto_normalizado`:** es el campo que consume `vectorizar_v3.py` directamente como entrada para TF-IDF. Contiene los lemas filtrados separados por espacio, con n-gramas del dominio como tokens únicos con guión bajo.
>
> **`tokens`:** lista equivalente a `texto_normalizado.split()`, útil para modelos que necesitan la secuencia (FastText, Word2Vec, transformers).

---

## 📈 Flujo de ejecución

```
instalar_spacy()
  ├── ¿spaCy importable?  → continuar
  └── NO → pip install spacy → descarga es_core_news_sm

cargar_spacy()
  ├── spacy.load(MODELO_SPACY)
  ├── select_pipes([morphologizer, lemmatizer, ner])
  └── retorna nlp + stopwords_spacy

pedir_archivos()  →  selección interactiva JSON de entrada
      │
      ▼
json.load()  →  extrae "datos" y "meta_enriquecimiento"
      │
      ▼
┌─── Bucle en lotes de BATCH_SIZE=64 ──────────────────────────────────┐
│                                                                       │
│  preprocesar_texto(comentario)                                        │
│    ├── :emoji: → texto plano                                          │
│    └── aplicar_ngramas() → "seguro social" → "seguro_social"         │
│                                                                       │
│  flush_batch() → procesar_batch(lote, nlp, stopwords)                │
│    nlp.pipe(textos, batch_size=64)                                    │
│      │                                                                │
│      ▼  Por cada token del doc:                                       │
│      ├── filtros de ruido (punct, space, num, len<2)                  │
│      ├── resolver_lema() → EXCEPCIONES_LEMA > spaCy > texto          │
│      └── filtro stopwords (protegidos pasan siempre)                 │
│                                                                       │
│    extraer_entidades_ner(doc)                                         │
│      ├── NER del modelo (ORG, LOC, GPE)                               │
│      └── búsqueda manual INSTITUCION_CANONICA (respaldo)             │
│                                                                       │
│    calcular_peso_reddit(reg)                                          │
│      └── log1p(score_c)×0.7 + log1p(score_p)×0.3 × factor_prof      │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
      │
      ▼
json.dump()  →  { "meta_normalizacion": {...}, "datos": [...] }
      │
      ▼
Reporte: top términos | top n-gramas | instituciones NER | estadísticas
```

---

## 🔬 Primeros pasos con el corpus normalizado

```python
import json
import pandas as pd
from collections import Counter

with open("dataset_normalizado_YYYYMMDD.json") as f:
    raw = json.load(f)

df = pd.DataFrame(raw["datos"])

# 1. Ver la reducción de vocabulario
print(f"Ratio reducción promedio: {df['ratio_reduccion'].mean():.1%}")
print(f"Tokens raw total:   {df['tokens_raw'].sum():,}")
print(f"Tokens final total: {df['tokens_final'].sum():,}")

# 2. Top 20 términos del corpus normalizado
todos_tokens = [tok for tokens in df["tokens"] for tok in tokens]
print(Counter(todos_tokens).most_common(20))

# 3. Top 10 n-gramas activos (tokens con guión bajo)
ngramas = [tok for tok in todos_tokens if "_" in tok]
print(Counter(ngramas).most_common(10))

# 4. Quejas filtradas por institución usando NER
quejas_imss = df[df["entidades_ner"].apply(
    lambda x: "IMSS" in x.get("instituciones", [])
)]
print(f"Comentarios que mencionan IMSS: {len(quejas_imss)}")

# 5. Distribución de pesos Reddit
print(df["peso_reddit"].describe())
print(f"Comentarios con peso < 0.5 (sin votos): {(df['peso_reddit'] < 0.5).sum()}")

# 6. Texto listo para TF-IDF
corpus_tfidf = df["texto_normalizado"].tolist()
print(f"Ejemplo: '{corpus_tfidf[0]}'")
```

---

## 🔗 Posición en el pipeline completo

```
[1] scraper_influenza_stealth_v3.py   →  dataset crudo
[2] limpiar_dataset_v3.py             →  corpus filtrado + clasificado
[3] enriquecer_dataset_v3.py          →  corpus con IFB, polaridad, impacto
                                              │
                                              ▼
[4a] normalizar_nlp_v3.py             ←  aquí estamos
     │  Lematiza, protege negaciones, consolida n-gramas,
     │  extrae NER, calcula peso Reddit
     ▼
dataset_normalizado_YYYYMMDD.json
     │  Campo clave: texto_normalizado
     │                ↓ TF-IDF / BoW
     │            tokens (lista)
     │                ↓ FastText / Word2Vec / Transformers
     │            entidades_ner
     │                ↓ análisis geoespacial o institucional
     │            peso_reddit
     │                ↓ ponderación de la matriz TF-IDF
     ▼
[4b] vectorizar_v3.py
     │  Construye X, y y artefactos de transformación
     ▼
[5] analizar_cnb_v2.py
     │  Validación, calibración, diagnóstico
     ▼
reporte final + narrativa para tesis
```

---

## ⚠️ Consideraciones metodológicas

- El campo `texto_normalizado` **elimina stopwords** que en otro contexto serían informativas. Solo debe usarse para TF-IDF y modelos de bolsa de palabras. Para BERT y modelos de lenguaje, usar el campo `comentario` (texto limpio con stopwords) del enriquecedor, como se describe en `vectorizar_v3.py`.
- Los registros `vaciados` (0 tokens tras filtrado) suelen ser comentarios muy cortos, con solo emojis o con texto completamente en stopwords. Son descartados automáticamente y contabilizados en `meta_normalizacion.registros_vaciados`.
- El campo `entidades_ner.raw_ner` preserva las entidades tal como las detectó spaCy antes de normalización. Útil para debugging y para detectar variantes no cubiertas por `INSTITUCION_CANONICA`.
- `ratio_reduccion_promedio` típicamente cae entre 0.35–0.55 para este corpus (35–55% de tokens eliminados). Valores por encima de 0.70 pueden indicar que `TERMINOS_DOMINIO` necesita ampliarse para el corpus específico.

---

## 📝 Licencia

Uso académico / investigación. Consultar con el equipo responsable del proyecto antes de redistribuir o publicar datos recolectados.

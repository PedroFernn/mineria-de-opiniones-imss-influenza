# 🔤 Manual de Usuario — Normalizador NLP v3
### Salud Pública MX / Influenza · `normalizar_nlp_v3.py`

---

## ¿Qué hace este script?

`normalizar_nlp_v3.py` es la **cuarta etapa** del pipeline de investigación y la primera del bloque de modelado. Toma el dataset enriquecido y produce el `texto_normalizado` que el vectorizador consume directamente.

Concretamente, por cada comentario:

- **Lematiza** el texto (reduce las palabras a su forma base)
- **Elimina stopwords** de forma inteligente — conservando negaciones y causales que son clave para el sentido de la queja
- **Consolida frases del dominio** como tokens únicos (p.ej. `"no hay medicamento"` → `no_hay_medicamento`)
- **Extrae entidades nombradas** (instituciones como IMSS, ISSSTE; lugares como CDMX)
- **Calcula un peso Reddit** por comentario según sus upvotes y profundidad en el hilo

---

## Requisitos

**Python 3.10 o superior**

### Dependencia principal — spaCy

```bash
pip install spacy
python -m spacy download es_core_news_sm
```

> El script detecta si spaCy no está instalado y ofrece instalarlo automáticamente al inicio. Si la descarga del modelo falla por permisos, imprime el comando exacto que debes correr manualmente.

### Modelos alternativos de spaCy

| Modelo | Tamaño | Mejor para |
|---|---|---|
| `es_core_news_sm` *(por defecto)* | ~12 MB | Velocidad en corpus grandes |
| `es_core_news_md` | ~43 MB | Balance velocidad / calidad |
| `es_core_news_lg` | ~566 MB | Máxima calidad NER y lematización |

Para cambiar el modelo: editar `MODELO_SPACY = "es_core_news_lg"` en la configuración del script.

---

## Cómo ejecutar el script

```bash
python3 normalizar_nlp_v3.py
```

El script es **completamente interactivo**. Al correrlo te pedirá:

1. **Elegir el archivo de entrada** — muestra los JSON disponibles en la carpeta. Debe ser el output de `enriquecer_dataset_v3.py`.
2. **Nombre del archivo de salida** — puedes escribir un nombre o presionar Enter para usar el sugerido automáticamente.

El progreso se muestra en la misma línea de terminal (`→ 1280/4790 (26%)...`) para no saturarla.

---

## Qué esperar durante la ejecución

| Lo que ves en consola | Causa | Duración aprox. |
|---|---|---|
| `Instalando spaCy...` | Primera ejecución sin spaCy | 30–60 s |
| `Descargando modelo es_core_news_sm (~12 MB)...` | Primera ejecución sin el modelo | 10–30 s |
| `→ XXXX/YYYY (ZZ%)...` actualizándose | Procesamiento en lotes de 64 | 2–4 min para ~4 800 registros |

> Con `es_core_news_lg` el tiempo de procesamiento se multiplica por ~4.

---

## Archivos de entrada y salida

```
normalizar_nlp_v3.py                           ← Script principal
dataset_enriquecido_YYYYMMDD.json              ← Entrada (viene del enriquecedor)
dataset_normalizado_YYYYMMDD_HHMMSS.json       ← Salida lista para vectorizar
```

---

## ¿Qué contiene el archivo de salida?

El JSON de salida tiene dos secciones:

### 1. Resumen de la ejecución (`meta_normalizacion`)

- Total de registros entrada / salida / descartados
- Total de tokens antes y después del filtrado (y ratio promedio de reducción)
- Vocabulario único del corpus
- Top 30 términos más frecuentes
- Top n-gramas del dominio detectados
- Top 10 instituciones detectadas por NER

### 2. Registros normalizados (`datos`)

Cada comentario conserva todos sus campos del enriquecedor y se le agregan:

| Campo nuevo | Descripción |
|---|---|
| `texto_normalizado` | Texto lematizado y filtrado, listo para TF-IDF. Frases del dominio aparecen como `no_hay_medicamento` |
| `tokens` | Lista equivalente a `texto_normalizado.split()`, útil para FastText o Word2Vec |
| `tokens_raw` | Número de tokens antes de filtrar |
| `tokens_final` | Número de tokens tras filtrar stopwords y ruido |
| `ratio_reduccion` | Proporción de tokens eliminados (valor típico: 0.35–0.55) |
| `entidades_ner.instituciones` | Instituciones normalizadas detectadas (ej. `["IMSS", "ISSSTE"]`) |
| `entidades_ner.lugares` | Lugares detectados (ej. `["CDMX", "Monterrey"]`) |
| `entidades_ner.raw_ner` | Entidades tal como las vio spaCy, antes de normalizar (útil para debugging) |
| `peso_reddit` | Peso calculado a partir de upvotes del comentario, del post y profundidad del hilo |

---

## Usos del campo `texto_normalizado`

- **TF-IDF / Bag of Words** → usar `texto_normalizado` directamente como corpus
- **FastText / Word2Vec** → usar el campo `tokens` (lista de tokens)
- **BERT / Transformers** → usar el campo `comentario` del enriquecedor (con stopwords), NO `texto_normalizado`
- **Análisis institucional** → filtrar por `entidades_ner.instituciones`
- **Ponderación de matriz TF-IDF** → usar `peso_reddit` como `sample_weight`

> Los comentarios completamente vaciados tras el filtrado (muy cortos, solo emojis, etc.) son descartados y contabilizados en `meta_normalizacion.registros_vaciados`.

---

## Cómo usar el corpus normalizado (ejemplos rápidos con pandas)

```python
import json
import pandas as pd
from collections import Counter

with open("dataset_normalizado_YYYYMMDD.json") as f:
    raw = json.load(f)

df = pd.DataFrame(raw["datos"])

# Reducción de vocabulario
print(f"Ratio reducción promedio: {df['ratio_reduccion'].mean():.1%}")

# Top 20 términos del corpus
todos_tokens = [tok for tokens in df["tokens"] for tok in tokens]
print(Counter(todos_tokens).most_common(20))

# Top n-gramas activos (tokens con guión bajo)
ngramas = [tok for tok in todos_tokens if "_" in tok]
print(Counter(ngramas).most_common(10))

# Quejas filtradas por institución
quejas_imss = df[df["entidades_ner"].apply(
    lambda x: "IMSS" in x.get("instituciones", [])
)]
print(f"Comentarios que mencionan IMSS: {len(quejas_imss)}")

# Texto listo para TF-IDF
corpus_tfidf = df["texto_normalizado"].tolist()
print(f"Ejemplo: '{corpus_tfidf[0]}'")
```

---

## Posición en el pipeline completo

```
[1] scraper_influenza_stealth_v3.py   →  dataset crudo
[2] limpiar_dataset_v3.py             →  corpus filtrado + clasificado
[3] enriquecer_dataset_v3.py          →  corpus con IFB, polaridad, impacto
     │
     ▼
[4a] normalizar_nlp_v3.py            ← aquí estamos
     │  Lematiza, protege negaciones, consolida n-gramas,
     │  extrae NER, calcula peso Reddit
     ▼
dataset_normalizado_YYYYMMDD.json
     │
     ▼
[4b] vectorizar_v3.py
     │  Construye X, y y artefactos de transformación
     ▼
[5] analizar_cnb_v2.py
     │  Validación, calibración, diagnóstico
     ▼
Reporte final + narrativa para tesis
```

---

## Consideraciones de uso

- El campo `texto_normalizado` elimina stopwords, por lo que **solo debe usarse para TF-IDF y modelos de bolsa de palabras**. Para BERT y modelos de lenguaje usar el campo `comentario` del enriquecedor.
- Un `ratio_reduccion` mayor a 0.70 en promedio puede indicar que el vocabulario de dominio necesita ampliarse para el corpus específico.
- El campo `entidades_ner.raw_ner` es útil para detectar variantes institucionales que aún no están cubiertas por el mapa de normalización.

---

## Licencia

Uso académico / investigación. Consultar con el equipo responsable del proyecto antes de redistribuir o publicar datos recolectados.

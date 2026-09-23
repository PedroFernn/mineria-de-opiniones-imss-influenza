# 🏷️ Auto-Etiquetador de Sentimientos BERT — Salud Pública MX / Influenza `v1`

Paso **1.5** del pipeline de investigación, situado entre `normalizar_nlp_v2.py` y `vectorizar_v3.py`. Recibe el JSON normalizado y enriquece cada registro con análisis de sentimientos BERT usando un modelo entrenado en ~60 millones de tweets en español, actualizando el campo `relevancia` con respaldo de inteligencia artificial y trazabilidad completa del origen de cada etiqueta.

---

## 📋 Descripción

El auto-etiquetador no filtra ni elimina registros (salvo los que carecen de `texto_normalizado`): su propósito es **añadir una capa de validación semántica** a las etiquetas de relevancia asignadas en pasos anteriores. Utiliza `pysentimiento/robertuito-sentiment-analysis` — un modelo RoBERTa especializado en español coloquial — para distinguir quejas sistémicas reales de comentarios negativos sobre otros temas que pudieron colar en el corpus. Los casos de baja confianza o ambigüedad se envían a una cola de revisión manual con interfaz interactiva integrada.

---

## ✨ Funciones principales

| # | Función | Descripción |
|---|---|---|
| ① | **Instalación automática de dependencias** | Detecta en tiempo de ejecución si `pysentimiento` y `torch` están disponibles y ofrece instalarlos sin interrumpir el flujo |
| ② | **Modelo robertuito (RoBERTa MX)** | Entrenado en ~60M tweets en español (AR, MX, ES); supera a modelos multilingüe genéricos en texto coloquial con emojis, siglas y jerga mexicana |
| ③ | **Análisis por lotes configurable** | Procesa comentarios en lotes (`BATCH_SIZE=32`) con fallback automático a NEU si un lote falla, sin detener el proceso |
| ④ | **Lógica híbrida de mapeo sentimiento → relevancia** | Combina el sentimiento BERT con un mini-lexicon de quejas sistémicas para evitar falsos positivos de negatividad no relacionada con salud pública |
| ⑤ | **Corrección de falsos positivos por vacunación** | Tokens como `vacuna_info_neutral` o `vacuna_gratuita` fuerzan `relevancia = "media"` aunque el sentimiento sea NEG |
| ⑥ | **Cola de revisión manual interactiva** | Interfaz en consola para etiquetar casos ambiguos (NEG sin señal sistémica) con opciones `[a]lta / [m]edia / [s]altar / [q]ue terminar` |
| ⑦ | **Trazabilidad de etiquetas** | Campo `etiqueta_fuente` en cada registro documenta el origen: `bert_auto`, `bert_queja`, `forzado_media`, `revisado_manual` o `bert_ambiguo` |
| ⑧ | **Feature numérica lista para vectorizar** | Campo `sentimiento_num` (NEG=2, NEU=1, POS=0) añadido directamente para su uso en `FEATURES_NUMERICAS` de `vectorizar_v3.py` |

---

## 🔄 Pipeline de etiquetado

El script aplica **6 pasos secuenciales** por cada registro. El orden importa: el análisis BERT debe ocurrir antes del mapeo de relevancia, y el mapeo antes de la revisión manual.

```
[1]  Verificar e instalar dependencias  (pysentimiento, torch)
 │
[2]  Cargar modelo robertuito-sentiment-analysis
     (~500 MB en primera ejecución, cacheado localmente tras eso)
 │
[3]  Análisis en lotes BERT
     ├── Seleccionar texto más rico: comentario original > texto_normalizado
     ├── Truncar a MAX_CHARS_BERT = 800 si es necesario
     └── Predecir POS / NEG / NEU + probabilidades por clase
 │
[4]  Mapear sentimiento → relevancia  (lógica híbrida)
     ├── Token vacunación informativa detectado  → forzar "media"
     ├── POS o NEU                               → "media"
     ├── NEG + score_queja ≥ 1.5                 → "alta"
     └── NEG + score_queja < 1.5                 → marcar "revision"
 │
[5]  Revisión manual interactiva  (opcional)
     └── Casos ambiguos presentados uno a uno en consola
 │
[6]  Validar, limpiar y exportar
     ├── dataset_etiquetado_YYYYMMDD_HHMMSS.json   ← entrada para vectorizar_v3.py
     └── cola_revision_YYYYMMDD_HHMMSS.json        ← casos pendientes de revisar
```

---

## 📂 Estructura del proyecto

```
auto_etiquetar.py                                   ← Script principal
dataset_normalizado_YYYYMMDD_HHMMSS.json            ← Entrada (output de normalizar_nlp_v2.py)
dataset_etiquetado_YYYYMMDD_HHMMSS.json             ← Salida: corpus etiquetado + metadatos
cola_revision_YYYYMMDD_HHMMSS.json                  ← Casos ambiguos para revisión posterior
```

---

## ⚙️ Requisitos

**Python 3.10+**

### Dependencias — instaladas automáticamente si no están presentes

```bash
pip install pysentimiento torch
```

> **¿Por qué `robertuito-sentiment-analysis`?**
> Es el modelo de sentimientos más robusto disponible para el español coloquial mexicano. Fue entrenado en ~60 millones de tweets de Argentina, México y España, lo que le permite reconocer jerga regional (`wey`, `chido`, `neta`), emojis, signos de exclamación y abreviaciones que modelos multilingüe genéricos como `twitter-xlm-roberta` subestiman sistemáticamente. Produce probabilidades calibradas para las tres clases (POS / NEG / NEU) con suma = 1.0 y acepta textos de hasta 512 tokens — suficiente para comentarios de Reddit.
>
> En la primera ejecución descarga ~500 MB. Las ejecuciones posteriores usan la caché local de Hugging Face.

---

## 🚀 Uso

```bash
python3 auto_etiquetar.py
```

El script es completamente interactivo. Muestra los archivos `dataset_normalizado*.json` disponibles en la misma carpeta y solicita:

1. El número del archivo a etiquetar (debe ser el output de `normalizar_nlp_v2.py`)
2. Si iniciar la revisión manual de casos ambiguos al finalizar el análisis BERT

### ⚠️ Advertencias de uso

| Situación | Consecuencia |
|---|---|
| Alimentar con el JSON **crudo o enriquecido** (sin normalizar) | ⚠️ Funciona técnicamente, pero `texto_normalizado` puede estar ausente; el script usará `comentario` como fallback |
| Alimentar con el JSON **normalizado** (output de `normalizar_nlp_v2.py`) | ✅ Uso correcto — todos los campos se heredan y enriquecen con los nuevos campos BERT |
| Saltarse la revisión manual | ⚠️ Los casos ambiguos conservan su etiqueta original del dataset anterior; se exportan igualmente en `cola_revision_*.json` |
| OOM (Out of Memory) durante el análisis | ⚠️ Reducir `BATCH_SIZE` a `16` en la configuración del script |
| Interrumpir con `Ctrl+C` durante el análisis | ⚠️ El archivo de salida no se crea; el archivo de entrada permanece intacto |

---

## 🔧 Configuración principal

| Constante | Valor | Descripción |
|---|---|---|
| `MODELO_SENTIMENT` | `pysentimiento/robertuito-sentiment-analysis` | Modelo BERT a usar; se puede sustituir por alternativas más ligeras |
| `BATCH_SIZE` | `32` | Comentarios por lote; reducir a `16` si hay errores de memoria |
| `UMBRAL_QUEJA_PARA_ALTA` | `1.5` | `score_queja_salud` mínimo para que un NEG se convierta en `"alta"` |
| `UMBRAL_CONFIANZA_AUTO` | `0.72` | Probabilidad mínima para etiquetar sin revisión |
| `UMBRAL_CONFIANZA_REVISION` | `0.50` | Probabilidad por debajo de la cual el caso va a revisión manual |
| `TOKENS_FUERZA_MEDIA` | 3 tokens | Tokens que fuerzan `"media"` aunque el sentimiento sea NEG (vacunación informativa) |
| `SENTIMIENTO_NUM` | `NEG=2, NEU=1, POS=0` | Codificación numérica del sentimiento como feature para vectorizar |
| `MAX_REVISION` | `400` | Máximo de casos en la cola de revisión manual |
| `MAX_CHARS_BERT` | `800` | Límite de caracteres antes de truncar el texto para el modelo |

### Modelos alternativos

```python
# Más ligero (~250 MB), multilingüe, útil si hay problemas de descarga
MODELO_SENTIMENT = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

# Multilingüe, entrenado en Twitter, menor precisión en es-MX coloquial
MODELO_SENTIMENT = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
```

---

## 🧠 Arquitectura técnica avanzada

Esta sección documenta la lógica de mapeo y las decisiones de diseño de cada componente. Es la información clave para evaluar la validez de las etiquetas en el corpus resultante.

---

### 1. Selección del texto para el modelo BERT

El script prioriza el **comentario original** sobre el `texto_normalizado` para el análisis BERT, con razón deliberada:

| Texto | Para BERT | Para TF-IDF |
|---|---|---|
| `comentario` original | ✅ Mejor — conserva signos `¡!`, emojis y stopwords emocionales | ❌ Peor — ruido para frecuencias de términos |
| `texto_normalizado` | ❌ Peor — limpieza elimina señales emocionales clave | ✅ Mejor — tokens limpios para vectorización |

El comentario se trunca a `MAX_CHARS_BERT = 800` caracteres conservando el inicio, donde generalmente se expresa la queja principal en posts de Reddit.

---

### 2. Lógica híbrida de mapeo sentimiento → relevancia

El etiquetador no asigna `relevancia = "alta"` a todo comentario negativo. Aplica cuatro reglas en cascada:

#### Las 4 reglas en orden de prioridad

| Prioridad | Condición | Resultado | Fuente |
|---|---|---|---|
| **Regla 0** | Token de vacunación informativa presente en texto | `"media"` | `forzado_media` |
| **Regla 1** | Sentimiento POS o NEU | `"media"` | `bert_auto` |
| **Regla 2** | NEG + `score_queja ≥ 1.5` | `"alta"` | `bert_queja` |
| **Regla 3** | NEG + `score_queja < 1.5` | `→ revisión manual` | `bert_ambiguo` |

**¿Por qué no basta con el sentimiento NEG?**

Un comentario puede ser negativo sin hablar del sistema de salud: crítica a un youtuber de medicina, opinión sobre la vacuna de COVID, frustración personal no relacionada. La regla 2 exige que la negatividad BERT coexista con señales léxicas de queja sistémica (`desabasto`, `negligencia`, `viacrucis`, `urgencias_llenas`, etc.) antes de marcar el registro como `"alta"`.

---

### 3. Score de queja sistémica

El `score_queja_salud` es un contador ponderado simple sobre el `texto_normalizado`:

```
score = Σ 1.5 por cada término del mini-lexicon presente en el texto
```

**Mini-lexicon de señales sistémicas (selección):**

| Categoría | Términos |
|---|---|
| Desabasto | `desabasto`, `no_hay_medicamento`, `no_hay_vacuna`, `escasez` |
| Burocracia | `viacrucis`, `odisea`, `vuelta_y_vuelta`, `vuelva_manana` |
| Pago privado | `de_mi_bolsillo`, `medico_particular`, `tuve_que_pagar` |
| Rechazo/falla | `no_me_atendieron`, `saturado`, `no_hay_cupo`, `urgencias_llenas` |
| Impacto clínico | `fallecer`, `morir`, `muerte`, `intubado`, `uci`, `empeorar` |
| Corrupción | `corrupcion`, `mordida`, `moche` |

El umbral `UMBRAL_QUEJA_PARA_ALTA = 1.5` equivale a detectar al menos **un término** del lexicon. Se puede elevar a `3.0` para exigir dos términos si se desea mayor precisión.

---

### 4. Revisión manual interactiva

Los casos marcados como `"revision"` (NEG pero sin señal sistémica suficiente) se presentan en consola con el contexto necesario para una decisión informada:

```
─── Caso 3/20 ───────────────────────────────────────────
  Texto   : [primeros 300 caracteres del comentario original]
  BERT    : NEG (prob_neg=0.84) | score_queja=0.0 | etiqueta_orig=alta
  Etiqueta [a/m/s/q]:
```

La sesión de revisión puede detenerse en cualquier momento con `[q]`. Los casos no revisados se exportan en `cola_revision_*.json` para tratarlos en una sesión posterior. El tiempo estimado de revisión es de **30–60 minutos para ~300 casos**.

---

## 📦 Campos añadidos al dataset

Cada registro de salida incluye los siguientes campos nuevos además de todos los heredados del paso anterior:

```json
{
  "sentimiento_bert":  "NEG",        // Clase predicha por el modelo BERT
  "prob_neg":          0.8732,       // Probabilidad de sentimiento negativo
  "prob_pos":          0.0481,       // Probabilidad de sentimiento positivo
  "prob_neu":          0.0787,       // Probabilidad de sentimiento neutro
  "confianza_bert":    0.8732,       // max(prob_neg, prob_pos, prob_neu)
  "sentimiento_num":   2,            // NEG=2, NEU=1, POS=0 (feature numérica)
  "relevancia":        "alta",       // Etiqueta actualizada (alta | media)
  "relevancia_bert":   "alta",       // Copia de la decisión BERT antes de revisión manual
  "etiqueta_fuente":   "bert_queja", // Trazabilidad: origen de la etiqueta
  "etiqueta_razon":    "NEG (prob=0.87) + score_queja=3.0 ≥ 1.5",
  "requiere_revision": false         // true si el caso quedó en cola sin revisar
}
```

### Valores posibles de `etiqueta_fuente`

| Valor | Significado |
|---|---|
| `bert_auto` | POS o NEU → asignado automáticamente como `"media"` |
| `bert_queja` | NEG con señal sistémica confirmada → `"alta"` |
| `forzado_media` | Token de vacunación informativa detectado → forzado a `"media"` |
| `bert_ambiguo` | NEG sin señal clara → marcado para revisión; conserva etiqueta anterior |
| `revisado_manual` | Etiquetado manualmente durante la sesión interactiva |

---

## 📈 Flujo de ejecución

```
verificar_e_instalar()  →  pysentimiento + torch disponibles o instalados
      │
      ▼
cargar_analizador()  →  carga robertuito desde caché o descarga ~500 MB
      │
      ▼
pedir_archivo()  →  selección interactiva del JSON de entrada
      │
      ▼
json.load()  →  extrae "datos" y "meta_normalizacion" del JSON
      │
      ▼
┌─── Bucle por lotes de BATCH_SIZE registros ────────────────────────────┐
│                                                                        │
│  preparar_texto_bert()    → comentario original, truncado a 800 chars  │
│  analizador.predict()     → POS|NEG|NEU + prob_neg/pos/neu             │
│        └── fallback NEU si el lote falla por error de memoria          │
│  mapear_relevancia()      → lógica híbrida en 4 reglas                 │
│  ┌─ Regla 0: token vacuna    → "media" (forzado_media)               ─┐ │
│  │  Regla 1: POS|NEU         → "media" (bert_auto)                    │ │
│  │  Regla 2: NEG + queja≥1.5 → "alta"  (bert_queja)                  │ │
│  │  Regla 3: NEG + queja<1.5 → revisión (bert_ambiguo)               │ │
│  └──────────────────────────────────────────────────────────────────── ┘ │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
      │
      ▼
revisar_casos_ambiguos()  →  interfaz interactiva [a/m/s/q] (opcional)
      │
      ▼
validar_y_limpiar()  →  normaliza etiquetas, descarta registros sin texto
      │
      ▼
guardar_resultados()  →  { "meta_etiquetado": {...}, "datos": [...] }
                          cola_revision_*.json  (casos sin revisar)
      │
      ▼
Reporte con distribuciones visualizadas (relevancia, sentimiento, fuente)
e instrucciones para el siguiente paso en vectorizar_v3.py
```

---

## 🔬 Primeros pasos con el corpus etiquetado (análisis con pandas)

```python
import pandas as pd
import json

# Cargar solo el array de datos
with open("dataset_etiquetado_YYYYMMDD_HHMMSS.json") as f:
    raw = json.load(f)

df = pd.DataFrame(raw["datos"])

# 1. Distribución de sentimientos BERT
print(df["sentimiento_bert"].value_counts())

# 2. Confianza media por sentimiento — detectar categorías difusas
print(df.groupby("sentimiento_bert")["confianza_bert"].mean())

# 3. Casos de alta relevancia respaldados por BERT
altas_bert = df[
    (df["relevancia"] == "alta") & (df["etiqueta_fuente"] == "bert_queja")
].nlargest(10, "prob_neg")
print(altas_bert[["comentario", "prob_neg", "confianza_bert"]])

# 4. Registros forzados a "media" por token de vacunación
vacuna = df[df["etiqueta_fuente"] == "forzado_media"]
print(f"Falsos positivos corregidos (vacunación): {len(vacuna)}")

# 5. Cola de revisión manual — cargar y explorar
with open("cola_revision_YYYYMMDD_HHMMSS.json") as f:
    cola = json.load(f)
df_cola = pd.DataFrame(cola["casos"])
print(f"Casos pendientes: {cola['pendientes']}")
print(df_cola[["comentario", "prob_neg", "etiqueta_razon"]].head(10))

# 6. Cruzar sentimiento_num con otras features numéricas
print(df[["sentimiento_num", "polaridad_score", "ifb_score"]].corr())
```

---

## 🔗 Posición en el pipeline completo

Este script es el **paso 1.5** del pipeline de investigación:

```
[1] scraper_influenza_stealth_v3.py
     │  Recolecta posts y comentarios de Reddit MX
     ▼
dataset_influenza_crudo_final.json
     │
     ▼
[2] limpiar_dataset_v3.py / enriquecer_dataset_v3.py
     │  Filtra, normaliza, enriquece con IFB y polaridad localizada
     ▼
dataset_normalizado_YYYYMMDD.json
     │
     ▼
[1.5] auto_etiquetar.py   ← aquí estamos
     │  Añade sentimiento BERT, actualiza relevancia con lógica híbrida
     ▼
dataset_etiquetado_YYYYMMDD.json
cola_revision_YYYYMMDD.json
     │
     ▼
[3] vectorizar_v3.py
     │  Genera embeddings TF-IDF + features numéricas (incluye sentimiento_num)
     ▼
Análisis supervisado / clustering / visualización / exportación
```

### Integración con `vectorizar_v3.py`

Tras ejecutar el auto-etiquetador, añadir `"sentimiento_num"` a `FEATURES_NUMERICAS` en `vectorizar_v3.py`:

```python
FEATURES_NUMERICAS = [
    "polaridad_score",
    "ifb_score",
    "impacto_escala",
    "num_palabras",
    "peso_reddit",
    "sentimiento_num",   # ← añadir esta línea
]
```

El campo `"relevancia"` ya contiene etiquetas de alta calidad respaldadas por un modelo BERT de 60M tweets en español y, opcionalmente, validación humana.

---

## ⚠️ Consideraciones éticas y de validez del corpus

- El corpus etiquetado contiene opiniones ciudadanas reales. Cualquier publicación debe anonimizar o agregar los datos para no exponer a usuarios individuales.
- El campo `sentimiento_bert` es una predicción de modelo, no una etiqueta humana validada. Para publicaciones académicas se recomienda revisar manualmente al menos una muestra estratificada de los casos automáticos.
- El `score_queja_salud` es un proxy léxico simple: un comentario con alto impacto narrado con lenguaje neutro puede quedar con `score = 0.0` y un IFB real alto. Complementar con la revisión de los casos en `cola_revision_*.json`.
- Los casos con `etiqueta_fuente = "bert_ambiguo"` son los de mayor incertidumbre y deben tratarse con precaución especial antes de incluirse en conjuntos de entrenamiento supervisado.
- La corrección de falsos positivos por vacunación (`TOKENS_FUERZA_MEDIA`) puede subestimar la relevancia de comentarios donde la vacunación informativa coexiste con una queja sistémica real. Revisar manualmente si el análisis requiere precisión en ese subconjunto.

---

## 📝 Licencia

Uso académico / investigación. Consultar con el equipo responsable del proyecto antes de redistribuir o publicar datos recolectados.

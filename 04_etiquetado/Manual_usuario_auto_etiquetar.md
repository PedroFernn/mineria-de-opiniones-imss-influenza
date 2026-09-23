# 🏷️ Manual de Usuario — Auto-Etiquetador de Sentimientos BERT
### Salud Pública MX / Influenza · `auto_etiquetar.py`

---

## ¿Qué hace este script?

`auto_etiquetar.py` es el **paso 1.5** del pipeline de investigación, ubicado entre el normalizador y el vectorizador. Toma el dataset normalizado y añade a cada comentario un análisis de sentimientos realizado por un modelo BERT especializado en español coloquial mexicano.

Concretamente, por cada comentario:

- **Predice el sentimiento** (positivo, negativo o neutro) usando el modelo `robertuito-sentiment-analysis`, entrenado en ~60 millones de tweets en español
- **Actualiza el campo `relevancia`** combinando el sentimiento BERT con señales léxicas de queja sistémica — no todo comentario negativo es una queja del sistema de salud
- **Envía los casos ambiguos** a una cola de revisión manual con interfaz interactiva integrada en consola
- **Registra el origen de cada etiqueta** en el campo `etiqueta_fuente` para trazabilidad completa

---

## Requisitos

**Python 3.10 o superior**

### Dependencias — instaladas automáticamente si no están presentes

```bash
pip install pysentimiento torch
```

> En la primera ejecución el script descarga el modelo (~500 MB). Las ejecuciones posteriores usan la caché local de Hugging Face y arrancan en segundos.

### Modelos alternativos (si hay problemas de descarga o memoria)

```python
# Más ligero (~250 MB), multilingüe
MODELO_SENTIMENT = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"

# Multilingüe, menor precisión en español coloquial
MODELO_SENTIMENT = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
```

Para cambiarlo: editar `MODELO_SENTIMENT` en la configuración del script.

---

## Cómo ejecutar el script

```bash
python3 auto_etiquetar.py
```

El script es **completamente interactivo**. Al correrlo te pedirá:

1. **Elegir el archivo de entrada** — muestra los archivos `dataset_normalizado*.json` disponibles en la carpeta. Debe ser el output de `normalizar_nlp_v3.py`.
2. **Si iniciar la revisión manual** de casos ambiguos al finalizar el análisis BERT.

---

## Archivos de entrada y salida

```
auto_etiquetar.py                                   ← Script principal
dataset_normalizado_YYYYMMDD_HHMMSS.json            ← Entrada (viene del normalizador)
dataset_etiquetado_YYYYMMDD_HHMMSS.json             ← Salida: corpus etiquetado
cola_revision_YYYYMMDD_HHMMSS.json                  ← Casos ambiguos para revisar después
```

---

## Advertencias de uso

| Situación | Qué pasa |
|---|---|
| Usar el JSON **normalizado** como entrada | ✅ Uso correcto |
| Usar el JSON **crudo o enriquecido** (sin normalizar) | Funciona, pero `texto_normalizado` puede estar ausente; el script usará `comentario` como alternativa |
| Saltarse la revisión manual | Los casos ambiguos conservan su etiqueta anterior y se exportan en `cola_revision_*.json` para tratarlos después |
| Error de memoria durante el análisis | Reducir `BATCH_SIZE` a `16` en la configuración del script |
| Interrumpir con `Ctrl+C` | El archivo de salida no se crea; el archivo de entrada queda intacto |

---

## ¿Qué contiene el archivo de salida?

### Campos nuevos en cada registro

| Campo | Descripción |
|---|---|
| `sentimiento_bert` | Clase predicha: `NEG`, `NEU` o `POS` |
| `prob_neg` / `prob_pos` / `prob_neu` | Probabilidades por clase (suman 1.0) |
| `confianza_bert` | Probabilidad de la clase ganadora |
| `sentimiento_num` | Codificación numérica: NEG=2, NEU=1, POS=0 (feature lista para vectorizar) |
| `relevancia` | Etiqueta actualizada: `alta` o `media` |
| `relevancia_bert` | Copia de la decisión BERT antes de cualquier revisión manual |
| `etiqueta_fuente` | Origen de la etiqueta (ver tabla abajo) |
| `etiqueta_razon` | Explicación textual de la decisión (útil para debugging) |
| `requiere_revision` | `true` si el caso quedó en cola sin revisar |

### Valores de `etiqueta_fuente`

| Valor | Significado |
|---|---|
| `bert_auto` | Sentimiento POS o NEU → `"media"` asignado automáticamente |
| `bert_queja` | Sentimiento NEG con señal sistémica confirmada → `"alta"` |
| `forzado_media` | Token de vacunación informativa detectado → forzado a `"media"` |
| `bert_ambiguo` | Sentimiento NEG sin señal sistémica clara → conserva etiqueta anterior, va a cola |
| `revisado_manual` | Etiquetado manualmente durante la sesión interactiva |

---

## Cómo funciona la revisión manual interactiva

Los casos marcados como ambiguos se presentan uno por uno en consola:

```
─── Caso 3/20 ───────────────────────────────────────────
  Texto   : [primeros 300 caracteres del comentario]
  BERT    : NEG (prob_neg=0.84) | score_queja=0.0 | etiqueta_orig=alta
  Etiqueta [a/m/s/q]:
```

Opciones disponibles:

| Tecla | Acción |
|---|---|
| `a` | Marcar como `alta` relevancia |
| `m` | Marcar como `media` relevancia |
| `s` | Saltar (conservar etiqueta anterior) |
| `q` | Terminar la sesión de revisión |

Los casos no revisados se exportan en `cola_revision_*.json` para tratarlos en otra sesión. El tiempo estimado para ~300 casos es de 30–60 minutos.

---

## Integración con el siguiente paso (`vectorizar_v3.py`)

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

---

## Cómo usar el corpus etiquetado (ejemplos rápidos con pandas)

```python
import pandas as pd
import json

with open("dataset_etiquetado_YYYYMMDD_HHMMSS.json") as f:
    raw = json.load(f)

df = pd.DataFrame(raw["datos"])

# Distribución de sentimientos
print(df["sentimiento_bert"].value_counts())

# Confianza media por sentimiento
print(df.groupby("sentimiento_bert")["confianza_bert"].mean())

# Casos de alta relevancia respaldados por BERT
altas = df[
    (df["relevancia"] == "alta") & (df["etiqueta_fuente"] == "bert_queja")
].nlargest(10, "prob_neg")
print(altas[["comentario", "prob_neg", "confianza_bert"]])

# Falsos positivos corregidos por token de vacunación
vacuna = df[df["etiqueta_fuente"] == "forzado_media"]
print(f"Casos corregidos (vacunación informativa): {len(vacuna)}")

# Cola de revisión pendiente
with open("cola_revision_YYYYMMDD_HHMMSS.json") as f:
    cola = json.load(f)
print(f"Casos pendientes de revisar: {cola['pendientes']}")

# Correlación entre sentimiento BERT y otras métricas
print(df[["sentimiento_num", "polaridad_score", "ifb_score"]].corr())
```

---

## Posición en el pipeline completo

```
[1] scraper_influenza_stealth_v3.py   →  dataset crudo
[2] limpiar_dataset_v3.py             →  corpus filtrado + clasificado
[3] enriquecer_dataset_v3.py          →  corpus con IFB, polaridad, impacto
[4] normalizar_nlp_v3.py              →  corpus lematizado con texto_normalizado
     │
     ▼
[1.5] auto_etiquetar.py              ← aquí estamos
     │  Añade sentimiento BERT, actualiza relevancia, genera cola de revisión
     ▼
dataset_etiquetado_YYYYMMDD.json
cola_revision_YYYYMMDD.json
     │
     ▼
[5] vectorizar_v3.py
     │  Genera matriz TF-IDF + features numéricas (incluye sentimiento_num)
     ▼
Análisis supervisado / clustering / visualización
```

---

## Consideraciones de uso

- El campo `sentimiento_bert` es una predicción de modelo, no una etiqueta humana validada. Para publicaciones académicas se recomienda revisar manualmente una muestra representativa de los casos automáticos.
- Los casos con `etiqueta_fuente = "bert_ambiguo"` son los de mayor incertidumbre y deben tratarse con precaución antes de incluirse en conjuntos de entrenamiento supervisado.
- La corrección por vacunación (`forzado_media`) puede subestimar quejas donde la vacunación informativa coexiste con una queja sistémica real. Revisar ese subconjunto manualmente si el análisis lo requiere.
- El corpus contiene opiniones ciudadanas reales. Cualquier publicación debe anonimizar o agregar los datos.

---

## Licencia

Uso académico / investigación. Consultar con el equipo responsable del proyecto antes de redistribuir o publicar datos recolectados.

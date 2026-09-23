# 🧹 Limpiador de Dataset — Salud Pública MX / Influenza `v3`

Pipeline de limpieza y depuración del dataset crudo generado por `scraper_influenza_stealth_v3.py`. Prepara el corpus para análisis de sentimientos, detección de temas y vigilancia epidemiológica social.

---

## 📋 Descripción

El script recibe el JSON producido por el scraper (bruto, con ruido, emojis sin convertir y duplicados residuales) y lo transforma en un corpus estructurado y semánticamente etiquetado. Cada registro de salida incluye el texto original intacto, su versión limpia, su versión normalizada para NLP y un campo `palabras_clave_activas` que documenta exactamente qué términos justificaron la inclusión del registro en el dataset final.

---

## ✨ Novedades en v3

| Mejora | Detalle |
|---|---|
| **`INDICADORES_PROBLEMA` ampliados** | Nuevas categorías: tiempo/burocracia, impacto clínico grave, pago privado como señal de falla del sistema |
| **Lematizador ligero** | Normaliza jerga digital mexicana: `k`/`q` → `que`, `xq` → `porque`, énfasis de letras repetidas (`pesimooooo` → `pesimo`) |
| **`SUSTANTIVOS_SALUD` ampliados** | Nivel local: UMFs, módulos de salud, referencias coloquiales a clínicas IMSS; farmacia/botica como nodo de suministro |
| **Preservación de puntuación** | `limpiar_texto` conserva `¡!¿?` y el espacio en negaciones `no + verbo` — señales críticas de sentimiento |
| **Contexto de activación** | `clasificar_relevancia` devuelve las palabras clave y categorías que activaron cada filtro |
| **Campo `palabras_clave_activas`** | Cada registro del dataset incluye la trazabilidad semántica de su clasificación |

---

## 🔄 Pipeline de limpieza

El script aplica **11 pasos secuenciales** en orden estricto. El orden importa: los pasos de conversión y normalización deben ocurrir antes de la clasificación para que los filtros léxicos capturen variantes coloquiales.

```
[1]  Cargar JSON (streaming si >50 MB)
 │
[2]  Eliminar registros sin comentario / campos vacíos
 │
[3]  Filtrar bots y cuentas eliminadas
 │
[4]  Emojis → texto descriptivo  (😡 → :cara_enojada:)
 │
[5]  Normalización ligera  (k→que, pesimooo→pesimo)
 │
[6]  Limpiar texto  (HTML, URLs, Markdown)  ← preserva ¡!¿? y negaciones
 │
[7]  Filtrar por longitud mínima  (MIN_CHARS = 30)
 │
[8]  Filtrar por score mínimo  (MIN_SCORE = -5)
 │
[9]  Filtro de relevancia combinado  (OR + AND) + categorías semánticas
 │
[10] Deduplicación residual por hash MD5
 │
[11] Guardar resultado + reporte detallado en consola
```

---

## 📂 Estructura del proyecto

```
limpiar_dataset_v3.py                            ← Script principal
dataset_influenza_crudo_final_YYYYMMDD.json      ← Entrada (output del scraper)
dataset_limpio_YYYYMMDD_HHMMSS.json              ← Salida: corpus limpio + metadatos
```

---

## ⚙️ Requisitos

**Python 3.10+**

### Dependencias obligatorias

```bash
pip install requests   # ya instalado si usaste el scraper
```

> El script no requiere `requests` directamente, pero sí usa la biblioteca estándar completa (`re`, `hashlib`, `json`, `pathlib`, `html`).

### Dependencias opcionales — Pre-requisitos críticos para la funcionalidad completa

```bash
pip install emoji    # Conversión de emojis a texto descriptivo
pip install ijson    # Carga en streaming para datasets >50 MB
```

> **¿Por qué son importantes?**
>
> - **`emoji`**: Sin esta librería, los emojis (`😡 😤 🏥`) permanecen como caracteres Unicode en el texto. Los modelos de análisis de sentimientos entrenados en texto español **no reconocen estos símbolos** como señales emocionales. Instalar `emoji` los convierte a tokens de texto (`:cara_enojada:`, `:hospital:`) que los clasificadores sí pueden procesar. Un corpus sin esta conversión subestima sistemáticamente la carga emocional negativa de las quejas.
>
> - **`ijson`**: El scraper en v3 puede generar datasets de varios cientos de MB. Cargarlos con `json.load()` estándar requiere que el archivo completo quepa en RAM. `ijson` permite un recorrido en streaming registro a registro, manteniendo el consumo de memoria constante independientemente del tamaño del archivo. Sin esta librería, datasets grandes pueden agotar la RAM disponible.
>
> El script detecta ambas librerías en tiempo de ejecución y **ofrece instalarlas automáticamente** al arrancar si no están presentes.

---

## 🚀 Uso

```bash
python3 limpiar_dataset_v3.py
```

El script es completamente interactivo. Al ejecutarse muestra los archivos JSON disponibles en la misma carpeta y solicita:

1. El número del archivo a limpiar
2. El nombre/ruta del archivo de salida (o Enter para usar el nombre sugerido)

Si faltan dependencias opcionales, pregunta si instalarlas antes de continuar.

### 🕐 Qué esperar durante la ejecución

| Lo que ves en consola | Causa | Normal |
|---|---|---|
| `📦 Archivo grande (XXX MB) — usando streaming (ijson)` | Dataset supera 50 MB | ✅ |
| `⚠️ Archivo grande — ijson no disponible, cargando en RAM` | `ijson` no instalado | ⚠️ Instalar recomendado |
| Barra de categorías con bloques `█` en el reporte final | Distribución visual de categorías semánticas | ✅ |
| `⚠️ Librerías opcionales no encontradas: emoji, ijson` | Primera ejecución sin dependencias | ✅ Escribe `s` para instalar |

### ⚠️ Advertencia: el script modifica solo los datos de salida

El archivo de entrada **nunca se modifica**. El script siempre escribe en un archivo de salida separado. Si el proceso se interrumpe antes de `guardar_dataset()`, el archivo de salida no se crea y los datos de entrada permanecen intactos.

---

## 🔧 Configuración principal

| Constante | Valor por defecto | Descripción |
|---|---|---|
| `MIN_CHARS` | `30` | Longitud mínima del comentario limpio (caracteres) |
| `MIN_SCORE_COMENTARIO` | `-5` | Score mínimo de Reddit para conservar el comentario |
| `UMBRAL_STREAMING_MB` | `50` | Tamaño a partir del cual se usa `ijson` en vez de `json.load` |
| `AUTORES_DESCARTADOS` | 8 entradas | Bots conocidos y cuentas eliminadas |
| `PALABRAS_OR` | ~80 términos | Red amplia de primera criba temática |
| `SUSTANTIVOS_SALUD` | ~30 términos | Entidades del sistema de salud (filtro AND) |
| `INDICADORES_PROBLEMA` | ~50 términos | Señales de falla o queja (filtro AND) |
| `CATEGORIAS_CONTEXTO` | 9 categorías | Etiquetas semánticas para trazabilidad |

---

## 🧠 Arquitectura técnica avanzada

Esta sección documenta la lógica interna del pipeline para investigadores que necesiten evaluar las decisiones de diseño y su impacto en la validez del corpus resultante.

---

### 1. Lematizador ligero — Normalización de jerga digital mexicana

Antes de aplicar los filtros léxicos, el texto pasa por `normalizar_texto()`. Esta función opera en **dos pasos independientes** y su resultado se usa exclusivamente para la clasificación; el texto original limpio se preserva intacto en el campo `comentario` del dataset de salida.

#### Paso 1 — Colapso de énfasis emocional

Reddit mexicano usa repetición de letras como marcador de intensidad: `"pesimooooo"`, `"maaalísimoooo"`, `"nooooo había nada"`. Sin normalización, estas variantes no coinciden con las palabras clave del diccionario.

```python
# Patrón: cualquier carácter repetido 3+ veces → colapsar a 2
_RE_LETRAS_REPETIDAS = re.compile(r"(.)\1{2,}", re.UNICODE)

# "maaalísimoooo" → "maalísimoo" → segunda pasada → "malísimo"
# (se preservan dígrafos legítimos: ll, rr, nn, cc)
```

El énfasis colapsado **no se pierde semánticamente**: la presencia de repetición queda registrada en el texto original (`comentario_raw`) y puede recuperarse para análisis de intensidad emocional.

#### Paso 2 — Substitución de abreviaturas

| Variante coloquial | Forma canónica | Impacto en clasificación |
|---|---|---|
| `k`, `q` | `que` | Detecta negaciones: `"k no había nada"` |
| `xq`, `xque`, `porqu` | `porque` | Captura causalidad: `"xq no había médico"` |
| `tb`, `tmb` | `también` | Señales de acumulación de quejas |
| `d` (preposición sola) | `de` | Frases compuestas: `"d mi bolsillo"` |
| `pa` (preposición) | `para` | Contexto: `"pa qué sirve el IMSS"` |

---

### 2. El sistema de clasificación OR + AND

La función `clasificar_relevancia()` aplica dos filtros en cascada sobre el texto normalizado. Este diseño de dos niveles balancea **cobertura** (no perder quejas legítimas) con **precisión** (no incluir menciones superficiales de salud sin contenido relevante).

#### Nivel 1 — Filtro OR (red amplia)

Pasa si el texto contiene **al menos una** de las ~80 palabras en `PALABRAS_OR`. Este filtro descarta comentarios completamente ajenos al tema (bromas, memes, conversaciones de política sin mención de salud). Un comentario que solo pasa el filtro OR recibe relevancia `"media"`.

#### Nivel 2 — Filtro AND (relevancia alta)

Un comentario pasa a `"alta"` si, además del OR, cumple **ambas** condiciones simultáneamente:

```
SUSTANTIVOS_SALUD (entidad institucional o clínica presente)
        AND
INDICADORES_PROBLEMA (señal explícita de falla, queja o consecuencia)
```

```
Ejemplo relevancia ALTA:
  "Fui al IMSS y no había oseltamivir, me mandaron a casa con fiebre"
  → Sustantivo: "imss" ✓
  → Indicador: "no había", "mandaron a casa" ✓
  → Relevancia: alta

Ejemplo relevancia MEDIA:
  "Mi vecina tuvo influenza AH1N1 el año pasado"
  → OR: "influenza", "h1n1" ✓
  → Sustantivo: ninguno ✗
  → Relevancia: media
```

La distinción alta/media es valiosa para el análisis: los registros `alta` son los candidatos prioritarios para entrenamiento de clasificadores; los `media` son útiles para análisis exploratorio y detección de contexto.

#### Las 9 categorías semánticas

Además de alta/media, el sistema asigna **etiquetas de categoría** a cada registro. Estas etiquetas no son excluyentes (un comentario puede activar múltiples):

| Categoría | Señal que captura |
|---|---|
| `desabasto` | Falta de medicamentos, vacunas agotadas, anaqueles vacíos |
| `tiempo_espera` | Filas, horas de espera, citas en meses, burocracia |
| `saturacion` | Urgencias llenas, sin camas, sin cupo |
| `corrupcion` | Moches, mordidas, palancas, privatización encubierta |
| `maltrato` | Rechazo de atención, grosería, negligencia explícita |
| `negligencia` | Diagnóstico incorrecto, pésimo servicio general |
| `impacto_salud` | Complicaciones graves, UCI, fallecimiento |
| `pago_privado` | Pago de bolsillo como señal de falla del sistema público |
| `recorte` | Austeridad, sistema caído, desfinanciamiento |

Estas categorías se guardan en `palabras_clave_activas.categorias_activas` y se contabilizan en el reporte final, produciendo una **distribución semántica del corpus** visualizada como barra en la consola.

---

### 3. Preservación deliberada de señales de sentimiento

`limpiar_texto()` toma decisiones explícitas sobre qué eliminar y qué conservar, justificadas por el objetivo de análisis de sentimientos:

**Se eliminan** (ruido sin valor semántico):
- Entidades HTML (`&amp;` → `&` vía `unescape`, luego residuos eliminados)
- URLs (`https://...`, `www....`)
- Menciones Reddit (`u/usuario`, `r/subreddit`)
- Marcado Markdown (`**negrita**`, `*cursiva*`, bloques de código, citas `>`)

**Se preservan deliberadamente** (señales de carga emocional):
- `¡!` — Los signos de exclamación son el marcador más directo de intensidad emocional en español. `"¡Pésimo servicio!"` tiene mayor carga que `"Pésimo servicio."`.
- `¿?` — Las preguntas retóricas son frecuentes en quejas: `"¿Para qué sirve el IMSS si no tienen nada?"`.
- Negaciones (`no + verbo`) — El espacio entre `no` y la palabra siguiente se preserva explícitamente. Un regex agresivo que eliminara puntuación podría romper `"no había camas"` en tokens desconectados, invirtiendo el sentido semántico.
- Comas y puntos — Útiles para segmentación de oraciones en análisis downstream.

---

### 4. Deduplicación residual por hash MD5

El scraper ya aplica deduplicación por `post_id` durante la recolección. Sin embargo, pueden existir duplicados **a nivel de texto** que el scraper no detecta:

- Respuestas automáticas con el mismo texto en múltiples hilos
- Copypaste de la misma queja en distintos posts
- Comentarios con el mismo contenido pero `post_id` diferente

El limpiador aplica una segunda capa de deduplicación usando hash MD5 sobre el texto normalizado (en minúsculas, espacios colapsados):

```python
def hash_texto(texto: str) -> str:
    return hashlib.md5(
        " ".join(texto.lower().split()).encode("utf-8")
    ).hexdigest()
```

El hash se calcula **después** de limpiar el texto para que variantes con diferente espaciado o capitalización sean correctamente identificadas como duplicados.

**Ciclo de vida del set de hashes:**
```
Inicio → hashes_vistos = set()  (vacío)
  ↓
Registro N: hash calculado → ¿ya está en hashes_vistos?
  ├── SÍ  → se descarta (duplicado_texto += 1)
  └── NO  → se agrega al set y al corpus limpio
  ↓
Fin → hashes_vistos se destruye (no se persiste en disco)
```

---

### 5. Carga adaptativa: RAM vs. Streaming

La función `cargar_registros()` toma una decisión automática al arrancar:

```
¿Tamaño del archivo > UMBRAL_STREAMING_MB (50 MB)?
  ├── SÍ y ijson disponible   → streaming con ijson.items()
  │     Memoria: O(1) — constante, independiente del tamaño del archivo
  ├── SÍ e ijson NO disponible → json.load() con advertencia
  │     Memoria: O(n) — todo el archivo en RAM
  └── NO  → json.load() estándar
        Memoria: O(n) — aceptable para archivos pequeños
```

El streaming con `ijson` parsea el JSON token a token, extrayendo solo los elementos bajo la clave `datos.item` (la estructura generada por el scraper). Esto permite procesar archivos de varios GB en entornos con RAM limitada, como laptops de investigación.

---

## 📊 Esquema del dataset de salida

El JSON de salida tiene dos claves de primer nivel: `meta_limpieza` y `datos`.

### `meta_limpieza` — Reporte de la ejecución

```json
{
  "meta_limpieza": {
    "generado":              "2025-04-01T15:30:00.000000",
    "version_script":        "3.0",
    "archivo_fuente":        "/ruta/dataset_influenza_crudo_final.json",
    "emoji_convertido":      true,
    "carga_streaming":       false,
    "registros_entrada":     12540,
    "registros_salida":      4821,
    "registros_eliminados":  7719,
    "tasa_retencion_pct":    38.44,
    "relevancia_alta":       3102,
    "relevancia_media":      1719,
    "detalle_eliminados": {
      "sin_comentario":      203,
      "autor_bot_eliminado": 87,
      "texto_muy_corto":     1544,
      "score_bajo":          312,
      "no_relevante":        5501,
      "duplicado_texto":     72
    },
    "distribucion_categorias": {
      "desabasto":    1204,
      "tiempo_espera": 987,
      "maltrato":      756,
      "saturacion":    431,
      "impacto_salud": 389,
      "pago_privado":  302,
      "negligencia":   298,
      "corrupcion":    201,
      "recorte":        88
    }
  }
}
```

### `datos` — Registros limpios

Cada elemento del array `datos` corresponde a **un comentario** con la siguiente estructura:

```json
{
  "post_id":           "abc123",
  "subreddit":         "Mexico",
  "titulo_post":       "¿Alguien ha tenido problemas con el IMSS?",
  "comentario_raw":    "Me tardaron 4 horaaaas y al final k no había tamiflu 😡",
  "comentario":        "Me tardaron 4 horaaaas y al final k no había tamiflu :cara_enojada:",
  "comentario_norm":   "Me tardaron 4 horas y al final que no había tamiflu cara enojada",
  "relevancia":        "alta",
  "palabras_clave_activas": {
    "sustantivos_activos": ["imss", "tamiflu"],
    "indicadores_activos": ["tardaron", "no había"],
    "categorias_activas":  ["tiempo_espera", "desabasto"]
  },
  "autor":             "usuario_anonimo",
  "score_comentario":  34,
  "score_post":        245,
  "num_comentarios":   87,
  "profundidad":       2,
  "comentario_utc":    1712001500,
  "post_creado_utc":   1712000000,
  "query_origen":      "influenza IMSS desabasto",
  "url_post":          "https://www.reddit.com/r/Mexico/comments/..."
}
```

> **`comentario_raw`**: texto original extraído del scraper, sin ninguna modificación.
> **`comentario`**: texto con emojis convertidos, HTML/URLs/Markdown eliminados, signos y negaciones preservados.
> **`comentario_norm`**: versión con jerga normalizada, usada internamente para clasificación. Incluida en el output para debugging y reproducibilidad.

---

## 📈 Flujo de ejecución

```
pedir_archivos()  →  selección interactiva del JSON de entrada
      │
      ▼
cargar_registros()
  ├── ¿>50 MB y ijson disponible? → streaming O(1) memoria
  └── caso contrario             → json.load() en RAM
      │
      ▼
┌─── Bucle por registro ────────────────────────────────────────────┐
│                                                                   │
│  [2] ¿comentario vacío?         → descartar                       │
│  [3] ¿autor es bot/eliminado?   → descartar                       │
│  [4] emojis_a_texto()           → 😡 → :cara_enojada:            │
│  [5] limpiar_texto()            → strip HTML/URLs/MD              │
│       └── preserva ¡!¿? y "no + verbo"                           │
│  [6] ¿len < MIN_CHARS?          → descartar                       │
│  [7] ¿score < MIN_SCORE?        → descartar                       │
│  [8] normalizar_texto()         → k→que, pesimooo→pesimo          │
│       └── resultado usado SOLO para clasificar, no se guarda      │
│  [9] clasificar_relevancia()                                      │
│  ┌─ Sistema OR + AND ─────────────────────────────────────────┐   │
│  │  ¿pasa PALABRAS_OR?  → NO → descartar                      │   │
│  │  ¿pasa AND?          → SÍ → relevancia "alta"              │   │
│  │                      → NO → relevancia "media"             │   │
│  │  Categorías semánticas activadas → palabras_clave_activas  │   │
│  └────────────────────────────────────────────────────────────┘   │
│  [10] ┌─ Deduplicación por hash MD5 ───────────────────────────┐  │
│       │  hash(comentario_norm) ∈ hashes_vistos?                │  │
│       │  SÍ → descartar   │   NO → agregar al set y al corpus  │  │
│       └─────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
      │
      ▼
[11] guardar JSON  →  { "meta_limpieza": {...}, "datos": [...] }
      │
      ▼
Reporte en consola con distribución por categoría semántica
```

---

## 🔬 Primeros pasos con el corpus limpio (análisis con pandas)

```python
import pandas as pd
import ast

# Cargar solo el array de datos (ignorar meta_limpieza)
with open("dataset_limpio_YYYYMMDD_HHMMSS.json") as f:
    import json
    raw = json.load(f)

df = pd.DataFrame(raw["datos"])

# 1. Distribución de relevancia
print(df["relevancia"].value_counts())

# 2. Categorías semánticas más frecuentes en el corpus
from collections import Counter
categorias = Counter(
    cat
    for row in df["palabras_clave_activas"]
    for cat in row.get("categorias_activas", [])
)
print(categorias.most_common())

# 3. Comentarios de alta relevancia, ordenados por score
alta = df[df["relevancia"] == "alta"].nlargest(10, "score_comentario")
print(alta[["subreddit", "comentario", "score_comentario"]])

# 4. Analizar solo quejas de desabasto
desabasto = df[df["palabras_clave_activas"].apply(
    lambda x: "desabasto" in x.get("categorias_activas", [])
)]
print(f"Registros de desabasto: {len(desabasto)}")

# 5. Palabras clave que más activaron registros de alta relevancia
indicadores = Counter(
    ind
    for row in df[df["relevancia"] == "alta"]["palabras_clave_activas"]
    for ind in row.get("indicadores_activos", [])
)
print(indicadores.most_common(15))
```

---

## ⚠️ Consideraciones éticas y de calidad del corpus

- El corpus limpio contiene **opiniones ciudadanas reales** sobre el sistema de salud. Cualquier publicación debe anonimizar o agregar los datos para no exponer a usuarios individuales.
- El campo `comentario_raw` preserva el texto original sin modificar, incluyendo potenciales datos sensibles. Evaluar si debe incluirse en publicaciones.
- La tasa de retención típica es del **35–45%** del dataset crudo. Una tasa significativamente menor puede indicar que los parámetros `MIN_CHARS` o `PALABRAS_OR` son demasiado restrictivos para el corpus específico.
- Los registros de relevancia `"media"` no son ruido: representan menciones generales de salud que podrían ser relevantes para análisis de contexto o detección de temas emergentes.

---

## 🔗 Relación con el scraper

Este script es el **segundo paso** del pipeline de dos etapas:

```
scraper_influenza_stealth_v3.py   →   dataset_influenza_crudo_final.json
                                              │
                                              ▼
                              limpiar_dataset_v3.py   →   dataset_limpio.json
                                                               │
                                                               ▼
                                                    Análisis de sentimientos / NLP
```

El dataset de salida del limpiador es directamente consumible por bibliotecas como `transformers`, `spaCy`, `NLTK` o cualquier pipeline de análisis de texto en Python.

---

## 📝 Licencia

Uso académico / investigación. Consultar con el equipo responsable del proyecto antes de redistribuir o publicar datos recolectados.

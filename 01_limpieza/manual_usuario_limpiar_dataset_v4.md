# Manual de Usuario
## Limpiador de Dataset — Salud Pública MX / Influenza `v4`

**Archivo:** `limpiar_dataset_v4.py`  
**Versión del manual:** 1.0  
**Propósito:** Depurar y estructurar el dataset crudo generado por el scraper para dejarlo listo para análisis de sentimientos, detección de temas y vigilancia epidemiológica social.

---

## Índice

1. [Descripción general](#1-descripción-general)
2. [Requisitos e instalación](#2-requisitos-e-instalación)
3. [Ejecución básica](#3-ejecución-básica)
4. [Mensajes en consola y comportamiento esperado](#4-mensajes-en-consola-y-comportamiento-esperado)
5. [El pipeline de limpieza — 11 pasos](#5-el-pipeline-de-limpieza--11-pasos)
6. [Sistema de clasificación de relevancia](#6-sistema-de-clasificación-de-relevancia)
7. [Archivos de salida](#7-archivos-de-salida)
8. [Configuración avanzada](#8-configuración-avanzada)
9. [Exploración rápida del corpus limpio con pandas](#9-exploración-rápida-del-corpus-limpio-con-pandas)
10. [Novedades en v4 respecto a v3](#10-novedades-en-v4-respecto-a-v3)
11. [Lugar en el pipeline general](#11-lugar-en-el-pipeline-general)
12. [Preguntas frecuentes](#12-preguntas-frecuentes)
13. [Consideraciones éticas y de calidad](#13-consideraciones-éticas-y-de-calidad)

---

## 1. Descripción general

Este script recibe el JSON producido por `scraper_influenza_stealth_v3.py` (bruto, con ruido, emojis sin convertir y posibles duplicados residuales) y lo transforma en un corpus estructurado y semánticamente etiquetado.

Cada registro de salida incluye:

- `comentario_raw` — texto original sin ninguna modificación
- `comentario` — texto limpio con emojis convertidos, HTML/URLs/Markdown eliminados
- `comentario_norm` — versión normalizada usada internamente para clasificación
- `palabras_clave_activas` — qué términos exactos justificaron la inclusión del registro

### Novedades en v4

> **Enfoque de v4:** correcciones de rendimiento para archivos ≥ 200 MB. Toda la lógica de limpieza y relevancia de v3 se conserva intacta.

| Fix | Componente | Problema en v3 | Solución en v4 |
|---|---|---|---|
| FIX 1 | Detección de ruta `ijson` | Asumía siempre `"datos.item"`, fallando con arrays crudos | Detección automática leyendo los primeros 8 KB del archivo |
| FIX 2 | `EscritorStreamingJSON` | Los registros se acumulaban en RAM hasta el final | Cada registro se escribe al disco en el momento en que pasa los filtros |
| FIX 3 | `EscritorStreamingJSON` | `json.dump()` final requería todo el dataset en memoria | Serialización incremental: `abrir()` → `escribir()` × N → `cerrar()` |
| FIX 4 | `cargar_registros` | Podía omitir el streaming si `ijson` no estaba instalado | Instala `ijson` automáticamente si falta; siempre usa streaming para archivos grandes |

---

## 2. Requisitos e instalación

**Python 3.10 o superior.**  
El script usa exclusivamente la biblioteca estándar (`re`, `hashlib`, `json`, `pathlib`, `html`, `decimal`). No requiere `requests`.

### Dependencias opcionales — críticas para la funcionalidad completa

```bash
pip install emoji    # Conversión de emojis a texto descriptivo
pip install ijson    # Carga en streaming para datasets grandes (>50 MB)
```

**¿Por qué son importantes?**

`emoji` convierte caracteres como `😡`, `😤`, `🏥` en tokens de texto (`:cara_enojada:`, `:hospital:`). Sin esta conversión, los modelos de análisis de sentimientos entrenados en texto español no reconocen esos símbolos como señales emocionales, lo que subestima sistemáticamente la carga negativa de las quejas.

`ijson` permite leer y escribir el dataset en streaming sin cargarlo completo en RAM. A partir de v4 es prácticamente obligatorio para archivos > 50 MB. Si no está instalado, el script **ofrecerá instalarlo automáticamente** al arrancar.

### Verificar la instalación

```bash
python -c "import emoji, ijson; print('Todo listo')"
```

---

## 3. Ejecución básica

```bash
python3 limpiar_dataset_v4.py
```

El script es completamente interactivo. Al ejecutarse:

1. Muestra los archivos JSON disponibles en la misma carpeta.
2. Solicita el número del archivo a limpiar.
3. Solicita el nombre del archivo de salida (o presiona Enter para usar el nombre sugerido).
4. Si faltan dependencias opcionales, pregunta si instalarlas antes de continuar.

No requiere argumentos en línea de comandos.

> **El archivo de entrada nunca se modifica.** El script siempre escribe en un archivo de salida separado. Si el proceso se interrumpe, los datos de entrada permanecen intactos.

---

## 4. Mensajes en consola y comportamiento esperado

| Lo que ves en consola | Causa | ¿Normal? |
|---|---|---|
| `🔍 Ruta detectada para streaming: 'datos.item'` | Detección automática de estructura JSON con wrapper | ✅ |
| `🔍 Ruta detectada para streaming: 'item'` | El archivo de entrada es un array crudo | ✅ |
| `📦 Archivo grande (XXX MB) — streaming con ijson.` | Dataset supera 50 MB | ✅ |
| `⚠️ Archivo grande — ijson no disponible, cargando en RAM` | `ijson` no instalado | ⚠️ Instalar recomendado |
| `Streaming escritura (salida): ✅ activo (sin acumulación en RAM)` | Escritura incremental activa | ✅ |
| `⚠️ Librerías opcionales no encontradas: emoji, ijson` | Primera ejecución sin dependencias | ✅ Escribe `s` para instalar |
| Barra con bloques `█` en el reporte final | Distribución visual de categorías semánticas | ✅ |

---

## 5. El pipeline de limpieza — 11 pasos

El script aplica los pasos en orden estricto. El orden importa: la conversión de emojis y la normalización deben ocurrir **antes** de la clasificación para que los filtros léxicos capturen variantes coloquiales.

```
[1]  Cargar JSON
       ↳ Streaming con ijson para archivos >50 MB
       ↳ json.load() estándar para archivos pequeños

[2]  Eliminar registros sin comentario o con campos vacíos

[3]  Filtrar bots y cuentas eliminadas
       ↳ Lista AUTORES_DESCARTADOS (8 entradas conocidas)

[4]  Convertir emojis a texto descriptivo
       ↳ 😡 → :cara_enojada:   🏥 → :hospital:

[5]  Limpiar texto
       ↳ Elimina HTML, URLs y formato Markdown
       ↳ Preserva signos ¡!¿? y negaciones ("no + verbo")

[6]  Filtrar por longitud mínima
       ↳ Descarta comentarios con menos de MIN_CHARS = 30 caracteres

[7]  Filtrar por score mínimo
       ↳ Descarta comentarios con score < MIN_SCORE_COMENTARIO = -5

[8]  Normalización ligera (lematizador de jerga digital)
       ↳ Colapsa énfasis: "pesimooooo" → "pesimo"
       ↳ Sustituye abreviaturas: k→que, xq→porque, tb→también
       ↳ El texto normalizado se usa SOLO para clasificar; no se guarda en el output

[9]  Clasificación de relevancia (sistema OR + AND)
       ↳ Ver sección 6 para detalle

[10] Deduplicación residual por hash MD5
       ↳ hash(comentario_norm) — si ya fue visto, se descarta

[11] Guardar registro en disco + reporte final en consola
```

---

## 6. Sistema de clasificación de relevancia

La función `clasificar_relevancia()` aplica dos filtros en cascada sobre el texto normalizado.

### Nivel 1 — Filtro OR (red amplia)

El comentario pasa si contiene **al menos una** de las ~80 palabras en `PALABRAS_OR`. Este filtro descarta bromas, memes y conversaciones sin ninguna mención de salud. Un comentario que solo pasa este nivel recibe relevancia `"media"`.

### Nivel 2 — Filtro AND (relevancia alta)

Un comentario sube a `"alta"` si, además del filtro OR, cumple **ambas** condiciones simultáneamente:

- Contiene al menos un término de `SUSTANTIVOS_SALUD` (entidad del sistema: IMSS, ISSSTE, clínica, urgencias, etc.)
- Contiene al menos un término de `INDICADORES_PROBLEMA` (señal de falla: desabasto, tardaron, no había, negligencia, etc.)

### Trazabilidad semántica

Independientemente del nivel de relevancia asignado, el campo `palabras_clave_activas` registra exactamente qué sustantivos, indicadores y categorías activaron la clasificación. Esto permite auditar y reproducir cualquier decisión de inclusión/exclusión.

### Las 9 categorías semánticas

| Categoría | Ejemplos de indicadores |
|---|---|
| `desabasto` | sin medicamento, no había tamiflu, no tenían vacuna |
| `tiempo_espera` | tardaron, horas, fila, no me atendieron |
| `maltrato` | me trató mal, grosero, ignoraron |
| `saturacion` | lleno, sin camas, no hay lugar |
| `impacto_salud` | empeoré, neumonía, UCI |
| `pago_privado` | tuve que pagar, médico privado, de mi bolsillo |
| `negligencia` | mal diagnóstico, error médico |
| `corrupcion` | mordida, me pidieron, cuello |
| `recorte` | cerraron, ya no hay, recortaron |

---

## 7. Archivos de salida

### Nombres de archivo

| Archivo | Descripción |
|---|---|
| `dataset_limpio_YYYYMMDD_HHMMSS.json` | Corpus limpio completo con metadatos |

### Estructura del JSON de salida

El archivo tiene dos secciones de primer nivel:

**`meta_limpieza`** — Metadatos del pipeline:

```json
{
  "meta_limpieza": {
    "generado":             "2024-04-01T14:23:00",
    "version_script":       "4.0",
    "archivo_fuente":       "/ruta/al/dataset_crudo.json",
    "emoji_convertido":     true,
    "carga_streaming":      true,
    "registros_entrada":    12540,
    "registros_salida":     4821,
    "registros_eliminados": 7719,
    "tasa_retencion_pct":   38.44,
    "relevancia_alta":      3102,
    "relevancia_media":     1719,
    "detalle_eliminados": {
      "sin_comentario":      203,
      "autor_bot_eliminado":  87,
      "texto_muy_corto":    1544,
      "score_bajo":          312,
      "no_relevante":       5501,
      "duplicado_texto":      72
    },
    "distribucion_categorias": {
      "desabasto":     1204,
      "tiempo_espera":  987,
      "maltrato":       756,
      "saturacion":     431,
      "impacto_salud":  389,
      "pago_privado":   302,
      "negligencia":    298,
      "corrupcion":     201,
      "recorte":         88
    }
  }
}
```

**`datos`** — Array de registros limpios. Cada elemento corresponde a un comentario:

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

| Campo | Descripción |
|---|---|
| `comentario_raw` | Texto original del scraper, sin ninguna modificación |
| `comentario` | Texto con emojis convertidos, HTML/URLs/Markdown eliminados |
| `comentario_norm` | Versión normalizada para clasificación; incluida para debugging y reproducibilidad |
| `relevancia` | `"alta"` (pasa filtro AND) o `"media"` (solo pasa filtro OR) |
| `palabras_clave_activas` | Sustantivos, indicadores y categorías que activaron la clasificación |

---

## 8. Configuración avanzada

Estas constantes se modifican directamente en el script antes de ejecutarlo:

| Constante | Valor por defecto | Descripción |
|---|---|---|
| `MIN_CHARS` | `30` | Longitud mínima del comentario limpio (caracteres) |
| `MIN_SCORE_COMENTARIO` | `-5` | Score mínimo de Reddit para conservar el comentario |
| `UMBRAL_STREAMING_MB` | `50` | Tamaño a partir del cual se activa el streaming con `ijson` |
| `AUTORES_DESCARTADOS` | 8 entradas | Bots conocidos y cuentas eliminadas |
| `PALABRAS_OR` | ~80 términos | Red amplia de primera criba temática |
| `SUSTANTIVOS_SALUD` | ~30 términos | Entidades del sistema de salud (filtro AND) |
| `INDICADORES_PROBLEMA` | ~50 términos | Señales de falla o queja (filtro AND) |
| `CATEGORIAS_CONTEXTO` | 9 categorías | Etiquetas semánticas para trazabilidad |

### Ajuste recomendado según tu objetivo

| Objetivo | Parámetro a modificar | Valor sugerido |
|---|---|---|
| Corpus más amplio (menor precisión) | `MIN_CHARS` | `15` |
| Corpus más estricto (mayor precisión) | `MIN_CHARS` | `50` |
| Incluir comentarios muy negativos | `MIN_SCORE_COMENTARIO` | `-20` |
| Excluir menciones superficiales | Ampliar `SUSTANTIVOS_SALUD` | Agregar términos específicos |
| Activar streaming antes | `UMBRAL_STREAMING_MB` | `20` |

---

## 9. Exploración rápida del corpus limpio con pandas

```python
import pandas as pd
import json
from collections import Counter

# Cargar el dataset (ignorar meta_limpieza, usar solo el array de datos)
with open("dataset_limpio_YYYYMMDD_HHMMSS.json") as f:
    raw = json.load(f)

df = pd.DataFrame(raw["datos"])

# 1. Distribución de niveles de relevancia
print(df["relevancia"].value_counts())

# 2. Categorías semánticas más frecuentes en el corpus
categorias = Counter(
    cat
    for row in df["palabras_clave_activas"]
    for cat in row.get("categorias_activas", [])
)
print(categorias.most_common())

# 3. Comentarios de alta relevancia ordenados por score
alta = df[df["relevancia"] == "alta"].nlargest(10, "score_comentario")
print(alta[["subreddit", "comentario", "score_comentario"]])

# 4. Filtrar solo quejas de desabasto
desabasto = df[df["palabras_clave_activas"].apply(
    lambda x: "desabasto" in x.get("categorias_activas", [])
)]
print(f"Registros de desabasto: {len(desabasto)}")

# 5. Indicadores que más activaron registros de alta relevancia
indicadores = Counter(
    ind
    for row in df[df["relevancia"] == "alta"]["palabras_clave_activas"]
    for ind in row.get("indicadores_activos", [])
)
print(indicadores.most_common(15))

# 6. Tasa de retención del pipeline (desde metadatos)
meta = raw["meta_limpieza"]
print(f"Registros entrada: {meta['registros_entrada']}")
print(f"Registros salida:  {meta['registros_salida']}")
print(f"Tasa de retención: {meta['tasa_retencion_pct']:.1f}%")
```

> **Nota:** Sustituye el nombre del archivo por el timestamp real generado por el script.

---

## 10. Novedades en v4 respecto a v3

### Escritura incremental (`EscritorStreamingJSON`)

En v3, todos los registros limpios se acumulaban en una lista en RAM y se volcaban al disco con un único `json.dump()` al finalizar. Con datasets de cientos de miles de registros, esto duplicaba el uso de memoria.

En v4, cada registro se serializa y escribe directamente al archivo en el momento en que pasa todos los filtros. El ciclo de vida es:

```
abrir(meta_placeholder)   → abre el archivo, escribe cabecera y meta inicial
escribir(registro) × N    → cada registro se serializa e imprime al instante
cerrar(meta_final)        → cierra el array y actualiza meta con conteos definitivos
```

### Detección automática de estructura JSON (`_detectar_ruta_ijson`)

En v3, el parser `ijson` asumía siempre la ruta `"datos.item"`, fallando silenciosamente con arrays crudos u otras estructuras. En v4, el script lee los primeros 8 KB del archivo para inferir la ruta correcta:

| Estructura del JSON de entrada | Ruta ijson detectada |
|---|---|
| `[{...}, ...]` (array crudo) | `"item"` |
| `{"datos": [{...}]}` | `"datos.item"` |
| `{"registros": [{...}]}` | `"registros.item"` |
| `{"cualquier_clave": [...]}` | `"<clave>.item"` |

---

## 11. Lugar en el pipeline general

Este script es el **segundo paso** del pipeline de datos del proyecto:

```
scraper_influenza_stealth_v3.py
          │
          ▼
dataset_influenza_crudo_final.json
          │
          ▼
limpiar_dataset_v4.py
          │
          ▼
dataset_limpio.json
          │
          ▼
Análisis de sentimientos / NLP
(transformers, spaCy, NLTK, sklearn...)
```

El dataset de salida es directamente consumible por cualquier pipeline de análisis de texto en Python.

---

## 12. Preguntas frecuentes

**¿Qué pasa si la tasa de retención es muy baja (< 20%)?**  
Puede indicar que `MIN_CHARS` o `PALABRAS_OR` son demasiado restrictivos para el corpus específico. Revisa el campo `detalle_eliminados` en `meta_limpieza` para identificar qué paso descarta más registros y ajusta la constante correspondiente.

**¿Los registros de relevancia `"media"` son ruido?**  
No necesariamente. Representan menciones generales de salud que pueden ser útiles para análisis de contexto o detección de temas emergentes. Se recomienda conservarlos y filtrar por `relevancia` según el análisis que se realice.

**¿Puedo usar el `comentario_norm` para entrenar modelos?**  
No es recomendable directamente. `comentario_norm` está diseñado para clasificación léxica interna, no como input de entrenamiento. Para NLP usa `comentario` (emojis como texto, sin HTML/URLs) o `comentario_raw` si necesitas el texto original.

**¿Por qué se preservan los signos ¡!¿? en la limpieza?**  
Son marcadores de intensidad emocional relevantes para análisis de sentimientos en español. Un `"¡Pesimo servicio!"` tiene carga diferente a `"Pesimo servicio"`.

**¿Qué son los duplicados residuales?**  
Son comentarios con texto prácticamente idéntico que el scraper pudo recolectar desde queries distintas aunque el `post_id` fuera diferente (p. ej. comentarios copiados o respuestas formulaicas). El hash MD5 sobre `comentario_norm` los detecta y descarta.

**¿El script puede procesar el JSON de checkpoints del scraper (no solo el final)?**  
Sí. Cualquier archivo JSON producido por el scraper (checkpoint o final) tiene el mismo esquema y puede usarse como entrada.

---

## 13. Consideraciones éticas y de calidad

- El corpus limpio contiene **opiniones ciudadanas reales** sobre el sistema de salud. Cualquier publicación debe anonimizar o agregar los datos para no exponer a usuarios individuales.
- El campo `comentario_raw` preserva el texto original sin modificar, incluyendo potenciales datos sensibles. Evaluar si debe incluirse en publicaciones académicas o reportes.
- La tasa de retención típica es del **35–45%** del dataset crudo. Una tasa significativamente menor puede indicar parámetros demasiado restrictivos; una tasa muy alta puede indicar baja precisión temática.
- Los datos son de carácter público pero fueron producidos en un contexto social específico. El análisis debe contextualizarse adecuadamente en reportes y publicaciones.
- Consultar con el equipo responsable del proyecto antes de redistribuir o publicar datos recolectados.

---

*Manual generado a partir de `manual_tecnico_limpiar_dataset_v4.md` — Proyecto "Burocracia del Dolor"*

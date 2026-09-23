# Manual de Usuario
## Scraper de Opinión Ciudadana — Salud Pública MX / Influenza `v3`

**Archivo:** `scraper_influenza_stealth_v3.py`  
**Versión del manual:** 1.0  
**Propósito:** Minería de opiniones en Reddit para investigación académica sobre la percepción ciudadana de la atención médica ante la influenza en México.

---

## Índice

1. [Descripción general](#1-descripción-general)
2. [Requisitos e instalación](#2-requisitos-e-instalación)
3. [Ejecución básica](#3-ejecución-básica)
4. [Interacción durante la ejecución](#4-interacción-durante-la-ejecución)
5. [Mensajes en consola y comportamiento esperado](#5-mensajes-en-consola-y-comportamiento-esperado)
6. [Interrupciones y pérdida de datos](#6-interrupciones-y-pérdida-de-datos)
7. [Archivos de salida](#7-archivos-de-salida)
8. [Configuración avanzada](#8-configuración-avanzada)
9. [Exploración rápida del dataset con pandas](#9-exploración-rápida-del-dataset-con-pandas)
10. [Preguntas frecuentes](#10-preguntas-frecuentes)
11. [Consideraciones éticas y legales](#11-consideraciones-éticas-y-legales)

---

## 1. Descripción general

Este script recopila posts y comentarios de subreddits mexicanos (p. ej. `r/Mexico`, `r/CDMX`) usando la API pública de Reddit en formato JSON. Combina automáticamente términos de **enfermedad**, **institución** y **problema** para construir un corpus de opinión ciudadana sobre el sistema de salud público.

El corpus resultante está diseñado para tareas de análisis de sentimiento, detección de temas y vigilancia epidemiológica social.

### Novedades en v3

| Mejora | Detalle |
|---|---|
| Soporte de emojis | Queries y dataset JSON preservan emojis (`ensure_ascii=False`) |
| Diccionarios ampliados | Nuevos términos en enfermedad, institución y problema |
| Emojis como señal emocional | Términos con emojis capturan posts de queja y frustración |
| Bloques de 100 queries | Procesamiento interactivo en bloques (`BLOQUE_QUERIES = 100`) |
| Modo interactivo | Pregunta al usuario si continuar tras cada bloque |
| Acumulación entre bloques | El dataset crece sin perder datos de bloques anteriores |
| Checkpoints automáticos | Guarda un JSON de respaldo al finalizar cada bloque |

---

## 2. Requisitos e instalación

**Python 3.10 o superior.**

### Dependencia obligatoria

```bash
pip install requests
```

### Dependencias opcionales — críticas para el sigilo

```bash
pip install brotli       # Soporte de codificación br (Brotli)
pip install zstandard    # Soporte de codificación zstd
```

> **¿Por qué importan?** Chrome y Firefox modernos declaran soporte para `br` y `zstd` en cada petición. Sin estas librerías, el tráfico del script se verá diferente al de un navegador real, lo que puede delatar al scraper ante los sistemas anti-bot de Reddit. Instalarlas hace que la huella HTTP sea **indistinguible de Chrome 122+**.

### Verificar la instalación

```bash
python -c "import requests, brotli, zstandard; print('Todo listo')"
```

---

## 3. Ejecución básica

```bash
python scraper_influenza_stealth_v3.py
```

No requiere argumentos. Al ejecutarse, el script:

1. Construye el pool completo de queries (producto cartesiano de los tres diccionarios).
2. Mezcla las queries en orden aleatorio para garantizar representatividad estadística.
3. Las divide en bloques de 100.
4. Comienza a procesar el primer bloque mostrando el progreso en tiempo real.

---

## 4. Interacción durante la ejecución

Al completar cada bloque de 100 queries, el script muestra el siguiente menú y **espera tu respuesta**:

```
╔══════════════════════════════════════════════════════╗
║  ✅ BLOQUE 01 COMPLETADO — 100 queries procesadas    ║
║  📊 Comentarios acumulados hasta ahora: 842          ║
╠══════════════════════════════════════════════════════╣
║  ¿Deseas continuar con las siguientes 100 búsquedas? ║
║    [S] Sí, continuar con otras 100 búsquedas         ║
║    [N] No, guardar dataset y finalizar               ║
╚══════════════════════════════════════════════════════╝
```

| Tecla | Acción |
|---|---|
| `S` + Enter | Procesa las siguientes 100 queries y muestra el menú de nuevo al terminar |
| `N` + Enter | Guarda el dataset consolidado y termina el script de forma limpia |

Puedes ejecutar tantos bloques como necesites. El dataset se acumula sin perder datos de bloques previos.

---

## 5. Mensajes en consola y comportamiento esperado

El script incluye **pausas deliberadas** que son parte del diseño para imitar comportamiento humano. La siguiente tabla describe cada comportamiento que puede parecer anormal:

| Lo que ves en consola | Causa | Duración aproximada |
|---|---|---|
| Silencio total, sin output | Pausa larga periódica (cada 20 requests) | 45–90 segundos |
| `[429] Rate limit. Pausando Xs...` | Reddit limitó la tasa de peticiones | 90–120 segundos |
| `[⏸] Pausa de Xs (simulando comportamiento humano)...` | Pausa larga programada | 45–90 segundos |
| `[Timeout] Intento N. Esperando Xs...` | Problema de red temporal | 4–32 segundos |

> **No cierres la terminal durante estas pausas.** El script retomará automáticamente. Solo interviene si aparece el mensaje `[FAIL] Abandonando tras N intentos`, que indica un fallo de red persistente que requiere reiniciar el script.

---

## 6. Interrupciones y pérdida de datos

| Situación | Consecuencia |
|---|---|
| Presionar `N` en el menú interactivo | ✅ **Seguro** — guarda el dataset antes de salir |
| Presionar `Ctrl+C` durante el menú interactivo | ✅ **Seguro** — el script detecta `EOFError` y finaliza limpiamente |
| Presionar `Ctrl+C` durante el scraping activo | ⚠️ Se pierden los datos del bloque actual; los checkpoints anteriores están intactos |
| Cerrar la terminal inesperadamente | ⚠️ Se pierden los datos del bloque actual; los checkpoints anteriores están intactos |

**Recomendación:** si necesitas detener el script a mitad de un bloque, espera a que aparezca el menú interactivo y presiona `N`.

---

## 7. Archivos de salida

El script genera los siguientes archivos en el mismo directorio desde donde se ejecuta:

| Archivo | Cuándo se genera | Contenido |
|---|---|---|
| `dataset_influenza_crudo_bloque01_YYYYMMDD_HHMMSS.json` | Al finalizar cada bloque | Checkpoint acumulado hasta ese bloque |
| `dataset_influenza_crudo_final_YYYYMMDD_HHMMSS.json` | Al salir con `N` o al terminar todos los bloques | Dataset consolidado completo |

Los timestamps en el nombre del archivo permiten identificar cuándo se generó cada checkpoint. Si el script se interrumpe de forma inesperada, puedes recuperar los datos del último checkpoint guardado.

### Esquema de cada registro JSON

Cada entrada en el dataset corresponde a **un comentario**:

```json
{
  "bloque":           1,
  "query_origen":     "influenza IMSS desabasto",
  "sort_usado":       "top",
  "subreddit":        "Mexico",
  "post_id":          "abc123",
  "titulo_post":      "¿Alguien ha tenido problemas con el IMSS?",
  "url_post":         "https://www.reddit.com/r/Mexico/comments/...",
  "score_post":       245,
  "num_comentarios":  87,
  "post_creado_utc":  1712000000,
  "comentario":       "Me tardaron 4 horas y al final no había tamiflu 😡",
  "autor":            "usuario_anonimo",
  "score_comentario": 34,
  "profundidad":      2,
  "comentario_utc":   1712001500,
  "fecha_extraccion": "2025-04-01T14:22:10.123456"
}
```

| Campo | Descripción |
|---|---|
| `bloque` | Número de bloque en que se recopiló el registro |
| `query_origen` | Combinación de términos que generó la búsqueda |
| `sort_usado` | Modo de ordenamiento de Reddit (`relevance`, `top`, `new`, `comments`) |
| `subreddit` | Comunidad de origen del post |
| `post_id` | Identificador único del hilo (garantiza deduplicación) |
| `profundidad` | Nivel de anidamiento del comentario (0 = top-level, hasta 5) |
| `fecha_extraccion` | Timestamp ISO 8601 de cuando fue recolectado |

---

## 8. Configuración avanzada

Estas constantes pueden modificarse directamente en el script antes de ejecutarlo:

| Constante | Valor por defecto | Descripción |
|---|---|---|
| `RESULTADOS_POR_QUERY` | `8` | Posts máximos recuperados por búsqueda |
| `MAX_PROFUNDIDAD` | `5` | Niveles de comentarios anidados a extraer |
| `BLOQUE_QUERIES` | `100` | Tamaño del bloque interactivo |
| `PAUSA_LARGA_CADA` | `20` | Requests entre cada pausa larga |
| `PAUSA_LARGA_DURACION` | `(45, 90)` | Rango en segundos de la pausa larga |
| `SORT_MODES` | `relevance, top, new, comments` | Modos de ordenamiento usados |
| `SUBREDDITS` | `Mexico, CDMX, ...` | Comunidades objetivo |

### Ajuste recomendado según tu objetivo

| Objetivo | Parámetro a modificar | Valor sugerido |
|---|---|---|
| Recolección rápida (exploración) | `MAX_PROFUNDIDAD` | `2` |
| Corpus muy amplio | `RESULTADOS_POR_QUERY` | `15–25` |
| Menos riesgo de rate-limit | `PAUSA_LARGA_CADA` | `10` |
| Enfoque en una comunidad específica | `SUBREDDITS` | Lista con un solo subreddit |

---

## 9. Exploración rápida del dataset con pandas

Una vez generado el JSON final, puedes cargarlo y explorar los datos de inmediato:

```python
import pandas as pd
from collections import Counter
import re

# Cargar el dataset
df = pd.read_json("dataset_influenza_crudo_final_YYYYMMDD_HHMMSS.json")

# 1. Total de comentarios únicos recolectados
print(f"Total de registros: {len(df)}")

# 2. Los 5 subreddits con más comentarios
print(df["subreddit"].value_counts().head(5))

# 3. Las 5 queries que generaron más datos
print(df["query_origen"].value_counts().head(5))

# 4. Palabras más frecuentes en los comentarios
todas_las_palabras = " ".join(df["comentario"].dropna()).lower()
palabras = re.findall(r'\b[a-záéíóúñü]{4,}\b', todas_las_palabras)
print(Counter(palabras).most_common(10))

# 5. Comentarios con mayor score (opiniones más valoradas por la comunidad)
print(df.nlargest(5, "score_comentario")[["titulo_post", "comentario", "score_comentario"]])

# 6. Distribución de profundidad de comentarios
print(df["profundidad"].value_counts().sort_index())
```

> **Nota:** Sustituye el nombre del archivo por el timestamp real generado por el script.

---

## 10. Preguntas frecuentes

**¿Por qué el script tarda tanto entre requests?**  
Las pausas son intencionales. El script imita el comportamiento de un usuario humano usando distribuciones gaussianas para los tiempos de espera. Deshabilitarlas puede resultar en bloqueo por parte de Reddit (error 429).

**¿Puedo ejecutar varias instancias en paralelo?**  
No se recomienda. La API pública de Reddit tiene límites de tasa. Múltiples instancias simultáneas aumentan el riesgo de bloqueo.

**¿Se pueden duplicar comentarios en el dataset?**  
No. El conjunto `posts_vistos` persiste durante toda la ejecución (incluyendo entre bloques) y garantiza que cada `post_id` aparezca como máximo una vez en el JSON final.

**¿Qué pasa si el script se corta a mitad de un bloque?**  
Los datos del bloque en curso se pierden. Sin embargo, todos los checkpoints de bloques anteriores están guardados en disco y son válidos. Puedes combinar los JSON de checkpoint manualmente si es necesario.

**¿Por qué los bloques tienen queries mezcladas aleatoriamente?**  
Para que cualquier subconjunto de bloques sea una muestra representativa del espacio total de búsqueda. Si el scraping se interrumpe, el corpus parcial no estará sesgado hacia ninguna institución o término en particular.

**¿Requiere credenciales o cuenta de Reddit?**  
No. Accede únicamente a la API JSON pública de Reddit, sin autenticación.

---

## 11. Consideraciones éticas y legales

- Este script tiene **propósito académico**: minería de opinión sobre salud pública en México.
- Accede únicamente a datos **públicos** de Reddit vía su API JSON no autenticada.
- No almacena contraseñas, tokens ni datos privados de usuarios.
- Se recomienda revisar los [Términos de Servicio de Reddit](https://www.redditinc.com/policies/user-agreement) antes de un uso extensivo o a gran escala.
- **No deshabilitar las pausas integradas.** Hacerlo viola la política de uso aceptable de Reddit y puede derivar en bloqueo permanente de la IP.
- Consultar con el equipo responsable del proyecto antes de redistribuir o publicar datos recolectados.

---

*Manual generado a partir de `README_scraper.md` — Proyecto "Burocracia del Dolor"*

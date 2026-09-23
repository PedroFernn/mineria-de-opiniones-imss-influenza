# 🏥 Scraper de Opinión Ciudadana — Salud Pública MX / Influenza `v3`

Herramienta de minería de opiniones en Reddit orientada a investigación académica sobre la percepción ciudadana de la respuesta del sector salud público en México ante la influenza.

---

## 📋 Descripción

El script recopila posts y comentarios de subreddits mexicanos (p. ej. `r/Mexico`, `r/CDMX`) usando la API pública de Reddit en formato JSON. Combina términos de enfermedad, instituciones y problemas reportados para construir un corpus de opinión que puede utilizarse en análisis de sentimiento, detección de temas y vigilancia epidemiológica social.

---

## ✨ Novedades en v3

| Mejora | Detalle |
|---|---|
| **Soporte de emojis** | Queries y dataset JSON preservan emojis (`ensure_ascii=False`) |
| **Diccionarios ampliados** | Nuevos términos en enfermedad, institución y problema |
| **Emojis como señal emocional** | Términos con emojis capturan posts de queja/frustración |
| **Bloques de 100 queries** | Procesamiento interactivo en bloques (`BLOQUE_QUERIES = 100`) |
| **Modo interactivo** | Pregunta al usuario si continuar tras cada bloque |
| **Acumulación entre bloques** | El dataset crece sin perder datos de bloques anteriores |
| **Checkpoints automáticos** | Guarda un JSON de respaldo al finalizar cada bloque |

---

## 🛡️ Técnicas anti-detección

- Pool de **13 User-Agents** rotativos (Chrome, Firefox, Safari, Edge, Linux)
- **Sesiones HTTP persistentes** con headers completos de navegador real
- **Delays gaussianos** que imitan el ritmo de lectura humano
- **Orden de queries aleatorizado** en cada ejecución
- **Backoff exponencial con jitter** ante respuestas `429` / `5xx`
- Rotación de `Accept-Language` y viewport hints
- **Pausa larga periódica** (45–90 s cada 20 requests) simulando descanso
- `Accept-Encoding` seguro: detecta `brotli`/`zstd` en runtime
- **Cadena de Referer coherente**: búsqueda → lista → hilo
- Profundidad de comentarios configurable (default: **5 niveles**)

---

## 📂 Estructura del proyecto

```
scraper_influenza_stealth_v3.py   ← Script principal
dataset_influenza_crudo_bloque01_YYYYMMDD_HHMMSS.json  ← Checkpoint por bloque
dataset_influenza_crudo_final_YYYYMMDD_HHMMSS.json     ← Dataset consolidado
```

---

## ⚙️ Requisitos

**Python 3.10+**

### Dependencias obligatorias

```bash
pip install requests
```

### Dependencias opcionales — Pre-requisitos críticos para el sigilo

```bash
pip install brotli       # Soporte br
pip install zstandard    # Soporte zstd
```

> **¿Por qué son importantes?**
> Chrome y Firefox modernos declaran soporte para `br` (Brotli) y `zstd` (Zstandard) en el header `Accept-Encoding` de cada petición. Si tu entorno Python no tiene estas librerías instaladas, el script omitirá esos encodings y el tráfico de red **se verá diferente al de un navegador real**, lo que puede delatar al scraper ante los sistemas anti-bot de Reddit.
>
> Instalarlas hace que la huella HTTP del script sea **indistinguible de la de Chrome 122+**, que es el objetivo central de la técnica stealth.

---

## 🚀 Uso

```bash
python scraper_influenza_stealth_v3.py
```

El script muestra en consola el progreso en tiempo real. Al completar cada bloque de 100 queries pregunta:

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

### 🕐 Qué esperar durante la ejecución

El script incluye pausas deliberadas que **son parte del diseño, no errores**. La siguiente tabla describe cada comportamiento que puede parecer anormal:

| Lo que ves en consola | Causa | Duración aprox. |
|---|---|---|
| Silencio total, sin output | Pausa larga periódica (cada 20 requests) | 45–90 segundos |
| `[429] Rate limit. Pausando Xs...` | Reddit limitó la tasa de peticiones | 90–120 segundos |
| `[⏸] Pausa de Xs (simulando comportamiento humano)...` | Pausa larga programada | 45–90 segundos |
| `[Timeout] Intento N. Esperando Xs...` | Problema de red temporal | 4–32 segundos |

> **No cierres la terminal durante estas pausas.** El script retomará automáticamente. Solo interviene si ves el mensaje de error `[FAIL] Abandonando tras N intentos`, que indica un fallo de red persistente.

### ⚠️ Advertencia: interrupciones y pérdida de datos

| Situación | Consecuencia |
|---|---|
| Presionar `N` en el menú interactivo | ✅ Seguro — guarda el dataset antes de salir |
| Presionar `Ctrl+C` **durante una pausa interactiva** | ✅ Seguro — el script lo detecta (`EOFError`) y finaliza limpiamente |
| Presionar `Ctrl+C` **durante el scraping activo** | ⚠️ Se pierden los datos del bloque actual; los checkpoints anteriores están intactos |
| Cerrar la terminal inesperadamente | ⚠️ Se pierden los datos del bloque actual; los checkpoints anteriores están intactos |

**Recomendación:** si necesitas detener el script a mitad de un bloque, espera a que aparezca el menú interactivo y presiona `N`.

---

## 🔧 Configuración principal

| Constante | Valor por defecto | Descripción |
|---|---|---|
| `RESULTADOS_POR_QUERY` | `8` | Posts máximos por búsqueda |
| `MAX_PROFUNDIDAD` | `5` | Niveles de comentarios anidados |
| `BLOQUE_QUERIES` | `100` | Tamaño del bloque interactivo |
| `PAUSA_LARGA_CADA` | `20` | Requests entre cada pausa larga |
| `PAUSA_LARGA_DURACION` | `(45, 90)` | Rango en segundos de la pausa larga |
| `SORT_MODES` | `relevance, top, new, comments` | Modos de ordenamiento usados |
| `SUBREDDITS` | `Mexico, CDMX, ...` | Comunidades objetivo |

---

## 🗂️ Diccionarios de búsqueda

Las queries se construyen combinando tres dimensiones:

### `TERMINOS_ENFERMEDAD`
Términos técnicos (influenza, H1N1, oseltamivir, vacuna antigripal), coloquialismos mexicanos (gripa, gripaza, calentura) y variantes con emojis (`influenza 🤧`, `vacuna 💉`).

### `TERMINOS_INSTITUCION`
Nombres formales (IMSS, ISSSTE, Secretaría de Salud, SEDESA), apodos ciudadanos ("el seguro", "la raza", "siglo xxi") y variantes con emojis (`IMSS 😤`, `Seguro Social 🏥`).

### `TERMINOS_PROBLEMA`
Desabasto, negligencia, mal trato, saturación, corrupción, falta de vacunas y consecuencias clínicas (neumonía, UCI, fallecimiento), incluyendo variantes con emojis de indignación (`sin medicamento 😡`, `urgencias 🚨 llenas`).

---

## 📊 Esquema del dataset de salida

Cada registro del JSON corresponde a **un comentario** con la siguiente estructura:

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

---

## 📈 Flujo de ejecución

```
construir_queries()
      │
      ▼
Mezcla aleatoria del pool completo
      │
      ▼
┌─── Bloque de 100 queries ────────────────────────────────────┐
│  buscar_posts() → SesionHumana.get()                         │
│       │                                                      │
│       ▼                                                      │
│  ┌─ Deduplicación ──────────────────────────────────────┐    │
│  │  posts_vistos = set()  ← persiste entre bloques      │    │
│  │  Si post_id ya fue procesado → se descarta           │    │
│  │  Garantiza que un post que aparece en múltiples      │    │
│  │  búsquedas (p. ej. "gripa IMSS" y "IMSS desabasto")  │    │
│  │  se descarga y analiza UNA SOLA VEZ                  │    │
│  └──────────────────────────────────────────────────────┘    │
│       │                                                      │
│       ▼                                                      │
│  extraer_hilo() → extraer_comentarios_recursivo()            │
│       │                                                      │
│       ▼                                                      │
│  Acumular en data_final                                      │
│       │                                                      │
│       ▼                                                      │
│  guardar_dataset() → checkpoint_bloque_N.json                │
└──────────────────────────────────────────────────────────────┘
      │
      ▼
¿Continuar? [S/N]
      │
      ▼ (al finalizar)
guardar_dataset() → dataset_final.json
```

---

## 🧠 Arquitectura técnica avanzada

Esta sección documenta la lógica interna del script para investigadores que necesiten evaluar la validez metodológica del corpus recolectado.

---

### 1. Extracción recursiva de comentarios (Depth-First Search)

La función `extraer_comentarios_recursivo` realiza un recorrido en profundidad (DFS) sobre el árbol de comentarios de Reddit, bajando hasta el nivel definido por `MAX_PROFUNDIDAD` (default: **5**).

```
Post (raíz)
 └── Comentario nivel 0  ← top-level comment
      └── Respuesta nivel 1
           └── Respuesta nivel 2  ← "micro-historia": el usuario ya narra
                └── Respuesta nivel 3      una experiencia c
                ompleta
                     └── Respuesta nivel 4 (límite)
```

**¿Por qué importa para la validez del dataset?**

Los comentarios de nivel 0 suelen ser reacciones cortas ("igual me pasó", "+1"). Las narrativas detalladas —*"fui al IMSS, me dijeron que no había tamiflu, me mandaron a otra clínica…"*— aparecen típicamente en los niveles 2 y 3, donde el hilo ya tomó contexto. Capturar 5 niveles garantiza que el corpus contenga estas **micro-historias clínicas**, que son el insumo principal para análisis cualitativo y de sentimiento.

Los comentarios con cuerpo `[deleted]` o `[removed]` se filtran automáticamente antes de añadirlos al dataset.

---

### 2. El algoritmo de la Sesión Humana (`SesionHumana`)

La clase `SesionHumana` es el núcleo del sigilo. Combina cuatro mecanismos independientes que operan en paralelo:

#### 2a. Delays con distribución gaussiana

Los tiempos de espera entre requests **no son fijos**. Se samplea de una distribución normal:

```
μ = (mínimo + máximo) / 2
σ = (máximo - mínimo) / 4
espera = clamp(gauss(μ, σ), mínimo, máximo)
```

Un delay fijo (p. ej. siempre 3 s) crea un patrón rítmico que los firewalls de Reddit detectan como robótico. Una distribución gaussiana produce la variabilidad natural que caracteriza a un humano que lee, piensa y hace clic. Adicionalmente, el 8 % de las peticiones añade una pausa extra aleatoria de 3–8 s, simulando que el usuario se distrajo.

#### 2b. Detección de encodings en tiempo de ejecución

Al iniciar el script, `_codificaciones_soportadas()` detecta qué librerías de compresión están instaladas y construye dinámicamente el valor del header `Accept-Encoding`:

| Librerías instaladas | Header resultante |
|---|---|
| Solo stdlib | `gzip, deflate` |
| + `brotli` | `gzip, deflate, br` |
| + `brotli` + `zstandard` | `gzip, deflate, br, zstd` |

Chrome 122+ declara `gzip, deflate, br, zstd`. Un script Python sin `brotli`/`zstandard` declara solo `gzip, deflate`, lo que es una **firma de bot detectable**. Instalar ambas librerías hace que la huella HTTP del script sea indistinguible de la de un navegador moderno.

#### 2c. Rotación de identidad

Con probabilidad 15 % en cada request, el User-Agent cambia. El Referer sigue la cadena lógica `búsqueda → lista → hilo`, replicando la navegación real de un usuario que llega desde Google, entra al subreddit y luego abre un post.

#### 2d. Pausa larga periódica

Cada 20 requests se dispara una pausa de 45–90 s. Esto simula que el usuario tomó un descanso y reduce la densidad de peticiones en ventanas de tiempo largas, que es el patrón que detectan los sistemas anti-abuso de Reddit.

---

### 3. Estrategia de combinatoria de queries

#### Producto cartesiano (Enfermedad × Institución × Problema)

El pool de queries se genera con `itertools.product` sobre los tres diccionarios:

```python
for enf, inst, prob in iterproduct(TERMINOS_ENFERMEDAD, TERMINOS_INSTITUCION, TERMINOS_PROBLEMA):
    query = f"{enf} {inst} {prob}"
```

Esto produce la **cobertura máxima**: cada combinación posible de término de enfermedad, institución y problema aparece exactamente una vez en el pool, sin huecos ni duplicados intencionales.

#### Aleatorización antes del procesamiento por bloques

Tras generar el pool completo, se aplica `random.shuffle` **una sola vez**, antes de dividirlo en bloques de 100:

```
Pool = [combo_1, combo_2, ..., combo_N]   ← orden determinista (producto cartesiano)
         ↓  random.shuffle()
Pool = [combo_847, combo_3, combo_291, ...]  ← orden aleatorio
         ↓  split en bloques de 100
Bloque 1 = [combo_847, combo_3, ...]
Bloque 2 = [combo_512, combo_91, ...]
```

**¿Por qué importa para la representatividad del corpus?**

Sin shuffle, el primer bloque contendría todas las combinaciones que empiezan con el primer término de enfermedad (`influenza`) y el primer subreddit del pool. Si el scraping se interrumpe tras el bloque 1, el dataset estaría **sesgado** hacia esa combinación. Con shuffle, cada bloque es una muestra aleatoria del espacio total de búsqueda, por lo que cualquier subconjunto de bloques es representativo del corpus completo.

---

### 4. Resiliencia y gestión de errores

#### Backoff exponencial con jitter

Ante errores de servidor (`5xx`) el tiempo de espera crece según $2^n$, donde $n$ es el número de intento, más un jitter aleatorio:

```
intento 0 → espera = 2⁰ × 5 + U(0, 5) = ~5–10 s
intento 1 → espera = 2¹ × 5 + U(0, 5) = ~10–15 s
intento 2 → espera = 2² × 5 + U(0, 5) = ~20–25 s
intento 3 → espera = 2³ × 5 + U(0, 5) = ~40–45 s
```

El jitter $U(0, 5)$ (uniforme) es crítico: sin él, múltiples clientes en backoff sincronizado generan picos de tráfico en exactamente $t$, $2t$, $4t$…, que los servidores detectan como un ataque coordinado. El jitter desincroniza los reintentos.

#### Estado de deduplicación global y persistente

```python
posts_vistos: set[str] = set()   # inicializado UNA sola vez antes del bucle principal
```

Este set se crea antes del primer bloque y **nunca se reinicia entre bloques**. Cuando una query en el bloque 3 devuelve un post que ya fue procesado en el bloque 1, el `post_id` ya está en `posts_vistos` y el post se descarta sin volver a descargarlo ni a duplicarlo en `data_final`.

Esto tiene dos consecuencias directas para la validez del corpus:

- **Sin duplicados de contenido**: un comentario no aparece dos veces en el dataset aunque el post padre haya sido encontrado por 10 queries distintas.
- **Sin requests redundantes**: el script no vuelve a descargar hilos que ya procesó, lo que reduce la carga sobre la API y la exposición del script a los sistemas anti-bot.

---

## 🧠 Arquitectura técnica avanzada

Esta sección documenta la lógica interna del script para investigadores que necesiten evaluar la validez metodológica de los datos recolectados.

---

### 1. Extracción recursiva (Depth-First Search)

La función `extraer_comentarios_recursivo()` realiza un **recorrido en profundidad (DFS)** sobre el árbol de comentarios de cada hilo de Reddit, siguiendo la estructura anidada de la API JSON (`replies → data → children`).

```
Post
 └── Comentario nivel 0  (top-level)
      └── Respuesta nivel 1
           └── Respuesta nivel 2  ← "micro-historia": el contexto real
                └── Respuesta nivel 3
                     └── Respuesta nivel 4  ← MAX_PROFUNDIDAD = 5
```

**¿Por qué importa para la investigación?**
Los comentarios de nivel 0 suelen ser quejas breves. Los niveles 2 y 3 contienen las *micro-historias*: el relato detallado de una experiencia ("me mandaron de urgencias a consulta externa, tardé 6 horas, al final no había oseltamivir…"). Limitar la extracción a top-level comments produciría un corpus superficial; el DFS hasta nivel 5 captura esa riqueza narrativa que justifica el valor cualitativo del dataset.

El parámetro `MAX_PROFUNDIDAD = 5` puede ajustarse: valores menores aceleran la recolección, valores mayores aumentan el volumen pero con rendimientos decrecientes a partir del nivel 4.

---

### 2. El algoritmo de la Sesión Humana

La clase `SesionHumana` es el núcleo del sigilo. Opera en tres capas independientes:

#### Capa 1 — Pausas con distribución gaussiana

Los delays entre peticiones **no son fijos**: siguen una distribución normal $\mathcal{N}(\mu, \sigma)$ donde $\mu$ es el punto medio del rango configurado y $\sigma = \text{rango}/4$.

```python
# Para un rango (2.0, 6.0):
μ = 4.0 s  |  σ = 1.0 s
# ~68% de las pausas caen entre 3 y 5 segundos
# ~5% aleatoriamente reciben +3 a +8 s adicionales (micro-descansos)
```

Un scraper robótico genera peticiones con intervalos fijos o uniformes, lo que produce un **patrón rítmico** detectable por los firewalls de Reddit. La distribución gaussiana imita la variabilidad natural del comportamiento humano (tiempo de lectura, distracción, desplazamiento) y elimina esa firma estadística.

#### Capa 2 — Fingerprint HTTP dinámico

Al arrancar, el script llama a `_codificaciones_soportadas()` para detectar en tiempo de ejecución qué librerías de compresión están disponibles:

| Librerías instaladas | `Accept-Encoding` enviado | Similitud con Chrome 122 |
|---|---|---|
| Solo `requests` | `gzip, deflate` | ❌ Incompleto — detectable |
| + `brotli` | `gzip, deflate, br` | ⚠️ Parcial |
| + `brotli` + `zstandard` | `gzip, deflate, br, zstd` | ✅ Indistinguible |

El header resultante se fija en `ACCEPT_ENCODING_SEGURO` y se inyecta en **cada petición**. Sin las librerías opcionales, la huella HTTP del script se desvía de la de un navegador moderno y puede activar heurísticas anti-bot.

#### Capa 3 — Rotación de identidad estocástica

En cada petición existe una probabilidad del **15 %** de rotar el User-Agent. Esto evita que una sesión larga presente demasiadas peticiones bajo el mismo UA, sin caer en el extremo opuesto de rotar en cada petición (patrón igualmente detectible).

---

### 3. Estrategia de combinatoria y sesgo cero

#### Producto cartesiano completo

Las queries se generan con `itertools.product` sobre las tres dimensiones del corpus:

```
Enfermedad (N₁ términos) × Institución (N₂ términos) × Problema (N₃ términos)
→ N₁ × N₂ × N₃ tripletas de búsqueda
```

Esto garantiza **cobertura exhaustiva**: ninguna combinación relevante queda sin explorar por omisión. A esto se suman pares con emojis generados por un segundo producto cartesiano sobre los términos que contienen caracteres Unicode.

#### Mezcla aleatoria y representatividad estadística

Tras construir el pool completo, se aplica `random.shuffle()` **antes** de dividirlo en bloques. Esta decisión tiene una consecuencia metodológica importante:

> Si el scraping se interrumpe después del bloque 3 de 10, los datos recolectados **no** están sesgados hacia ninguna institución, subreddit o categoría de problema en particular. Son una muestra aleatoria representativa del espacio de búsqueda completo.

Sin el shuffle, el orden sería alfabético/lexicográfico y los primeros bloques estarían dominados por términos que empiezan con las mismas letras, sesgando cualquier análisis exploratorio preliminar.

---

### 4. Resiliencia y gestión de estado

#### Backoff exponencial con jitter

Ante errores de servidor (`5xx`) o límite de tasa (`429`), el tiempo de espera sigue la fórmula:

$$t_{\text{espera}} = 2^n \times k + \mathcal{U}(0, j)$$

donde $n$ es el número de intento, $k$ es una constante base (4–6 s según el tipo de error) y $\mathcal{U}(0, j)$ es un jitter uniforme aleatorio. El jitter es crítico: sin él, múltiples instancias paralelas o reintentos simultáneos generarían un **thundering herd** que volvería a saturar el servidor exactamente al mismo tiempo.

| Intento | Espera base (`5xx`, k=5) | + Jitter máx. | Total máx. |
|---|---|---|---|
| 1 | 5 s | 5 s | ~10 s |
| 2 | 10 s | 5 s | ~15 s |
| 3 | 20 s | 5 s | ~25 s |
| 4 | 40 s | 5 s | ~45 s |

#### Deduplicación global persistente

El conjunto `posts_vistos` es una variable en memoria que **persiste durante toda la ejecución**, incluyendo entre bloques interactivos. Su ciclo de vida es:

```
Inicio del script → posts_vistos = set()  (vacío)
     ↓
Bloque 1: agrega 340 post_id únicos
     ↓
[menú interactivo → usuario presiona S]
     ↓
Bloque 2: encuentra 120 posts → 38 ya están en posts_vistos → se descartan
          agrega 82 nuevos post_id
     ↓
Bloque N: el set crece acumulativamente
     ↓
Fin del script → posts_vistos se destruye (no se persiste en disco)
```

**Implicación para el dataset:** cada `post_id` aparece como máximo una vez en el JSON final, independientemente de cuántas queries distintas lo hayan recuperado. Los comentarios de ese post se registran una única vez, manteniendo la integridad referencial del corpus.

---

## 🔬 Primeros pasos con el dataset (análisis con pandas)

Una vez generado el JSON final, puedes cargarlo y explorar los datos con pocas líneas:

```python
import pandas as pd
from collections import Counter

# Cargar el dataset
df = pd.read_json("dataset_influenza_crudo_final_YYYYMMDD_HHMMSS.json")

# 1. Ver cuántos comentarios únicos se recolectaron
print(f"Total de registros: {len(df)}")

# 2. Los 5 subreddits con más comentarios
print(df["subreddit"].value_counts().head(5))

# 3. Los 5 términos de búsqueda que generaron más datos
print(df["query_origen"].value_counts().head(5))

# 4. Palabras más frecuentes en los comentarios (análisis básico de texto)
from collections import Counter
import re

todas_las_palabras = " ".join(df["comentario"].dropna()).lower()
palabras = re.findall(r'\b[a-záéíóúñü]{4,}\b', todas_las_palabras)
print(Counter(palabras).most_common(10))

# 5. Comentarios con mayor score (opiniones más valoradas por la comunidad)
print(df.nlargest(5, "score_comentario")[["titulo_post", "comentario", "score_comentario"]])
```

---

## ⚠️ Consideraciones éticas y legales

- Este script tiene **propósito académico** (minería de opinión sobre salud pública).
- Accede únicamente a datos **públicos** de Reddit vía su API JSON no autenticada.
- No almacena contraseñas, tokens ni datos privados de usuarios.
- Se recomienda revisar los [Términos de Servicio de Reddit](https://www.redditinc.com/policies/user-agreement) antes de un uso extensivo.
- Respetar los límites de tasa (`429`) y no deshabilitar las pausas integradas.

---

## 📝 Licencia

Uso académico / investigación. Consultar con el equipo responsable del proyecto antes de redistribuir o publicar datos recolectados.

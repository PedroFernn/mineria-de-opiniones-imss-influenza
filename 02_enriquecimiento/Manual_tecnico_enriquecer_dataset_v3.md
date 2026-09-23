# 📊 Enriquecedor de Dataset — Salud Pública MX / Influenza `v2`

Tercera etapa del pipeline de investigación. Recibe el JSON producido por `limpiar_dataset_v3.py` y añade métricas cuantitativas originales orientadas al análisis de la experiencia ciudadana ante el sistema de salud público: polaridad emocional localizada, fricción burocrática, nivel de impacto y detección de sarcasmo.

---

## 📋 Descripción

El enriquecedor no filtra ni elimina registros (salvo los que quedaron vacíos tras la limpieza profunda): su propósito es **añadir dimensiones analíticas** que los modelos genéricos de NLP no pueden capturar sobre el español coloquial mexicano en contexto de salud pública. Cada registro de salida incluye cuatro nuevos grupos de campos que permiten segmentar, ordenar y visualizar el corpus sin necesidad de entrenar un modelo propio.

---

## ✨ Novedades en v2

| # | Mejora | Descripción |
|---|---|---|
| ① | **Lexicon localizado MX** | Diccionario de ~60 términos con pesos de negatividad calibrados para regionalismos ("viacrucis", "vuelta y vuelta", "particular") que los modelos genéricos subestiman |
| ② | **Índice de Fricción Burocrática (IFB)** | Métrica compuesta 0–10 que cuantifica el esfuerzo del ciudadano para navegar el sistema. Seis componentes: tiempo, burocracia, pago privado, rechazo, múltiples intentos e impacto clínico por fricción |
| ③ | **Polaridad con intensidad amplificada** | Score de negatividad que combina el lexicon con 6 capas de amplificación: amplificadores contextuales, signos `¡!`, emojis, letras repetidas, sarcasmo y techo anti-inflación |
| ④ | **Nivel de Impacto en 3 dimensiones** | Clasificación jerárquica ADMINISTRATIVO → ECONÓMICO → CLÍNICO con escala numérica de gravedad 0–3 para ordenar el corpus |

**Heredadas de v1:**

| # | Función | Descripción |
|---|---|---|
| ⑤ | `limpiar_profundo()` | Segunda pasada de limpieza: artefactos `->` de flujos burocráticos, separadores y puntos suspensivos excesivos |
| ⑥ | `reclasificar_relevancia()` | Re-aplica el filtro OR+AND con el vocabulario extendido de v2 |
| ⑦ | `detectar_contextos()` | Etiquetado temático múltiple en 8 categorías |
| ⑧ | `detectar_sarcasmo()` | Detección de ironía mediante 15 patrones regex |
| ⑨ | `utc_a_fecha()` | Conversión UTC → ISO + campos `anio_mes` para series de tiempo |
| ⑩ | `calcular_metricas()` | `num_palabras`, `num_caracteres`, `densidad_keywords` |

---

## 🔄 Pipeline de enriquecimiento

El script aplica **8 pasos secuenciales** por cada registro. El orden importa: la limpieza profunda debe ocurrir antes de las métricas, y la detección de sarcasmo antes del cálculo de polaridad.

```
[1]  Emojis → texto descriptivo  (segunda oportunidad, por si el limpiador no tenía emoji)
 │
[2]  Limpieza profunda residual  (artefactos ->, separadores, puntos suspensivos)
 │
[3]  Descartar si quedó vacío tras limpieza  (MIN_CHARS = 25)
 │
[4]  Re-clasificar relevancia  (alta | media | ninguna → descarta "ninguna")
 │
[5]  Detectar contextos temáticos  (8 categorías, múltiple)
 │
[6]  Detectar sarcasmo  (15 patrones regex compilados)
 │
[7]  Convertir timestamps UTC → fechas ISO legibles
 │
[8]  Calcular las 4 métricas nuevas de v2:
      ├── Índice de Fricción Burocrática (IFB)
      ├── Polaridad localizada con amplificación
      ├── Nivel de Impacto en 3 dimensiones
      └── Métricas básicas de texto
 │
[9]  Acumular registro enriquecido
 │
[10] Guardar JSON + reporte en consola con distribuciones visualizadas
```

---

## 📂 Estructura del proyecto

```
enriquecer_dataset_v3.py                         ← Script principal
dataset_limpio_YYYYMMDD_HHMMSS.json              ← Entrada (output del limpiador)
dataset_enriquecido_YYYYMMDD_HHMMSS.json         ← Salida: corpus enriquecido + metadatos
```

---

## ⚙️ Requisitos

**Python 3.10+**

### Dependencia opcional — Pre-requisito crítico para la señal emocional

```bash
pip install emoji    # Conversión de emojis a tokens de texto
```

> **¿Por qué es importante?**
> El enriquecedor aplica una **segunda conversión de emojis** sobre el campo `comentario`, ya que algunos registros podrían provenir de pipelines anteriores sin esta librería activa. Sin `emoji`, los caracteres Unicode como `😡` `😤` `🏥` no se convierten a tokens de texto como `:cara_enojada:` y no son reconocidos por las capas de amplificación de polaridad ni por los patrones de sarcasmo. El resultado sería una subestimación sistemática de la carga emocional negativa del corpus.
>
> El script detecta la librería en tiempo de ejecución y **ofrece instalarla automáticamente** si no está presente.

---

## 🚀 Uso

```bash
python3 enriquecer_dataset_v3.py
```

El script es completamente interactivo. Muestra los archivos JSON disponibles en la misma carpeta y solicita:

1. El número del archivo a enriquecer (debe ser el output de `limpiar_dataset_v3.py`)
2. El nombre/ruta del archivo de salida (o Enter para usar el nombre sugerido)

### ⚠️ Advertencias de uso

| Situación | Consecuencia |
|---|---|
| Alimentar con el JSON **crudo** (del scraper, sin limpiar) | ✅ Funciona técnicamente, pero el campo `palabras_clave_activas` estará vacío y la re-clasificación de relevancia puede ser imprecisa |
| Alimentar con el JSON **limpio** (output del limpiador) | ✅ Uso correcto — todos los campos se heredan y enriquecen |
| Interrumpir con `Ctrl+C` durante el procesamiento | ⚠️ El archivo de salida no se crea; el archivo de entrada permanece intacto |

---

## 🔧 Configuración principal

| Constante | Valor | Descripción |
|---|---|---|
| `MIN_CHARS_POST_ENRIQUECIMIENTO` | `25` | Longitud mínima tras limpieza profunda para conservar el registro |
| `LEXICON_NEGATIVO_LOCALIZADO` | ~60 términos | Pesos de negatividad 1.0–3.0 para regionalismos MX |
| `AMPLIFICADORES_CONTEXTUALES` | 10 entradas | Multiplicadores 1.1–1.3 para intensificadores ("muy", "jamás", "totalmente") |
| `IFB_MAX` | `10.0` | Techo de la escala del Índice de Fricción Burocrática |
| `EMOJIS_NEGATIVOS_PESO` | `0.4` | Peso que suma cada emoji negativo al score de polaridad |
| `CONTEXTOS` | 8 categorías | Diccionario de etiquetas temáticas múltiples |
| `PATRONES_SARCASMO` | 15 patrones | Regex compilados para detección de ironía |

---

## 🧠 Arquitectura técnica avanzada

Esta sección documenta la lógica matemática y las decisiones de diseño de cada nueva métrica. Es la información clave para evaluar la validez científica de los valores numéricos en el corpus.

---

### 1. Lexicon de negatividad localizado para el contexto MX

El enriquecedor no depende de modelos genéricos de sentimiento (BERT, BETO, SentiWordNet) para su métrica principal. En su lugar usa un lexicon artesanal con pesos calibrados específicamente para el español coloquial mexicano en salud pública.

**Escala de pesos:**

| Peso | Categoría | Ejemplos |
|---|---|---|
| `1.0 – 1.5` | Negativo leve | términos ambivalentes en contexto general |
| `2.0 – 2.5` | Negativo moderado | "desabasto", "maltrato", "negligencia", "particular" |
| `2.8 – 3.0` | Muy negativo | "viacrucis", "murió", "nos dejan solos", "no hay sistema" |

**¿Por qué un lexicon propio en vez de un modelo preentrenado?**

Los modelos entrenados en corpus genéricos (noticias, Wikipedia, Twitter general) tienen puntos ciegos específicos para este dominio:

- `"viacrucis"` → neutro en un modelo general; en Reddit MX de salud = agotamiento total ante el sistema
- `"vuelta y vuelta"` → ausente en cualquier corpus estándar; es una expresión puramente mexicana de frustración burocrática repetida
- `"particular"` → puede ser positivo ("atención particular = personalizada") en otros contextos; aquí siempre significa que el sistema público falló y la persona tuvo que pagar
- `"desmadre"`, `"chingadera"` → jerga coloquial con alta carga negativa que los modelos BERT/BETO típicamente subestiman

El lexicon actúa como **primera capa** de la polaridad. No reemplaza un modelo de NLP sino que complementa su punto ciego contextual.

---

### 2. Índice de Fricción Burocrática (IFB)

El IFB es una métrica compuesta original que cuantifica el **esfuerzo total que el ciudadano tuvo que invertir** para navegar el sistema de salud. Va más allá de detectar si hubo una queja: mide la magnitud del obstáculo sistémico.

#### Los 6 componentes y sus pesos base

| Componente | Peso base | Señal que mide |
|---|---|---|
| `tiempo_espera` | 1.5 | Filas, horas de espera, citas en meses |
| `burocracia_proceso` | 1.5 | Trámites, viacrucis, sistema caído, formularios |
| `pago_privado` | 2.0 | Tuvo que salir del sistema y pagar — señal de falla total |
| `rechazo_negacion` | 2.0 | Le negaron atención o medicamento explícitamente |
| `multiples_intentos` | 1.5 | Fue más de una vez: "volví", "de nuevo", "otra vez" |
| `impacto_salud_por_friccion` | 3.0 | La fricción causó daño clínico — peso máximo |

#### Fórmula de cálculo

```
Para cada componente C que tenga ≥ 1 término detectado:
  aporte(C) = peso_base(C) × (1 + (n_términos - 1) × 0.2)

IFB_score = min( Σ aporte(C), IFB_MAX=10.0 )
```

El factor `0.2` por término adicional premia la especificidad: un comentario que menciona "fila", "horas de espera" **y** "sala de espera" expresa más fricción que uno que solo menciona "fila". Sin embargo, el peso base se gana con el primero para evitar que textos muy largos inflen artificialmente el score.

#### Escala de niveles

| Score | Nivel | Interpretación |
|---|---|---|
| `0.0` | `ninguno` | Sin señales de fricción detectadas |
| `0.1 – 2.4` | `bajo` | Fricción puntual y resuelta |
| `2.5 – 4.9` | `medio` | Varios obstáculos; experiencia negativa pero sin consecuencias graves |
| `5.0 – 7.4` | `alto` | Múltiples componentes activados; posible pago privado o rechazo |
| `7.5 – 10.0` | `crítico` | Fricción extrema con impacto en salud o combinación pago+rechazo+tiempo |

Un comentario que activa `pago_privado (2.0)` + `rechazo_negacion (2.0)` + `impacto_salud_por_friccion (3.0)` alcanza 7.0 antes de normalizar, llegando directamente a nivel `alto` casi `crítico`.

---

### 3. Polaridad localizada con 6 capas de amplificación

La función `calcular_polaridad_localizada()` construye el score de negatividad de forma **aditiva y multiplicativa en capas**. Cada capa captura una señal diferente del lenguaje emocional digital mexicano.

```
Score inicial = 0.0
     │
     ▼  Capa 1 — Lexicon localizado
     +  Suma peso(término) por cada término del LEXICON_NEGATIVO_LOCALIZADO presente
     │
     ▼  Capa 2 — Amplificadores contextuales
     ×  Multiplica por el factor del amplificador más fuerte presente
     │   (toma el máximo, no acumula exponencialmente)
     │   "muy" → ×1.1 | "totalmente" → ×1.2 | "jamás" → ×1.3
     │
     ▼  Capa 3 — Signos de exclamación ¡!
     ×  factor_excl = min(1 + n_signos × 0.1, 1.5)  ← techo en +50%
     │   "¡¡¡Pésimo!!!" (6 signos) → ×1.5 (máximo)
     │
     ▼  Capa 4 — Emojis negativos
     +  +0.4 por cada emoji negativo detectado (😡 😤 🤬 😢 etc.)
     │
     ▼  Capa 5 — Letras repetidas residuales
     ×  ×1.1 si se detecta patrón (.)\1{2,}  (énfasis emocional no normalizado)
     │   "pésimoooo" que sobrevivió al limpiador → señal de intensidad extra
     │
     ▼  Capa 6 — Ajuste por sarcasmo
     ×  ×1.3 si es_sarcasmo=True y score > 0
         Justificación: el sarcasmo oculta negatividad mayor que las palabras
         literales expresan. "¡Excelente servicio! Me tardaron 5 horas"
         tiene score literal bajo pero carga real alta.
```

#### Escala de niveles de polaridad

| Score | Nivel |
|---|---|
| `0.0` | `neutro` |
| `0.1 – 1.9` | `leve` |
| `2.0 – 4.9` | `moderado` |
| `5.0 – 8.9` | `alto` |
| `≥ 9.0` | `muy_alto` |

**Nota metodológica:** este score es deliberadamente simple. No reemplaza un modelo de análisis de sentimientos de propósito general (BERT/BETO/RoBERTa), sino que captura la carga negativa del español coloquial mexicano en salud pública mejor que cualquier modelo entrenado en corpus genérico. Ambos pueden complementarse: usar `polaridad_nivel` como feature adicional en el entrenamiento fino de un clasificador.

---

### 4. Nivel de Impacto en 3 dimensiones

La función `clasificar_nivel_impacto()` opera sobre una jerarquía de gravedad creciente. Un comentario puede activar múltiples dimensiones simultáneamente; el campo `nivel_maximo` siempre refleja la peor consecuencia detectada.

```
ADMINISTRATIVO  (escala = 1)
  Fricción pura: citas, trámites, viacrucis, maltrato, sistema caído
  El ciudadano sufrió obstáculos pero sin consecuencias físicas ni económicas
        │
        ▼ (superset de gravedad)
ECONÓMICO  (escala = 2)
  Pago de bolsillo, médico particular, farmacia privada, deuda
  El sistema falló y la persona tuvo que costear la atención por su cuenta
        │
        ▼ (superset de gravedad)
CLÍNICO  (escala = 3)
  Complicaciones por la demora, hospitalización, UCI, fallecimiento, secuelas
  La falla del sistema tuvo consecuencias directas sobre la salud
```

La escala numérica `escala_gravedad` (0–3) permite ordenar el corpus de menos a más grave para análisis comparativos y visualizaciones de escalada de crisis. Por ejemplo: filtrar solo registros con `escala_gravedad >= 2` produce los casos donde el sistema fallido tuvo consecuencias económicas o clínicas concretas.

**Un comentario puede tener múltiples dimensiones activas:**
```json
"impacto_dimensiones": ["administrativo", "economico", "clinico"],
"impacto_nivel_max":   "clinico",
"impacto_escala":      3
```

---

### 5. Detección de sarcasmo

La función `detectar_sarcasmo()` usa 15 patrones regex compilados en un solo objeto `re.Pattern` para eficiencia. Los patrones capturan dos familias de ironía frecuentes en Reddit MX de salud:

**Familia 1 — Expresiones de incredulidad fingida:**
- `"qué raro"`, `"vaya sorpresa"`, `"no me digas"`, `"¿en serio?"` → indignación expresada como sorpresa
- `"funciona perfecto"`, `"excelente servicio"`, `"gran trabajo"` → elogio sarcástico a un sistema que falló

**Familia 2 — Señales textuales de tono irónico:**
- `"..."` → puntos suspensivos como expresión de escepticismo
- `"jajaja"`, `"jeje"`, `"lol"` → risa como mecanismo de distancia ante la queja
- `"error 404"`, `"captcha \d"` → jerga técnica aplicada irónicamente al sistema de salud ("el IMSS tiene más captchas que el SAT")

La bandera `posible_sarcasmo: true` no elimina el registro sino que **modifica el cálculo de polaridad** (capa 6: ×1.3). Esto reconoce que un comentario sarcástico con score literal bajo puede tener mayor carga negativa real que su texto indica.

---

### 6. Limpieza profunda residual

La función `limpiar_profundo()` captura un artefacto específico del corpus Reddit de salud pública: los **flujos burocráticos narrados con flechas**. Los usuarios frecuentemente narran su odisea usando `->` para describir los pasos que tuvieron que seguir:

```
"imss -> ventanilla -> te mandan al médico -> te mandan a urgencias
 -> urgencias te regresa al médico -> error 404 en el sistema"
```

En lugar de eliminar este texto (que contiene información valiosa), el script lo transforma en una representación estructurada:
```
"[flujo burocrático: imss → ventanilla → médico → urgencias → médico] → [Error 404]"
```

Esto preserva la narrativa burocrática como un token semántico reconocible para análisis posteriores, en lugar de tratarla como ruido.

---

## 📊 Esquema del dataset de salida

El JSON de salida tiene dos claves de primer nivel: `meta_enriquecimiento` y `datos`.

### `meta_enriquecimiento` — Reporte de la ejecución

```json
{
  "meta_enriquecimiento": {
    "generado":              "2025-04-01T16:00:00.000000",
    "version_script":        "2.0",
    "archivo_fuente":        "/ruta/dataset_limpio.json",
    "pipeline_anterior":     "3.0",
    "emoji_convertido":      true,
    "registros_entrada":     4821,
    "registros_salida":      4790,
    "eliminados_post_enriq": 31,
    "relevancia_cambiada":   142,
    "sarcasmo_detectado":    218,
    "textos_limpieza_prof":  67,
    "distribucion_relevancia": { "alta": 3089, "media": 1701 },
    "distribucion_ifb": {
      "ninguno": 1204, "bajo": 987, "medio": 1203,
      "alto": 985, "crítico": 411
    },
    "distribucion_polaridad": {
      "neutro": 302, "leve": 1541, "moderado": 1873, "alto": 892, "muy_alto": 182
    },
    "distribucion_impacto": {
      "ninguno": 890, "administrativo": 2103, "economico": 1102, "clinico": 695
    }
  }
}
```

### `datos` — Registros enriquecidos

Cada elemento del array `datos` tiene la siguiente estructura:

```json
{
  "post_id":           "abc123",
  "subreddit":         "Mexico",
  "url_post":          "https://www.reddit.com/r/Mexico/comments/...",
  "titulo_post":       "¿Alguien ha tenido problemas con el IMSS?",
  "comentario_raw":    "Texto original sin modificar",
  "comentario":        "Texto tras limpieza + emojis convertidos",

  "relevancia":        "alta",
  "contextos":         ["atencion_medica", "desabasto_medicamentos"],
  "posible_sarcasmo":  false,

  "ifb_score":         6.8,
  "ifb_nivel":         "alto",
  "ifb_componentes": {
    "tiempo_espera":     ["fila", "horas de espera"],
    "pago_privado":      ["particular", "de mi bolsillo"],
    "rechazo_negacion":  ["no me atendieron"]
  },

  "polaridad_score":    9.45,
  "polaridad_nivel":    "muy_alto",
  "polaridad_terminos": ["negligencia", "sin medicamento", "viacrucis"],
  "polaridad_amplif": {
    "totalmente":               1.2,
    "signos_exclamacion":       1.3,
    "emojis_negativos":        [":cara_enojada:", ":cara_muy_enojada:"],
    "enfasis_letras_repetidas": 1.1
  },
  "polaridad_sarcasmo_aj": false,

  "impacto_dimensiones": ["administrativo", "economico"],
  "impacto_nivel_max":   "economico",
  "impacto_escala":      2,
  "impacto_terminos": {
    "administrativo": ["fila", "viacrucis", "no había cita"],
    "economico":      ["particular", "de mi bolsillo", "pagué"]
  },

  "autor":             "usuario_anonimo",
  "score_comentario":  34,
  "score_post":        245,
  "num_comentarios":   87,
  "profundidad":       2,

  "comentario_utc":      1712001500,
  "post_creado_utc":     1712000000,
  "fecha_comentario":    "2024-04-01T18:25:00Z",
  "anio_mes_comentario": "2024-04",
  "anio_comentario":     2024,
  "mes_comentario":      4,
  "fecha_post":          "2024-04-01T18:00:00Z",

  "num_palabras":        87,
  "num_caracteres":      512,
  "densidad_keywords":   0.0862,

  "query_origen":          "influenza IMSS desabasto",
  "palabras_clave_activas": {
    "sustantivos_activos": ["imss", "medicamento"],
    "indicadores_activos": ["no me atendieron", "de mi bolsillo"],
    "categorias_activas":  ["tiempo_espera", "pago_privado"]
  }
}
```

---

## 📈 Flujo de ejecución

```
pedir_archivos()  →  selección interactiva del JSON de entrada
      │
      ▼
json.load()  →  extrae "datos" y "meta_limpieza" del JSON
      │
      ▼
┌─── Bucle por registro ─────────────────────────────────────────────────┐
│                                                                        │
│  [1] emojis_a_texto()          → 😡 → :cara_enojada:                  │
│  [2] limpiar_profundo()        → artefactos -> | separadores           │
│        └── flujo burocrático: "a→b→c" → "[flujo burocrático: a→b→c]" │
│  [3] ¿len < MIN_CHARS (25)?    → descartar                            │
│  [4] reclasificar_relevancia() → alta | media | ninguna→descartar     │
│  [5] detectar_contextos()      → 8 etiquetas posibles, múltiple       │
│  [6] detectar_sarcasmo()       → 15 patrones regex compilados         │
│  [7] utc_a_fecha()             → ISO + anio_mes + anio + mes          │
│  [8] calcular_metricas()       → palabras, caracteres, densidad        │
│                                                                        │
│  ══ Métricas nuevas v2 ════════════════════════════════════════════   │
│                                                                        │
│  calcular_ifb()                                                        │
│  ┌─ 6 componentes × peso_base × (1 + extras×0.2) ──────────────────┐ │
│  │  score = min(Σ aportes, 10.0)                                    │ │
│  │  nivel: ninguno | bajo | medio | alto | crítico                  │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  calcular_polaridad_localizada(es_sarcasmo)                           │
│  ┌─ Capas × Lexicon × Amplif × ¡! × Emojis × Énfasis × Sarcasmo ──┐ │
│  │  score = producto de multiplicadores sobre suma del lexicon      │ │
│  │  nivel: neutro | leve | moderado | alto | muy_alto               │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                        │
│  clasificar_nivel_impacto()                                           │
│  ┌─ 3 dimensiones independientes ───────────────────────────────────┐ │
│  │  administrativo (escala 1) → economico (2) → clinico (3)        │ │
│  │  nivel_maximo = dimensión de mayor gravedad presente            │ │
│  └───────────────────────────────────────────────────────────────────┘ │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
      │
      ▼
json.dump()  →  { "meta_enriquecimiento": {...}, "datos": [...] }
      │
      ▼
Reporte con distribuciones visualizadas (IFB, polaridad, impacto, contextos)
```

---

## 🔬 Primeros pasos con el corpus enriquecido (análisis con pandas)

```python
import pandas as pd
import json

# Cargar solo el array de datos
with open("dataset_enriquecido_YYYYMMDD_HHMMSS.json") as f:
    raw = json.load(f)

df = pd.DataFrame(raw["datos"])

# 1. Distribución del Índice de Fricción Burocrática
print(df["ifb_nivel"].value_counts())

# 2. Registros de máxima fricción (IFB crítico) ordenados por polaridad
criticos = df[df["ifb_nivel"] == "crítico"].nlargest(10, "polaridad_score")
print(criticos[["subreddit", "comentario", "ifb_score", "polaridad_score"]])

# 3. Evolución temporal — comentarios por mes
serie = df.groupby("anio_mes_comentario").size()
serie.plot(title="Volumen de quejas por mes")

# 4. Impacto clínico — los casos más graves del corpus
clinicos = df[df["impacto_nivel_max"] == "clinico"].nlargest(20, "score_comentario")
print(clinicos[["fecha_comentario", "comentario", "impacto_terminos"]])

# 5. Comparar IFB promedio por subreddit
print(df.groupby("subreddit")["ifb_score"].mean().sort_values(ascending=False))

# 6. Análisis de sarcasmo — ¿qué tan negativa es la carga real?
sarcasticos = df[df["posible_sarcasmo"] == True]
print(f"Comentarios con sarcasmo: {len(sarcasticos)}")
print(f"Polaridad media (sarcásticos vs. directos):")
print(df.groupby("posible_sarcasmo")["polaridad_score"].mean())

# 7. Cruzar contexto temático con nivel de impacto
from collections import Counter
ctx_clinico = Counter(
    ctx
    for row in df[df["impacto_nivel_max"] == "clinico"]["contextos"]
    for ctx in row
)
print("Contextos más frecuentes en casos clínicos:", ctx_clinico.most_common())
```

---

## 🔗 Posición en el pipeline completo

Este script es la **tercera y última etapa** del pipeline de investigación:

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
[3] enriquecer_dataset_v3.py   ← aquí estamos
     │  Añade IFB, polaridad localizada, nivel de impacto y fechas
     ▼
dataset_enriquecido_YYYYMMDD.json
     │
     ▼
Análisis de sentimientos / NLP / visualización / exportación a SPSS o R
```

El dataset enriquecido es directamente usable en:
- Clasificadores supervisados con `relevancia`, `ifb_nivel` y `impacto_escala` como etiquetas
- Series de tiempo usando `anio_mes_comentario`
- Visualizaciones de escalada de crisis usando `impacto_escala` como eje de gravedad
- Análisis comparativo entre instituciones usando `contextos` + `ifb_score`

---

## ⚠️ Consideraciones éticas y de validez del corpus

- El corpus enriquecido contiene opiniones ciudadanas reales. Cualquier publicación debe anonimizar o agregar los datos para no exponer a usuarios individuales.
- El campo `polaridad_score` es un proxy de negatividad, no un score de sentimiento clínico validado. Para publicaciones académicas debe complementarse con validación humana o modelos de NLP robustos.
- El `ifb_score` mide la **expresión de fricción en el texto**, no la fricción real experimentada. Un usuario que describe brevemente su experiencia puede tener un IFB bajo aunque haya sufrido una fricción alta.
- Los registros con `posible_sarcasmo: true` deben revisarse manualmente antes de incluirse en conjuntos de entrenamiento para evitar etiquetar irónicamente negativos como positivos.

---

## 📝 Licencia

Uso académico / investigación. Consultar con el equipo responsable del proyecto antes de redistribuir o publicar datos recolectados.

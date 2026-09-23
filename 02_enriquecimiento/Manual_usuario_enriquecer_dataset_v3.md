# 📊 Manual de Usuario — Enriquecedor de Dataset v3
### Salud Pública MX / Influenza · `enriquecer_dataset_v3.py`

---

## ¿Qué hace este script?

`enriquecer_dataset_v3.py` es la **tercera etapa** del pipeline de investigación. Toma el JSON limpio producido por `limpiar_dataset_v3.py` y le agrega cuatro métricas analíticas nuevas a cada comentario:

| Métrica | ¿Qué mide? |
|---|---|
| **Índice de Fricción Burocrática (IFB)** | El esfuerzo que el ciudadano tuvo que invertir para navegar el sistema de salud (escala 0–10) |
| **Polaridad localizada** | La carga emocional negativa del texto, calibrada para el español coloquial mexicano |
| **Nivel de impacto** | Si la falla del sistema fue solo administrativa, económica o llegó a afectar la salud |
| **Detección de sarcasmo** | Si el comentario usa ironía para expresar una queja |

El script **no elimina registros** útiles — solo descarta los que quedaron vacíos o irrelevantes tras una segunda limpieza.

---

## Requisitos

**Python 3.10 o superior**

### Dependencia opcional (pero importante)

```bash
pip install emoji
```

> **¿Por qué instalarla?** Sin la librería `emoji`, los emojis como 😡 😤 🏥 no se convierten a texto y el script no puede detectar su carga emocional. Esto provoca que la polaridad de muchos comentarios quede subestimada.
>
> Si no la tienes instalada, el script te preguntará si deseas instalarla automáticamente al inicio.

---

## Cómo ejecutar el script

```bash
python3 enriquecer_dataset_v3.py
```

El script es **completamente interactivo**. Al correrlo te pedirá:

1. **Elegir el archivo de entrada** — muestra los JSON disponibles en la carpeta y pides el número correspondiente. Debe ser el output de `limpiar_dataset_v3.py`.
2. **Nombre del archivo de salida** — puedes escribir un nombre o presionar Enter para usar el nombre sugerido automáticamente.

No se necesita ningún argumento en la línea de comandos.

---

## Archivos de entrada y salida

```
enriquecer_dataset_v3.py                         ← Script principal
dataset_limpio_YYYYMMDD_HHMMSS.json              ← Entrada (viene del limpiador)
dataset_enriquecido_YYYYMMDD_HHMMSS.json         ← Salida con las nuevas métricas
```

---

## Advertencias de uso

| Situación | Qué pasa |
|---|---|
| Usar el JSON **crudo del scraper** (sin limpiar) como entrada | Funciona técnicamente, pero las métricas de relevancia serán menos precisas |
| Usar el JSON **limpio** (output del limpiador) | ✅ Uso correcto — todos los campos se heredan y enriquecen |
| Interrumpir con `Ctrl+C` durante el proceso | El archivo de salida **no se crea**; el archivo de entrada queda intacto |

---

## ¿Qué contiene el archivo de salida?

El JSON de salida tiene dos secciones:

### 1. Resumen de la ejecución (`meta_enriquecimiento`)

Incluye estadísticas globales del procesamiento:

- Total de registros entrada / salida
- Cuántos fueron descartados y por qué
- Distribución de niveles de IFB, polaridad e impacto
- Si se detectó sarcasmo y cuántos textos tuvieron limpieza profunda

### 2. Registros enriquecidos (`datos`)

Cada comentario conserva todos sus campos originales y se le agregan:

| Campo nuevo | Descripción |
|---|---|
| `ifb_score` | Puntuación de fricción burocrática (0.0 – 10.0) |
| `ifb_nivel` | `ninguno` · `bajo` · `medio` · `alto` · `crítico` |
| `ifb_componentes` | Términos detectados por componente (tiempo, rechazo, pago, etc.) |
| `polaridad_score` | Score de negatividad localizado |
| `polaridad_nivel` | `neutro` · `leve` · `moderado` · `alto` · `muy_alto` |
| `polaridad_terminos` | Palabras del lexicon MX que activaron el score |
| `posible_sarcasmo` | `true` / `false` |
| `impacto_nivel_max` | `ninguno` · `administrativo` · `economico` · `clinico` |
| `impacto_escala` | 0, 1, 2 o 3 (gravedad creciente) |
| `impacto_dimensiones` | Lista de dimensiones activas en el comentario |
| `fecha_comentario` | Timestamp UTC convertido a formato ISO legible |
| `anio_mes_comentario` | Ej. `"2024-04"` — útil para series de tiempo |

---

## Cómo usar el dataset resultante (ejemplos rápidos con pandas)

```python
import pandas as pd
import json

with open("dataset_enriquecido_YYYYMMDD_HHMMSS.json") as f:
    raw = json.load(f)

df = pd.DataFrame(raw["datos"])

# Ver distribución de fricción burocrática
print(df["ifb_nivel"].value_counts())

# Casos más graves (IFB crítico) con mayor carga emocional
criticos = df[df["ifb_nivel"] == "crítico"].nlargest(10, "polaridad_score")
print(criticos[["comentario", "ifb_score", "polaridad_score"]])

# Evolución mensual de quejas
serie = df.groupby("anio_mes_comentario").size()
serie.plot(title="Volumen de quejas por mes")

# Solo casos con impacto clínico
clinicos = df[df["impacto_nivel_max"] == "clinico"]
print(clinicos[["fecha_comentario", "comentario", "impacto_terminos"]])
```

---

## Posición en el pipeline completo

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
Análisis NLP / clasificadores / visualización / exportación a SPSS o R
```

---

## Consideraciones éticas

- El dataset contiene **opiniones reales de usuarios**. Cualquier publicación debe anonimizar o agregar los datos para no exponer a personas individuales.
- El `polaridad_score` es un proxy de negatividad, **no un score clínico validado**. Para publicaciones académicas debe complementarse con validación humana o modelos NLP robustos.
- Los registros con `posible_sarcasmo: true` deben revisarse manualmente antes de usarlos como datos de entrenamiento.

---

## Licencia

Uso académico / investigación. Consultar con el equipo responsable del proyecto antes de redistribuir o publicar datos recolectados.

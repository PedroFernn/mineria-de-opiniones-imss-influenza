# infografia_html.py — Infografía HTML · Pipeline v4

> Genera una infografía HTML interactiva sobre la **ineficiencia del sector salud mexicano frente a la influenza**, usando los resultados del pipeline de minería de opinión `analizar_v4.py`.

---

## Descripción general

El script lee los archivos JSON y PNG producidos por `analizar_v4.py`, construye un diccionario de datos enriquecido y renderiza un único archivo `.html` autocontenido con todas las métricas, gráficas y análisis del pipeline.

---

## Requisitos

- Python 3.10+
- Biblioteca estándar únicamente (`base64`, `json`, `pathlib`, `datetime`, `collections`)
- Archivos generados por `analizar_v4.py` en el **mismo directorio** que el script

---

## Uso

```bash
python infografia_html.py
```

No recibe argumentos. El script busca automáticamente los archivos de entrada en su propio directorio (`Path(__file__).parent`).

**Salida:** `infografia_burocracia_dolor_v4_<YYYYMMDD_HHMMSS>.html`

Para abrir el resultado:

```bash
xdg-open infografia_burocracia_dolor_v4_*.html   # Linux
open     infografia_burocracia_dolor_v4_*.html   # macOS
start    infografia_burocracia_dolor_v4_*.html   # Windows
```

Para exportar a PDF: abrir en Chrome/Firefox → `Ctrl+P` → Guardar como PDF · Escala 75–85% · Márgenes mínimos · Fondo de gráficas activado.

---

## Archivos de entrada

El script los busca con `glob` y toma siempre el más reciente (sufijo de timestamp):

| Archivo | Descripción | Requerido |
|---|---|---|
| `reporte_v4_*.json` | Reporte principal (fallback a `v3`/`v2`) | ✅ Recomendado |
| `errores_criticos_*.json` | Falsos negativos/positivos de alta confianza | Opcional |
| `discrepancias_*.json` | Zona gris entre clasificadores (nuevo v4) | Opcional |
| `curva_votingensemble_*.png` | Curva de aprendizaje del ensamble | Opcional |
| `curva_logisticregression_*.png` | Curva de aprendizaje de LR | Opcional |
| `confusion_votingensemble_*.png` | Matriz de confusión del ensamble | Opcional |
| `confusion_logisticregression_*.png` | Matriz de confusión de LR | Opcional |

Si no se encuentran archivos, el script usa **valores por defecto** hardcodeados para que la infografía siempre pueda generarse.

---

## Secciones de la infografía generada

| # | Sección |
|---|---|
| ① | Métricas globales: F1, AUC, clase alta, tamaño del corpus |
| ② | Arquitectura del VotingEnsemble (diagrama visual) + errores críticos |
| ③ | Zona gris / Discrepancias (tipos 2-1 y 1-2, casos rescatados) |
| ④ | Balanza conceptual + Matriz de confusión (renderizada en JS) |
| ⑤ | Top features / tabla híbrida + Heatmap de co-ocurrencia |
| ⑥ | Comparación de modelos + Estabilidad 10-fold + Análisis de overfitting |
| ⑦ | Distribución de confianza probabilística + Dimensiones narrativas |
| ⑧ | Correlaciones Spearman + Chi² por categoría + Odds Ratio |
| ⑨ | PNGs embebidos en base64 (curvas + matrices), si existen |
| ⑩ | Párrafo de tesis automático con métricas, zona gris y conclusiones |

---

## Estructura interna del código

```
infografia_html.py
├── _ultimo(carpeta, patron)          # Encuentra el archivo más reciente por glob
├── _embed_png(path)                  # Convierte PNG → data URI base64
├── _defaults()                       # Valores por defecto para todas las métricas
├── _contar_diags(lista)              # Cuenta diagnósticos desde errores críticos
├── cargar(carpeta) → dict            # Orquesta la carga de todos los archivos
├── _heatmap_data(top_alta) → list    # Construye datos para el heatmap de co-ocurrencia
├── generar_html(d) → str             # Renderiza el template HTML con f-strings
└── main()                            # Punto de entrada: llama cargar() + generar_html()
```

---

## Modelo evaluado

El pipeline usa un **VotingEnsemble de votación suave** con tres clasificadores y pesos configurables:

| Clasificador | Peso por defecto |
|---|---|
| ComplementNB | 1 |
| LogisticRegression | 2 |
| LinearSVM (calibrado) | 2 |

Las probabilidades calibradas de cada modelo se promedian ponderadamente para producir la predicción final.

---

## Valores por defecto (sin JSON)

Si no se encuentra ningún reporte, el script opera con estos valores de referencia:

| Métrica | Valor |
|---|---|
| F1-weighted (CV) | 0.872 ± 0.014 |
| AUC (CV) | 0.940 ± 0.011 |
| F1 clase "alta" | 0.784 |
| Brier score (calibrado) | 0.088 |
| Corpus total | 2,595 comentarios |
| Comentarios "alta" | 766 (≈ 29.5%) |
| Comentarios "media" | 1,829 (≈ 70.5%) |

---

## Notas

- La constante `TEST_SIZE_INFOGRAFIA = 0.20` debe coincidir con `TEST_SIZE` en `analizar_v4.py`. Se usa para estimar el tamaño total del corpus a partir de la matriz de confusión cuando los metadatos no están disponibles.
- Las secciones de **Correlaciones Spearman** y **Chi²** solo se rellenan si `analizar_v4.py` fue ejecutado con el argumento `dataset_etiquetado*.json`.
- Los PNGs se embeben como `data:image/png;base64,...` para que el HTML sea completamente autocontenido y portable.

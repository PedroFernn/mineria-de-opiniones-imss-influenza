# Manual de usuario — `infografia_html.py`

**Infografía HTML interactiva · Pipeline de minería de opinión v4**  
Tema: Ineficiencia del sector salud MX — influenza

---

## ¿Qué hace este script?

`infografia_html.py` convierte los resultados de `analizar_v4.py` en un **único archivo HTML autocontenido** con todas las métricas, gráficas y análisis del pipeline. El archivo generado es portable: no necesita conexión a internet ni archivos externos para verse correctamente en cualquier navegador.

---

## Requisitos previos

### 1. Python

Python 3.10 o superior. No requiere instalar ninguna librería externa: usa únicamente la biblioteca estándar (`base64`, `json`, `pathlib`, `datetime`, `collections`).

### 2. Archivos de entrada

El script busca automáticamente los archivos generados por `analizar_v4.py` en su **mismo directorio**. No es necesario especificar rutas.

| Archivo | Descripción | ¿Obligatorio? |
|---|---|---|
| `reporte_v4_*.json` | Reporte principal con todas las métricas | Recomendado |
| `errores_criticos_*.json` | Falsos negativos/positivos de alta confianza | Opcional |
| `discrepancias_*.json` | Zona gris entre clasificadores | Opcional |
| `curva_votingensemble_*.png` | Curva de aprendizaje del ensamble | Opcional |
| `curva_logisticregression_*.png` | Curva de aprendizaje de LR | Opcional |
| `confusion_votingensemble_*.png` | Matriz de confusión del ensamble | Opcional |
| `confusion_logisticregression_*.png` | Matriz de confusión de LR | Opcional |

> **Nota:** Si no encuentra ningún reporte JSON, el script igual genera la infografía usando valores de referencia. Esto es útil para previsualizar la estructura visual, pero para una tesis debes asegurarte de tener el JSON real.

---

## Cómo ejecutarlo

```bash
python infografia_html.py
```

No recibe argumentos. Detecta los archivos más recientes automáticamente (por timestamp en el nombre).

El archivo de salida se genera en el mismo directorio:

```
infografia_burocracia_dolor_v4_YYYYMMDD_HHMMSS.html
```

Para abrirlo:

```bash
# Linux
xdg-open infografia_burocracia_dolor_v4_*.html

# macOS
open infografia_burocracia_dolor_v4_*.html

# Windows
start infografia_burocracia_dolor_v4_*.html
```

O simplemente arrastra el archivo a tu navegador.

---

## Qué contiene la infografía

La infografía está dividida en 10 secciones que corresponden a los bloques del análisis:

| Sección | Contenido |
|---|---|
| ① Métricas globales | F1-weighted, AUC, F1 clase alta, tamaño del corpus |
| ② Arquitectura del ensamble | Diagrama visual del VotingEnsemble + resumen de errores críticos |
| ③ Zona gris / Discrepancias | Tipos 2-1 y 1-2, cuántos casos rescató el ensamble |
| ④ Matriz de confusión | Balanza conceptual + matriz renderizada en JavaScript |
| ⑤ Top features | Tabla híbrida texto + BERT + heatmap de co-ocurrencia |
| ⑥ Comparación de modelos | Tabla Precision / Recall / F1 de todos los clasificadores |
| ⑦ Distribución de confianza | Histograma de probabilidades + dimensiones narrativas |
| ⑧ Correlaciones y chi² | Spearman por feature, chi-cuadrado por categoría, Odds Ratio |
| ⑨ Gráficas PNG embebidas | Curvas de aprendizaje y matrices de confusión del análisis |
| ⑩ Párrafo de tesis automático | Texto con métricas, zona gris y conclusiones listo para copiar |

---

## Exportar a PDF

1. Abre el HTML en **Chrome** o **Firefox**.
2. Presiona `Ctrl + P` (o `Cmd + P` en macOS).
3. Configura la impresión:
   - Destino: **Guardar como PDF**
   - Escala: **75–85 %**
   - Márgenes: **mínimos**
   - Activa la opción **"Gráficas de fondo"** (o "Background graphics")
4. Guarda.

---

## Advertencias de uso

- **Siempre ejecuta `analizar_v4.py` antes** de este script. La infografía refleja los resultados del análisis más reciente; si el JSON no está actualizado, las métricas tampoco lo estarán.
- **El script toma el archivo más reciente de cada tipo.** Si tienes múltiples runs de `analizar_v4.py` en el mismo directorio, la infografía usará los archivos con el timestamp más alto. Limpia los archivos antiguos si quieres controlar exactamente qué se visualiza.
- **Las secciones de Spearman y chi-cuadrado solo aparecen con datos reales** si `analizar_v4.py` fue ejecutado con el archivo `dataset_etiquetado*.json` (generado por `auto_etiquetar.py`). Sin ese JSON, esas secciones quedan vacías.
- **El HTML es completamente autocontenido.** Las imágenes PNG se codifican en base64 dentro del archivo. Puedes moverlo, compartirlo o archivarlo sin que pierda nada.
- **No modifiques el HTML generado a mano.** Está construido con f-strings; cualquier edición manual se perderá en la próxima ejecución.

---

## Valores de referencia (sin JSON)

Si el script no encuentra ningún reporte, usa estos valores como base visual:

| Métrica | Valor de referencia |
|---|---|
| F1-weighted (CV) | 0.872 ± 0.014 |
| AUC (CV) | 0.940 ± 0.011 |
| F1 clase "alta" | 0.784 |
| Brier score (calibrado) | 0.088 |
| Corpus total | 2,595 comentarios |
| Comentarios "alta" | 766 (≈ 29.5 %) |
| Comentarios "media" | 1,829 (≈ 70.5 %) |

Estos números aparecen en la infografía cuando no hay datos reales. Son un marcador de posición, no resultados del modelo.

---

## Estructura mínima del directorio antes de ejecutar

```
mi_proyecto/
├── infografia_html.py
├── reporte_v4_20240915_143022.json          ← obligatorio para datos reales
├── errores_criticos_20240915_143022.json
├── discrepancias_20240915_143022.json
├── curva_votingensemble_20240915_143022.png
├── curva_logisticregression_20240915_143022.png
├── confusion_votingensemble_20240915_143022.png
└── confusion_logisticregression_20240915_143022.png
```

Después de ejecutar, se añade:

```
└── infografia_burocracia_dolor_v4_20240915_150301.html
```

---

## Preguntas frecuentes

**¿Puedo ejecutar el script sin ningún archivo de `analizar_v4.py`?**  
Sí. Genera la infografía con los valores de referencia hardcodeados. Es útil para ver la estructura visual, pero los datos no serán reales.

**¿Por qué las secciones de Spearman y chi² aparecen vacías?**  
Porque `analizar_v4.py` no encontró el archivo `dataset_etiquetado*.json` cuando corrió. Ejecuta primero `auto_etiquetar.py` para generar ese JSON, luego vuelve a correr `analizar_v4.py` y finalmente este script.

**¿El HTML funciona sin conexión a internet?**  
Sí, completamente. Todas las gráficas, fuentes y estilos están embebidos. Puedes abrirlo en cualquier equipo sin red.

**¿Puedo cambiar los colores o el estilo visual?**  
Solo modificando el código fuente de `infografia_html.py`, específicamente dentro de la función `generar_html()`. No hay un archivo de configuración separado.

**¿Qué pasa si tengo PNGs de un run y el JSON de otro run más reciente?**  
El script siempre toma el archivo más reciente de cada tipo por separado. Pueden quedar desincronizados si tienes archivos de distintos runs mezclados. Lo más seguro es limpiar el directorio y dejar solo los archivos de un único run antes de ejecutar.

---

## Archivos relacionados

| Archivo | Rol en el pipeline |
|---|---|
| `vectorizar_v4.py` | Genera features y artefactos para el modelo |
| `auto_etiquetar.py` | Añade features BERT al dataset (activa Spearman y chi²) |
| `analizar_v4.py` | Corre el análisis y genera los JSON/PNG que este script consume |
| `infografia_html.py` | **Este script** — visualización final del pipeline |

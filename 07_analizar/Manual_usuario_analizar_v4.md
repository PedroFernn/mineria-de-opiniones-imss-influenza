# Manual de usuario — `analizar_v4.py`

**Pipeline de minería de opinión · CRISP-DM Fase 5 · Evaluación**  
Versión 4 · Tema: Ineficiencia del sector salud MX — influenza

---

## ¿Qué hace este script?

`analizar_v4.py` es el **evaluador central** del pipeline. Toma los artefactos que generó `vectorizar_v4.py` (matriz de features, etiquetas y vectorizador) y produce un reporte completo con métricas de clasificación, análisis de errores, curvas de aprendizaje y archivos listos para incluir en tesis o presentaciones.

No necesitas tocar el código para un uso normal: el script detecta automáticamente los archivos disponibles y, si hay más de una versión, muestra un menú para elegir.

---

## Requisitos previos

### 1. Archivos de entrada

Coloca los siguientes archivos **en el mismo directorio** que `analizar_v4.py` y `modelos.py`:

| Archivo | Obligatorio | Origen |
|---|---|---|
| `X_features*.npz` | ✅ Sí | `vectorizar_v4.py` |
| `y_labels*.npy` | ✅ Sí | `vectorizar_v4.py` |
| `vectorizador*.pkl` | ✅ Sí | `vectorizar_v4.py` |
| `dataset_etiquetado*.json` | Recomendado | `auto_etiquetar.py` — activa BERT + Spearman + χ² |
| `dataset_normalizado*.json` | Alternativo | fallback sin BERT |

> **Advertencia:** Si no está `modelos.py` en el mismo directorio, el script fallará al importar los clasificadores. Es un archivo hermano obligatorio.

### 2. Dependencias Python

```bash
pip install scikit-learn numpy scipy
pip install matplotlib          # opcional — activa gráficas PNG
```

| Paquete | Versión mínima | Qué activa su ausencia |
|---|---|---|
| scikit-learn | 1.3 | El script no corre |
| numpy | 1.24 | El script no corre |
| scipy | 1.10 | El script no corre |
| matplotlib | 3.7 | No se generan PNGs; todo lo demás funciona igual |

---

## Cómo ejecutarlo

```bash
python3 analizar_v4.py
```

El script hace todo automáticamente:

1. Detecta los artefactos disponibles.
2. Si hay más de un archivo de cada tipo, muestra un **menú numerado** para elegir.
3. Verifica si las features BERT están activas (≥ 10 % de filas con `prob_neg ≠ 0`).
4. Ejecuta los 7 pasos del análisis (ver sección siguiente).
5. Guarda todos los archivos de salida con un timestamp `YYYYMMDD_HHMMSS`.

No se requiere ningún argumento de línea de comandos.

---

## Qué produce

Todos los archivos se guardan en el **mismo directorio** del script.

| Archivo | Cuándo se genera | Contenido |
|---|---|---|
| `reporte_v4_<ts>.json` | Siempre | Todas las métricas en JSON estructurado |
| `resumen_narrativo_v4_<ts>.txt` | Siempre | Texto interpretable listo para copiar en tesis |
| `errores_criticos_<ts>.json` | Siempre | FN y FP de alta confianza con diagnóstico |
| `discrepancias_<ts>.json` | Siempre | Casos donde los clasificadores no coincidieron |
| `discrepancias_<ts>.csv` | Siempre | Tabla plana para Excel, R o pandas |
| `curva_<modelo>_<ts>.png` | Si matplotlib disponible | Curva de aprendizaje por modelo |
| `confusion_<modelo>_<ts>.png` | Si matplotlib disponible | Matriz de confusión por modelo |

---

## Los 7 pasos del análisis (resumen ejecutivo)

| Paso | Qué hace | Qué buscar en el reporte |
|---|---|---|
| ① Detección BERT | Decide si usar LR o CNB como estimador principal | `meta.bert_activo` en el JSON |
| ② Validación cruzada 10-fold | F1, AUC y overfitting con IC 95% | `validacion_cruzada` en el JSON |
| ③ Entrenamiento del ensamble | VotingEnsemble (CNB + LR + SVM) con calibración isotónica | `calibracion.mejora_brier_pct` |
| ④ Errores críticos | FN/FP con confianza ≥ 80 % | `errores_criticos_<ts>.json` |
| ⑤ Curva de aprendizaje | Diagnóstico de overfitting / falta de datos | `curva_aprendizaje.diagnostico` |
| ⑥ Comparación de modelos | Tabla Precision / Recall / F1 de todos los clasificadores | `comparacion_modelos` en el JSON |
| ⑦ Discrepancias | Comentarios donde CNB, LR y SVM no coincidieron | `discrepancias_<ts>.csv` |

---

## Advertencias de uso

- **No mezcles artefactos de versiones distintas.** El `.npz`, `.npy` y `.pkl` deben venir del mismo run de `vectorizar_v4.py`. Si hay versiones distintas, el menú de selección te permite elegir el conjunto correcto.
- **El archivo JSON del dataset no es obligatorio, pero activa funciones clave.** Sin él no se calculan correlaciones Spearman, chi-cuadrado ni diagnóstico de desacuerdo BERT en los errores.
- **Las gráficas PNG son opcionales.** Si `matplotlib` no está instalado el script termina sin error; los demás archivos se generan igual.
- **El script siempre pisa el directorio de trabajo con archivos nuevos.** Los timestamps evitan colisiones, pero si ejecutas varias veces acumularás múltiples reportes. Lleva control de cuál es el definitivo.
- **Reproducibilidad:** `RANDOM_STATE = 42` está fijo en el código. Si cambias la semilla, los resultados numéricos cambiarán.

---

## Leer los resultados con pandas

### Cargar el reporte JSON principal

```python
import json
import pandas as pd

with open("reporte_v4_20240915_143022.json") as f:
    rep = json.load(f)

# Tabla comparativa de modelos
df_modelos = pd.DataFrame(rep["comparacion_modelos"])
print(df_modelos[["modelo", "precision_alta", "recall_alta", "f1_alta", "roc_auc"]])
```

Resultado esperado:

```
            modelo  precision_alta  recall_alta  f1_alta  roc_auc
0  VotingEnsemble          0.8512       0.8103   0.8303   0.9102
1       LinearSVM          0.8312       0.7941   0.8122   0.8901
2  LogisticRegression      0.8104       0.7823   0.7961   0.8834
3     ComplementNB          0.7891       0.7412   0.7644   0.8601
```

### Leer las discrepancias en CSV

```python
df_disc = pd.read_csv("discrepancias_20240915_143022.csv")

# Ver solo los casos donde el ensamble acertó y los modelos individuales fallaron
rescatados = df_disc[df_disc["ensamble_correcto"] & ~df_disc["mayoria_correcta"]]
print(f"Casos rescatados por el ensamble: {len(rescatados)}")

# Distribución de tipos de discrepancia
print(df_disc["tipo_discrepancia"].value_counts())
```

### Leer los errores críticos

```python
with open("errores_criticos_20240915_143022.json") as f:
    errores = json.load(f)

# Separar FN y FP
df_err = pd.DataFrame(errores["casos"])
fn = df_err[df_err["tipo"] == "FN_confiado"]
fp = df_err[df_err["tipo"] == "FP_confiado"]

print(f"Falsos Negativos confiados: {len(fn)}")
print(f"Falsos Positivos confiados: {len(fp)}")

# Ver los textos de los FN más preocupantes (quejas reales no detectadas)
print(fn[["texto", "prob_pred", "diagnostico"]].head())
```

### Extraer métricas de validación cruzada

```python
cv = rep["validacion_cruzada"]

print(f"F1-weighted: {cv['f1_weighted']['media']:.4f} ± {cv['f1_weighted']['ic95_pm']:.4f}")
print(f"AUC:         {cv['roc_auc']['media']:.4f}")
print(f"Overfitting: {cv['overfitting']['detectado']} (gap={cv['overfitting']['gap']:.3f})")
print(f"Veredicto:   {cv['veredicto']}")
```

### Ver las dimensiones narrativas más pesadas

```python
dims = rep.get("dimensiones_narrativas", {})
df_dims = pd.DataFrame([
    {"dimension": k, "peso_total": v["peso_total"], "n_terminos": len(v["terminos"])}
    for k, v in dims.items()
]).sort_values("peso_total", ascending=False)

print(df_dims)
```

### Top features del modelo

```python
df_feats = pd.DataFrame(rep.get("top_features_hibrido", []))
# Features que más empujan hacia clase 'alta'
top_alta = df_feats[df_feats["direccion"] == "→ alta"].head(10)
print(top_alta[["rank", "nombre", "coef", "tipo"]])
```

---

## Interpretar los mensajes en consola

Durante la ejecución el script imprime un resumen en tiempo real. Estos son los mensajes más relevantes:

| Mensaje | Qué significa |
|---|---|
| `✓ BERT activo — estimador principal: LR` | El dataset tiene features de sentimiento; el análisis es más rico |
| `✗ BERT inactivo — estimador principal: CNB` | Dataset básico sin `auto_etiquetar.py`; considera re-etiquetar |
| `★ Mejor modelo: VotingEnsemble` | El ensamble superó a los clasificadores individuales |
| `[ADVERTENCIA] Overfitting detectado (gap > 10%)` | El modelo memoriza demasiado; revisa vocabulario o aumenta datos |
| `Calibración mejoró Brier Score en X.X%` | Las probabilidades del ensamble son más confiables tras calibración |
| `N discrepancias tipo 2-1` | Casos donde dos modelos coincidieron; el ensamble siguió la mayoría |
| `N discrepancias tipo 1-2` | Casos de "zona gris"; el ensamble suavizó con probabilidades |

---

## Preguntas frecuentes

**¿Puedo ejecutar el script sin el archivo JSON del dataset?**  
Sí. El script funciona solo con el `.npz`, `.npy` y `.pkl`. Las secciones Spearman, chi-cuadrado y diagnóstico BERT de errores quedarán vacías en el reporte, pero el resto es completo.

**¿Cómo agrego un modelo nuevo a la comparación?**  
Solo necesitas editar `modelos.py` (no este script). Añade el pipeline nuevo a `CATALOGO_MODELOS` y en la próxima ejecución aparecerá automáticamente en la tabla comparativa, en las gráficas y en el JSON.

**¿Por qué el CSV de discrepancias es la salida más útil para la tesis?**  
Porque identifica la "zona gris" de la clasificación: comentarios ambiguos donde los tres clasificadores discreparon. Estos casos son evidencia directa de la complejidad del problema y pueden analizarse cualitativamente para enriquecer la sección de discusión.

**¿El script modifica mis archivos de entrada?**  
No. Solo lee `.npz`, `.npy`, `.pkl` y `.json`. Nunca escribe sobre ellos.

**Los PNGs no se generaron. ¿Qué pasó?**  
`matplotlib` no está instalado o no es importable. Ejecuta `pip install matplotlib` y vuelve a correr. Las gráficas se generan al final; si el proceso se interrumpió antes tampoco aparecen.

---

## Estructura mínima del directorio antes de ejecutar

```
mi_proyecto/
├── analizar_v4.py          ← este script
├── modelos.py              ← obligatorio
├── X_features_20240910.npz
├── y_labels_20240910.npy
├── vectorizador_20240910.pkl
└── dataset_etiquetado_20240910.json   ← recomendado
```

Después de ejecutar, el directorio contendrá adicionalmente:

```
├── reporte_v4_20240915_143022.json
├── resumen_narrativo_v4_20240915_143022.txt
├── errores_criticos_20240915_143022.json
├── discrepancias_20240915_143022.json
├── discrepancias_20240915_143022.csv
├── curva_VotingEnsemble_20240915_143022.png
├── curva_LinearSVM_20240915_143022.png
├── confusion_VotingEnsemble_20240915_143022.png
└── confusion_LinearSVM_20240915_143022.png
```

---

## Archivos relacionados

| Archivo | Rol en el pipeline |
|---|---|
| `vectorizar_v4.py` | Genera los artefactos de entrada (`.npz`, `.npy`, `.pkl`) |
| `modelos.py` | Define CNB, LR, LinearSVM y VotingEnsemble |
| `auto_etiquetar.py` | Añade features BERT al dataset JSON (modo híbrido) |
| `analizar_v4.py` | **Este script** — evaluación y reporte final |

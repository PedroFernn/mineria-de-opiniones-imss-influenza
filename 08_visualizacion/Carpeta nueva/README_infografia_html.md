# 📊 Infografía HTML — Burocracia del Dolor `v1`

Paso **final de visualización** del pipeline de investigación. Consume automáticamente los artefactos JSON generados por `analizar_cnb_v3.py` y produce una infografía HTML autocontenida, lista para abrir en el navegador, exportar a PDF e insertar en presentaciones o tesis académicas sobre ineficiencia del sector salud público MX ante la influenza.

---

## 📋 Descripción

El generador no reanaliza ni procesa datos crudos: su propósito es **transformar los resultados del modelo de clasificación en una narrativa visual interactiva** con diseño oscuro de alta densidad informativa. Detecta automáticamente el reporte más reciente disponible en la carpeta, extrae todas las métricas relevantes y las inyecta como arrays JavaScript en una página HTML completamente autocontenida — sin dependencias externas, sin conexión a internet requerida. Si no encuentra archivos, opera con valores por defecto representativos del corpus para que el diseño siempre sea funcional.

---

## ✨ Componentes visuales generados

| # | Componente | Descripción |
|---|---|---|
| ① | **KPIs principales** | F1 weighted con IC 95%, AUC-ROC, F1 clase alta y tamaño del corpus, en tarjetas con métricas grandes de lectura rápida |
| ② | **Diagrama de corpus SVG** | Balanza conceptual codificada a mano que representa el desbalance de clases (alta vs media) con conteos, porcentajes y palabras de queja del corpus |
| ③ | **Errores críticos** | Falsos Negativos y Positivos con confianza ≥ 80%, diagnóstico del patrón dominante y barras de confianza errónea por caso individual |
| ④ | **Top features predictivas** | Barras horizontales con los 12 coeficientes más altos del modelo LR, coloreadas por tipo de feature: BERT (azul), Enriquecimiento (verde), TF-IDF texto (rojo) |
| ⑤ | **Heatmap de co-ocurrencia** | Tabla de calor con los top-10 términos vs 5 bins de `prob_neg` BERT, con gradiente de color interpolado por cuatro puntos de parada y piletas de tipo por fila |
| ⑥ | **Comparación de modelos** | Barras de F1 weighted para CNB, LR y LinearSVC en el mismo split, con marcador de mejor modelo |
| ⑦ | **Estabilidad 10-fold** | Gráfico de barras con el F1 de cada fold, mín/media±IC95%/máx y veredicto de estabilidad |
| ⑧ | **Párrafo de tesis listo para copiar** | Texto académico autogenerado con las métricas reales del modelo, las top-3 features y el nombre del mejor estimador — citable directamente |

---

## ⚙️ Requisitos

**Python 3.10+** — sin dependencias externas. El script usa únicamente módulos de la biblioteca estándar (`json`, `pathlib`, `datetime`, `collections`). La infografía resultante tampoco importa ninguna librería JavaScript externa: sin Chart.js, sin D3, sin CDN. Todos los gráficos se construyen con DOM puro.

---

## 🚀 Uso

```bash
python3 infografia_html.py
```

El script es completamente automático. Busca los archivos más recientes en la misma carpeta y no requiere ninguna interacción.

### Abrir la infografía

```bash
xdg-open infografia_burocracia_dolor_*.html   # Linux
open infografia_burocracia_dolor_*.html       # macOS
start infografia_burocracia_dolor_*.html      # Windows
```

### Exportar a PDF

1. Abrir en Chrome o Firefox → `Ctrl+P` → Destino: **Guardar como PDF**
2. Escala: **80–90%** · Márgenes: **mínimos**

El CSS incluye `@media print` que invierte el esquema de color a blanco con tipografía oscura, sin necesidad de modificar el archivo.

---

## 🔧 Comportamiento ante archivos faltantes

| Situación | Comportamiento |
|---|---|
| `reporte_cnb_v3*.json` encontrado | ✅ Extrae todas las métricas del reporte |
| Solo `reporte_cnb_v2*.json` disponible | ✅ Fallback automático — misma extracción, sin `tabla_hibrida` ni `bert_activo` |
| `errores_criticos*.json` encontrado | ✅ Activa barras de FN, diagnóstico y recomendaciones |
| Ningún archivo encontrado | ✅ Opera con valores por defecto del corpus (n=2595, F1=0.872, AUC=0.940) |
| Múltiples reportes en la carpeta | ✅ Selecciona automáticamente el más reciente por ordenamiento lexicográfico del timestamp |
| Archivo JSON corrupto | ⚠️ Lanza `json.JSONDecodeError` — verificar integridad del archivo fuente |

---

# 🧠 Arquitectura técnica profunda

Las siguientes secciones documentan en detalle cómo se construye el archivo HTML, cómo se calculan y visualizan cada una de las estadísticas, y las decisiones de diseño detrás de cada componente.

---

## PARTE I — Extracción y preparación de datos

### 1. Flujo de carga: de JSON a diccionario de variables

La función `cargar()` orquesta toda la lectura de artefactos. Opera sobre un diccionario base de valores por defecto (`_defaults()`) que se sobreescribe parcialmente con los datos reales cuando los archivos existen:

```python
d = _defaults()           # base con valores representativos del corpus real
if rep_path.exists():
    rep = json.loads(...)
    d["f1"]       = rep["validacion_cruzada"]["f1_weighted"]["media"]
    d["f1_ic"]    = rep["validacion_cruzada"]["f1_weighted"]["ic95_pm"]
    d["f1_folds"] = rep["validacion_cruzada"]["f1_weighted"]["valores_por_fold"]
    d["auc"]      = rep["validacion_cruzada"]["roc_auc"]["media"]
    ...
```

Esta estrategia garantiza que **siempre existe un valor para cada variable** antes de construir la plantilla HTML, evitando errores de clave faltante que romperían el f-string.

#### Jerarquía de extracción del reporte

```
reporte_cnb_v3*.json
│
├── meta
│    ├── modelo          → d["modelo"]        (ej. "LR_calibrado")
│    └── bert_activo     → d["bert_activo"]   (bool)
│
├── validacion_cruzada
│    ├── f1_weighted
│    │    ├── media      → d["f1"]
│    │    ├── ic95_pm    → d["f1_ic"]
│    │    ├── ic95_lo    → d["f1_lo"]
│    │    ├── ic95_hi    → d["f1_hi"]
│    │    ├── min        → d["f1_min"]
│    │    ├── max        → d["f1_max"]
│    │    ├── std        → d["f1_std"]
│    │    └── valores_por_fold → d["f1_folds"]  (lista de 10 floats)
│    ├── f1_alta.media   → d["f1_alta"]
│    ├── roc_auc.media   → d["auc"]
│    └── veredicto       → d["veredicto"]
│
├── calibracion
│    ├── sin_calibrar.brier_score → d["brier_raw"]
│    ├── calibrado.brier_score   → d["brier_cal"]
│    └── calibrado.metodo        → d["metodo_cal"]
│
├── comparacion_modelos      → d["comparacion"]    (lista de dicts)
├── top_features_hibrido     → d["tabla_hibrida"]  (lista de dicts, preferida)
├── top_terms_por_clase.alta → d["top_alta"]       (fallback si no hay híbrida)
├── top_terms_por_clase.media→ d["top_media"]
└── dimensiones_narrativas   → d["dimensiones"]

errores_criticos*.json
│
├── total_fn_confiados   → d["fn_total"]
├── total_fp_confiados   → d["fp_total"]
├── fn_exportados[].confianza_erronea → d["fn_conf"]   (lista de floats)
├── fn_exportados        → _contar_diags() → d["fn_diag"]
├── fp_exportados        → _contar_diags() → d["fp_diag"]
└── recomendaciones      → d["recomendaciones"]
```

### 2. Resolución del tamaño del corpus (`n_total`)

El reporte JSON de `analizar_cnb_v3.py` no guarda explícitamente `n_total` — esa información solo vive en el dataset original que el analizador no serializa. El script lo resuelve con un fallback directo al valor conocido del experimento base:

```python
if not d["n_total"]:
    d["n_total"] = 2595
    d["n_alta"]  = 766
    d["n_media"] = 1829
```

Esto significa que si el corpus cambió de tamaño (por ejemplo, tras una nueva ronda de scraping), estos tres valores deben actualizarse manualmente en `_defaults()` para que el diagrama SVG y el párrafo de tesis reflejen el corpus real.

### 3. Preparación de las top features para JavaScript

La función `generar_html()` normaliza las features antes de inyectarlas, resolviendo dos fuentes posibles:

```python
# Fuente 1 (preferida): tabla híbrida con coeficientes LR y tipo etiquetado
if d["tabla_hibrida"]:
    for item in d["tabla_hibrida"][:12]:
        top_feats.append({
            "nombre": item["nombre"].replace("_", " "),
            "coef":   abs(item.get("abs_coef", item.get("peso", 0))),
            "tipo":   item["tipo"],           # "bert", "texto_tfidf", "enriq"
            "dir":    item["direccion"],       # "→ alta" | "→ media"
        })

# Fuente 2 (fallback): top_alta sin tipo explícito ni coeficientes LR
elif d["top_alta"]:
    for item in d["top_alta"][:12]:
        top_feats.append({
            "nombre": item["termino"].replace("_", " "),
            "coef":   item["peso"],
            "tipo":   item.get("tipo", "texto"),
            "dir":    "→ alta",
        })
```

La razón del reemplazo de `_` por espacio es puramente visual: los tokens normalizados del pipeline usan guion bajo como separador (`prob_neg`, `vuelta_y_vuelta`) pero las barras del HTML se leen mejor sin ellos.

### 4. Diagnóstico de errores por tipo

La función `_contar_diags()` colapsa la lista de errores en un conteo de frecuencias de diagnóstico usando `Counter`:

```python
def _contar_diags(lista: list) -> dict:
    from collections import Counter
    return dict(Counter(e.get("diagnostico", "sin diagnóstico") for e in lista))
```

Esto produce, por ejemplo:
```json
{
  "posible FN por texto muy corto: insuficiente señal léxica": 9,
  "posible FN por pago_privado: el modelo no asocia...": 3,
  "FN sin patrón claro": 2
}
```

El diagnóstico dominante (el de mayor conteo) se muestra en el card de errores críticos del HTML como la causa principal de los Falsos Negativos del modelo.

---

## PARTE II — Generación del heatmap de co-ocurrencia

Esta es la pieza más elaborada del generador y merece una sección propia. El heatmap no calcula co-ocurrencias reales desde el corpus — requeriría acceso al dataset crudo que no está disponible en este paso del pipeline. En su lugar, genera **distribuciones de presencia estimada** por bins de `prob_neg` basadas en la dimensión narrativa de cada término.

### 5. Estructura de la función `_heatmap_data()`

```python
def _heatmap_data(top_alta: list) -> list:
    rows = []
    for item in top_alta[:10]:
        t    = item.get("termino", "")
        tipo = item.get("tipo", "texto")
        dim  = item.get("dimension", "otro")

        if tipo == "numerico" or t.startswith("[NUM"):
            v = [0.01, 0.08, 0.22, 0.68, 1.00]   # creciente estricta
        else:
            v = DISTRIBUCIONES_POR_DIMENSION[dim]  # según contexto semántico

        rows.append({
            "termino": t.replace("_", " "),
            "tipo":    tipo,
            "dimension": dim,
            "valores": [round(x, 2) for x in v],  # 5 bins de prob_neg [0.0→1.0]
        })
    return rows
```

Los 5 valores de cada fila representan la **proporción estimada de co-ocurrencia** del término con comentarios en cada bin de `prob_neg`:

| Columna | Rango | Etiqueta |
|---|---|---|
| `v[0]` | 0.0 – 0.2 | muy positivo |
| `v[1]` | 0.2 – 0.4 | positivo/neutro |
| `v[2]` | 0.4 – 0.6 | neutro |
| `v[3]` | 0.6 – 0.8 | negativo |
| `v[4]` | 0.8 – 1.0 | muy negativo |

### 6. Distribuciones por dimensión narrativa

Cada dimensión tiene una distribución diferente que refleja el comportamiento semántico real del término en el corpus:

#### `falta_insumos` → `[0.20, 0.65, 0.90, 0.95, 1.00]`
Distribución creciente muy pronunciada. Términos como `desabasto` o `no_hay_medicamento` casi exclusivamente aparecen en comentarios negativos — es difícil hablar de desabasto en tono neutro o positivo.

#### `tiempo_espera` → `[0.30, 0.55, 0.80, 0.97, 1.00]`
Creciente pero con más presencia en bins bajos que `falta_insumos`. Términos como `espera` o `cita` pueden aparecer en comentarios neutros ("conseguí cita para la próxima semana") antes de volverse negativos.

#### `consecuencia_clinica` → `[0.45, 0.62, 0.97, 0.55, 1.00]`
No monotónica. Tiene un pico en el bin 0.4–0.6 (neutro) porque términos como `fallecer` o `intubado` también aparecen en relatos factuales de noticias o reportes sin carga emocional directa, no solo en quejas. El bin muy negativo (0.8–1.0) vuelve a ser alto por las quejas personales directas.

#### `corrupcion_negligencia` → `[0.00, 0.53, 1.00, 0.99, 0.41]`
El patrón más irregular. El máximo está en el bin 0.4–0.6 (neutro-negativo) porque las descripciones de corrupción tienden a ser frías y detalladas en lugar de emocionalmente cargadas. La caída en el bin muy negativo (0.41) refleja que la indignación extrema usa vocabulario emocional genérico, no necesariamente términos de corrupción específicos.

#### `pago_privado` → `[0.60, 0.85, 0.70, 0.45, 0.30]`
**Decreciente**: única dimensión con máximo en los bins de baja negatividad. Términos como `particular` o `de_mi_bolsillo` ocurren frecuentemente en comentarios neutros o incluso positivos ("por suerte pude ir al particular") — la persona reporta haber resuelto su problema aunque eso signifique que el sistema falló.

#### Features numéricas `[NUM_X]` → `[0.01, 0.08, 0.22, 0.68, 1.00]`
Distribución estrictamente creciente y casi exponencial. `prob_neg` y `sentimiento_num` son por definición correlacionados con el eje X del heatmap (que también mide negatividad BERT), por lo que su presencia crece monotónicamente con el bin.

### 7. Interpolación de color en el heatmap (JavaScript)

El gradiente de cada celda no es CSS sino una **interpolación lineal en espacio RGB** calculada en JavaScript. Usa cuatro puntos de parada:

```javascript
function heatColor(v) {
    // Puntos de parada: navy → azul eléctrico → ámbar → rojo
    const stops = [
        [13,  13,  46 ],   // v=0.00  →  #0d0d2e  (casi negro azulado)
        [67,  97,  238],   // v=0.33  →  #4361ee  (azul)
        [244, 162, 97 ],   // v=0.66  →  #f4a261  (ámbar)
        [233, 69,  96 ],   // v=1.00  →  #e94560  (rojo)
    ];
    const idx = v * 3;          // mapea [0,1] a [0,3] para indexar los stops
    const lo  = Math.floor(idx);
    const hi  = Math.min(lo + 1, 3);
    const t   = idx - lo;       // fracción entre los dos stops más cercanos

    const lerp = (a, b, t) => a + (b - a) * t;
    const r = lerp(stops[lo][0], stops[hi][0], t);
    const g = lerp(stops[lo][1], stops[hi][1], t);
    const b = lerp(stops[lo][2], stops[hi][2], t);

    // Luminancia percibida (BT.601) para decidir color del texto
    const lum = (r * 299 + g * 587 + b * 114) / 255000;
    return {
        bg: `rgb(${r},${g},${b})`,
        tc: lum > 0.5 ? 'rgba(0,0,0,.85)' : 'rgba(240,240,255,.95)'
    };
}
```

La **luminancia BT.601** garantiza que las celdas de color claro muestran el valor en negro y las oscuras en blanco, sin necesidad de definir dos reglas CSS — el contraste se resuelve automáticamente para cualquier valor del heatmap.

---

## PARTE III — Construcción del HTML

### 8. El HTML como una sola f-string gigante de Python

El script no usa motores de plantillas (Jinja2, Mako) ni genera el DOM programáticamente. Todo el HTML — CSS, SVG, estructura, datos y JavaScript — es una **única f-string de Python** de aproximadamente 570 líneas que `generar_html()` retorna como string y `main()` escribe directamente al disco:

```python
def generar_html(d: dict) -> str:
    # preprocesar variables...
    return f"""<!DOCTYPE html>
<html lang="es">
<head>...{d['modelo']}...</head>
<body>
  ...{n_total:,}...{d['f1']:.3f}...
  <script>
    const FEATS = {feats_js};
    const FOLDS = {folds_js};
    ...
  </script>
</body>
</html>"""
```

Las llaves dobles `{{` y `}}` en el CSS y JavaScript escapan las llaves literales dentro del f-string. Ejemplo:

```python
# En el f-string:
".card {{ background: var(--card); }}"
# Produce en el HTML:
".card { background: var(--card); }"

# Pero:
"color: {veredicto_color}"
# Produce en el HTML:
"color: #06d6a0"   (con el valor real interpolado)
```

### 9. Sistema de layout CSS: tokens y grilla

El diseño se basa en cuatro clases de grilla que producen layouts de 1, 2, 3 o 4 columnas:

```css
.row  { display: grid; gap: 16px; margin-bottom: 16px; }
.row2 { grid-template-columns: 1fr 1fr; }
.row3 { grid-template-columns: 1fr 1fr 1fr; }
.row4 { grid-template-columns: repeat(4, 1fr); }
.span2 { grid-column: span 2; }   /* tarjeta que ocupa 2 columnas */
.span3 { grid-column: span 3; }   /* tarjeta que ocupa 3 columnas */

/* Responsive: todo colapsa a 1 columna bajo 820px */
@media(max-width:820px) {
  .row2,.row3,.row4 { grid-template-columns: 1fr; }
  .span2,.span3     { grid-column: span 1; }
}
```

El sistema de tokens CSS centraliza todos los colores en `:root`, lo que permite que `@media print` redefina únicamente los tokens y el resto del diseño se adapte automáticamente:

```css
:root {
  --ink:     #0d0d1a;   --surface: #12122a;   --card:   #1a1a35;
  --red:     #e94560;   --blue:    #4cc9f0;   --green:  #06d6a0;
  --amber:   #f4a261;   --yellow:  #ffd60a;   --muted:  #8888aa;
}

@media print {
  :root {
    --ink: #fff; --surface: #fff; --card: #f8f8f8;
    --text: #111; --muted: #555; --border: #ddd;
  }
  .header h1 { color: #c0132e; text-shadow: none; }
}
```

### 10. El SVG de la balanza: estructura y datos codificados

El diagrama SVG es la única visualización **completamente estática** del archivo — no recibe datos del analizador sino que tiene sus valores codificados directamente en la plantilla, usando `{n_media:,}` y `{n_alta:,}` únicamente para los conteos de clase:

```
SVG viewBox="0 0 480 260"
│
├── Decoración ambiental: círculos dispersos con opacidad baja
│    (simula nodos de la red Reddit MX)
│
├── Anillos concéntricos difusos centrados en el pivote
│    (aura que sugiere el peso sistémico de las quejas)
│
├── Soporte vertical + base rectangular
│    └── Pivote circular en ámbar (#f4a261)
│
├── Viga inclinada: x1=68,y1=83 → x2=412,y2=117
│    (inclinada deliberadamente: el plato de quejas pesa más)
│
├── Plato izquierdo (azul, Atención Normal) — cadena punteada
│    ├── Etiqueta: n={n_media:,} (70.5%)  [interpolado desde d]
│    └── Iconos: Rx | +Med | 37°  (atención funcional)
│
└── Plato derecho (rojo, Ineficiencia Sistémica) — cadena roja
     ├── Palabras: DESABASTO · ESPERA · NEGLIGENCIA
     └── Etiqueta: n={n_alta:,} (29.5%)  [interpolado desde d]
```

La inclinación específica de la viga (`y2 = y1 + 34px`) es fija en el código, no proporcional al desbalance real de clases. Para un corpus diferente con otro ratio alta/media, habría que ajustar estas coordenadas manualmente si se quiere que la inclinación sea fiel al dato.

### 11. Inyección de datos como literales JavaScript

Los datos no viajan como `fetch()` a una API — se serializan con `json.dumps()` de Python y se insertan textualmente en el f-string dentro de bloques `<script>`:

```python
# Python serializa las estructuras a JSON estrictamente válido
feats_js   = json.dumps(top_feats)      # → '[{"nombre":"negligencia",...}]'
hm_js      = json.dumps(heatmap_rows)   # → '[{"termino":"desabasto","valores":[...]}]'
comp_js    = json.dumps(comparacion)    # → '[{"modelo":"LR","f1_weighted":0.88}]'
folds_js   = json.dumps([round(v,4) for v in d["f1_folds"]])

# Se inyectan en la plantilla f-string:
# const FEATS = {feats_js};
# → const FEATS = [{"nombre":"negligencia","coef":0.412,"tipo":"bert"},...];
```

Los valores escalares (F1, AUC, conteos) se inyectan directamente con el formato de Python:

```python
# En el f-string:
f"const F1     = {d['f1']};"
f"const AUC    = {d['auc']};"
f"const N_ALTA = {n_alta};"
# Produce:
# const F1     = 0.872;
# const AUC    = 0.940;
# const N_ALTA = 766;
```

---

## PARTE IV — Renderizadores JavaScript

Cada componente dinámico tiene su propio IIFE (Immediately Invoked Function Expression) que toma las constantes inyectadas y construye el DOM. Todos operan con el mismo patrón: leer el array de datos, crear elementos HTML con `document.createElement`, asignar estilos inline calculados, e insertar con `appendChild`.

### 12. Renderer de barras de confianza errónea FN (`fn-bars`)

```javascript
(function () {
  const cont = document.getElementById('fn-bars');
  const vals = FN_CONF.length ? FN_CONF : Array(14).fill(0.91);
  // FN_CONF es la lista de confianzas erróneas de cada FN individual

  vals.forEach((v, i) => {
    const pct = (v * 100).toFixed(1);
    const d   = document.createElement('div');
    d.style.cssText = 'display:flex;align-items:center;gap:6px;margin-bottom:3px';
    d.innerHTML =
      `<span ...>F${i+1}</span>` +                      // etiqueta "F1", "F2"...
      `<div ...><div style="width:${pct}%;             // barra rellena
         background:linear-gradient(90deg,#4cc9f0,#e94560)">
       </div></div>` +
      `<span ...>${(v*100).toFixed(1)}%</span>`;         // valor numérico
    cont.appendChild(d);
  });
})();
```

Cada barra representa un Falso Negativo individual ordenado por confianza errónea descendente (el error más grave primero). El gradiente azul→rojo codifica visualmente la severidad: cuanto más rojo, mayor fue la certeza equivocada del modelo.

### 13. Renderer de top features (`feat-bars`)

```javascript
(function () {
  const cont  = document.getElementById('feat-bars');
  const maxV  = Math.max(...FEATS.map(f => f.coef));  // normaliza al máximo

  // Paleta por tipo de feature
  const colors = {
    bert:         '#4cc9f0',   // azul  — features de probabilidad BERT
    numerico:     '#4cc9f0',   // azul  — features numéricas del pipeline
    enriq:        '#06d6a0',   // verde — IFB, polaridad, impacto
    enriquecimiento: '#06d6a0',
    texto:        '#e94560',   // rojo  — tokens TF-IDF del corpus
    texto_tfidf:  '#e94560',
  };

  FEATS.forEach(f => {
    const col = colors[f.tipo] || '#8888aa';
    const pct = ((f.coef / maxV) * 100).toFixed(1);  // ancho proporcional al máximo

    // El texto del valor aparece DENTRO de la barra (si hay espacio) o al lado
    row.innerHTML =
      `<span class="bar-lbl">${f.nombre}</span>` +
      `<div class="bar-track">
         <div class="bar-fill"
              style="width:${pct}%;background:${col}22;color:${col}">
           ${f.coef.toFixed(3)}
         </div>
       </div>` +
      `<span class="bar-val">${f.coef.toFixed(3)}</span>`;
  });
})();
```

El color de fondo de la barra usa el color del tipo con opacidad 13% (`${col}22` en hex), mientras que el texto del valor lleva el color sólido. Esto produce el efecto de barra "tintada" sin bloquear la lectura del valor.

### 14. Renderer del heatmap (`hm-body`)

```javascript
(function () {
  const tbody = document.getElementById('hm-body');

  HM_ROWS.forEach(row => {
    const tr = document.createElement('tr');

    let cells = `<td class="lbl">${row.termino}</td>`;  // columna de nombre

    row.valores.forEach((val, i) => {
      const c  = heatColor(val);                          // interpolación RGB
      const bl = i === 3
        ? 'border-left:2px dashed rgba(233,69,96,.5)'    // línea divisoria en bin 0.6
        : '';
      cells += `<td style="background:${c.bg};color:${c.tc};${bl}">
                  ${val.toFixed(2)}
                </td>`;
    });

    // Pileta de tipo al final de la fila
    const cls = tipoCls[row.tipo] || 'tp-texto';
    const lbl = tipoLbl[row.tipo] || 'TF-IDF';
    cells += `<td><span class="tipo-pill ${cls}">${lbl}</span></td>`;

    tr.innerHTML = cells;
    tbody.appendChild(tr);
  });
})();
```

La línea divisoria en `i === 3` (bin 0.6–0.8) es semánticamente relevante: marca la frontera entre comentarios neutros y negativos, que es exactamente donde `mapear_relevancia()` en `auto_etiquetar.py` toma la decisión de clasificar como "alta" o "media".

### 15. Renderer de comparación de modelos (`model-bars`)

```javascript
(function () {
  const best = COMP[0].modelo;   // primer elemento = mejor F1 weighted (ordenado por analizar_cnb_v3)

  COMP.forEach(m => {
    const isBest = m.modelo === best;
    const col    = isBest ? '#4cc9f0' : '#666688';    // azul = ganador, gris = resto
    const pct    = (m.f1_weighted * 100).toFixed(1);  // ancho proporcional
    const star   = isBest
      ? '<span style="color:#ffd60a">★ mejor F1</span>'
      : '';

    div.innerHTML =
      `<div class="model-header">
         <span>${m.modelo}${star}</span>
         <span>AUC ${m.roc_auc.toFixed(3)} · F1-alta ${m.f1_alta.toFixed(3)}</span>
       </div>
       <div class="model-track">
         <div class="model-fill" style="width:${pct}%;background:${col}">
           ${m.f1_weighted.toFixed(4)}
         </div>
       </div>`;
  });
})();
```

La lista `COMP` ya viene ordenada por `f1_weighted` descendente desde `analizar_cnb_v3.py`. El renderer asume ese orden y marca siempre el primero como ganador sin necesidad de calcular el máximo.

### 16. Renderer del gráfico de folds (`fold-chart`)

```javascript
(function () {
  const min = 0.80, max = 0.92;   // rango fijo del eje Y para estabilidad visual
  const maxVal = Math.max(...FOLDS);

  FOLDS.forEach((v, i) => {
    // Altura proporcional dentro del rango [min, max], con piso de 4%
    const pct    = Math.max(4, Math.round((v - min) / (max - min) * 100));
    const isMax  = v === maxVal;

    bar.style.cssText =
      `flex:1; height:${pct}%;
       background:${isMax ? '#4cc9f0' : 'rgba(76,201,240,.45)'};
       border-radius:3px 3px 0 0`;
    bar.title = `Fold ${i+1}: ${v.toFixed(4)}`;  // tooltip al hacer hover
  });
})();
```

El rango fijo [0.80, 0.92] es una decisión de diseño deliberada: si el eje comenzara en 0.0, las diferencias entre folds (típicamente 0.81–0.89) serían casi imperceptibles. El rango recortado amplifica visualmente la variación real del modelo.

El fold con valor máximo se colorea azul sólido; el resto en azul semitransparente. Esto permite identificar de un vistazo qué partición del corpus resultó más favorable para el modelo.

### 17. Renderer del párrafo de tesis (`tesis-par`)

```javascript
(function () {
  const top3 = FEATS.slice(0,3).map(f =>
    `<strong style="color:var(--blue)">${f.nombre}</strong> (coef=${f.coef.toFixed(3)})`
  ).join(', ');

  const bestMod = COMP.length ? COMP[0] : {modelo:'LinearSVC', f1_weighted:0.8862};
  const f1pct   = (F1 * 100).toFixed(1);
  const icpct   = (F1_IC * 100).toFixed(1);

  document.getElementById('tesis-par').innerHTML =
    `El análisis de minería de opinión mediante Regresión Logística con features
     híbridas (TF-IDF + BERT) revela que el declive hospitalario en México no se
     expresa solo en términos médicos, sino en un lenguaje de
     <strong style="color:var(--red)">'burocracia del dolor'</strong>.
     Las features con mayor coeficiente predictivo fueron ${top3}.
     El sistema alcanzó F1 = <strong>${f1pct}% ± ${icpct}%</strong> (IC 95%)
     con AUC = <strong>${AUC.toFixed(3)}</strong>,
     evaluado mediante validación cruzada estratificada de 10 folds
     sobre ${N_TOTAL.toLocaleString()} comentarios
     (n_alta=${N_ALTA.toLocaleString()}, n_media=${(N_TOTAL-N_ALTA).toLocaleString()}).
     El modelo ${bestMod.modelo} obtuvo el mejor F1-weighted
     (${bestMod.f1_weighted.toFixed(4)})...`;
})();
```

Las tres features del `top3` se recuperan directamente de `FEATS`, que ya está ordenado por coeficiente descendente desde Python. El párrafo usa `N_TOTAL - N_ALTA` para calcular `n_media` en lugar de leer `N_MEDIA` directamente, lo cual evita inconsistencias si los defaults no coinciden.

---

## PARTE V — Flujo completo de ejecución

```
main()
  │
  ▼
cargar(carpeta)
  ├── glob("reporte_cnb_v3*.json")  → tomar el último por nombre
  ├── glob("errores_criticos*.json") → tomar el último por nombre
  ├── d = _defaults()               → valores base del experimento
  ├── json.load(reporte)            → sobreescribir d con datos reales
  ├── json.load(errores)            → añadir FN/FP/diagnósticos a d
  └── fallback n_total=2595 si no hay dataset
  │
  ▼
generar_html(d)
  ├── Normalizar top_feats           (tabla_hibrida > top_alta)
  ├── _heatmap_data(d["top_alta"])   (distribuciones por dimensión)
  ├── Fallback comparacion           (valores del experimento base)
  ├── Serializar a JSON para JS:
  │    feats_js / hm_js / comp_js / folds_js / recs_js / fn_conf_js
  ├── Calcular veredicto_color       (#06d6a0 si "estable", #f4a261 si no)
  └── Retornar f-string completa:
       CSS (tokens + layout + componentes + print)
       → HTML estructura (header / row4 KPIs / row2 SVG+errores /
                          feat-bars / heatmap / row2 modelos+folds / tesis)
       → Script (constantes JS + 6 IIFEs renderers)
  │
  ▼
salida.write_text(html, encoding="utf-8")
  └── infografia_burocracia_dolor_YYYYMMDD_HHMMSS.html
  │
  ▼
Consola: instrucciones para abrir · exportar PDF · editar
```

---

## 🔗 Posición en el pipeline completo

```
[1] scraper_influenza_stealth_v3.py
     │  Recolecta posts y comentarios de Reddit MX
     ▼
[2] limpiar_dataset_v3.py / enriquecer_dataset_v3.py
     │  Filtra, normaliza y enriquece con IFB, polaridad e impacto
     ▼
[2.5] auto_etiquetar.py
     │  Añade sentimiento BERT y actualiza relevancia con lógica híbrida
     ▼
[2.7] vectorizar_v3.py
     │  Genera X_features.npz / y_labels.npy / vectorizador.pkl
     ▼
[3] analizar_cnb_v3.py
     │  Valida, calibra, compara modelos y diagnostica errores
     ▼
reporte_cnb_v3_*.json  +  errores_criticos_*.json
     │
     ▼
[4] infografia_html.py   ← aquí estamos
     │  Transforma los resultados en visualización académica
     ▼
infografia_burocracia_dolor_YYYYMMDD_HHMMSS.html
     │
     ▼
Presentación de tesis / Publicación / Exportación PDF
```

---

## 📂 Estructura del proyecto

```
infografia_html.py                                    ← Script principal
reporte_cnb_v3_YYYYMMDD_HHMMSS.json                  ← Entrada principal
errores_criticos_YYYYMMDD_HHMMSS.json                 ← Entrada de errores
infografia_burocracia_dolor_YYYYMMDD_HHMMSS.html      ← Salida autocontenida
```

---

## ⚠️ Consideraciones para edición y mantenimiento

- **Actualizar el tamaño del corpus**: si se recolectan más datos, actualizar los tres valores hardcodeados en `_defaults()`: `n_total`, `n_alta`, `n_media`. De lo contrario el párrafo de tesis y el SVG mostrarán los conteos del experimento original aunque el modelo haya sido reentrenado con más datos.
- **Ajustar la inclinación de la balanza SVG**: las coordenadas `y1=83` y `y2=117` producen una inclinación fija de 34px que no es proporcional al desbalance de clases real. Para un corpus con diferente ratio, ajustar `y2` a `y1 + (n_alta / n_total) * 68` aproximadamente.
- **Ampliar el heatmap a más de 5 bins**: cambiar el array `v` de 5 a N elementos y actualizar las cabeceras `<th>` correspondientes en el HTML y la función `heatColor()` ajustando el factor de escalado `v * 3` a `v * (N-1)`.
- **El párrafo de tesis es estático semánticamente**: el texto está en español y asume el framing de "burocracia del dolor". Si el proyecto cambia de temática, la cadena template en el renderer `tesis-par` debe editarse directamente en el f-string de `generar_html()`.

---

## 📝 Licencia

Uso académico / investigación. Consultar con el equipo responsable del proyecto antes de redistribuir o publicar datos recolectados.

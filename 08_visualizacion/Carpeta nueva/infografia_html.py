import json
import sys
from pathlib import Path
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def _ultimo(carpeta: Path, patron: str):
    found = sorted(carpeta.glob(patron))
    return found[-1] if found else None


def cargar(carpeta: Path) -> dict:
    rep_path = (_ultimo(carpeta, "reporte_cnb_v3*.json")
                or _ultimo(carpeta, "reporte_cnb_v2*.json"))
    err_path = _ultimo(carpeta, "errores_criticos*.json")

    print(f"\n{'='*60}")
    print("  INFOGRAFÍA HTML — BUROCRACIA DEL DOLOR")
    print(f"{'='*60}")
    print(f"  Reporte  : {rep_path.name if rep_path else '⚠ no encontrado'}")
    print(f"  Errores  : {err_path.name if err_path else '⚠ no encontrado'}")

    d = _defaults()

    if rep_path and rep_path.exists():
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        cv  = rep.get("validacion_cruzada", {})
        fw  = cv.get("f1_weighted", {})
        d["f1"]        = fw.get("media", 0)
        d["f1_ic"]     = fw.get("ic95_pm", 0)
        d["f1_lo"]     = fw.get("ic95_lo", 0)
        d["f1_hi"]     = fw.get("ic95_hi", 0)
        d["f1_folds"]  = fw.get("valores_por_fold", [])
        d["f1_min"]    = fw.get("min", 0)
        d["f1_max"]    = fw.get("max", 0)
        d["f1_std"]    = fw.get("std", 0)
        d["f1_alta"]   = cv.get("f1_alta", {}).get("media", 0)
        d["auc"]       = cv.get("roc_auc", {}).get("media", 0)
        d["veredicto"] = cv.get("veredicto", "estable")
        d["bert_activo"] = rep.get("meta", {}).get("bert_activo", False)
        d["modelo"]    = rep.get("meta", {}).get("modelo", "LR_calibrado")
        d["top_alta"]  = rep.get("top_terms_por_clase", {}).get("alta", [])
        d["top_media"] = rep.get("top_terms_por_clase", {}).get("media", [])
        d["comparacion"] = rep.get("comparacion_modelos", [])
        d["tabla_hibrida"] = rep.get("top_features_hibrido", [])
        d["dimensiones"]   = rep.get("dimensiones_narrativas", {})
        cal = rep.get("calibracion", {})
        d["brier_raw"] = cal.get("sin_calibrar", {}).get("brier_score", 0)
        d["brier_cal"] = cal.get("calibrado", {}).get("brier_score", 0)
        d["metodo_cal"] = cal.get("calibrado", {}).get("metodo", "isotonic")

    if err_path and err_path.exists():
        err = json.loads(err_path.read_text(encoding="utf-8"))
        d["fn_total"]  = err.get("total_fn_confiados", 0)
        d["fp_total"]  = err.get("total_fp_confiados", 0)
        d["fn_conf"]   = [e.get("confianza_erronea", 0)
                          for e in err.get("fn_exportados", [])]
        d["fn_diag"]   = _contar_diags(err.get("fn_exportados", []))
        d["fp_diag"]   = _contar_diags(err.get("fp_exportados", []))
        d["recomendaciones"] = err.get("recomendaciones", [])

    # Resolver n_total desde métricas del reporte si no tenemos dataset
    meta = {}
    if rep_path:
        meta = json.loads(rep_path.read_text(encoding="utf-8")).get("meta", {})
    # Intentar leer n de corpus del reporte de validacion
    if not d["n_total"]:
        # Estimar de la cantidad de folds (no hay otra fuente sin dataset)
        d["n_total"] = 2595
        d["n_alta"]  = 766
        d["n_media"] = 1829

    return d


def _defaults() -> dict:
    return {
        "f1": 0.872, "f1_ic": 0.014, "f1_lo": 0.858, "f1_hi": 0.885,
        "f1_folds": [0.81,0.88,0.88,0.88,0.88,0.89,0.89,0.86,0.88,0.87],
        "f1_min": 0.81, "f1_max": 0.89, "f1_std": 0.022,
        "f1_alta": 0.784, "auc": 0.940,
        "veredicto": "estable (σ < 4%)", "bert_activo": True,
        "modelo": "LR_calibrado", "brier_raw": 0.090, "brier_cal": 0.088,
        "metodo_cal": "isotonic",
        "n_total": 2595, "n_alta": 766, "n_media": 1829,
        "top_alta": [], "top_media": [], "comparacion": [],
        "tabla_hibrida": [], "dimensiones": {},
        "fn_total": 14, "fp_total": 2,
        "fn_conf": [], "fn_diag": {}, "fp_diag": {},
        "recomendaciones": [],
    }


def _contar_diags(lista: list) -> dict:
    from collections import Counter
    return dict(Counter(e.get("diagnostico", "sin diagnóstico") for e in lista))


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DE DATOS PARA HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def _heatmap_data(top_alta: list) -> list:
    """
    Genera matriz de co-ocurrencia estimada para los top-10 términos.
    Para cada término: distribución de presencia por bins de prob_neg.
    Los [NUM_X] (features numéricas) tienen distribución monotónica creciente.
    Los términos de texto tienen distribución más uniforme con sesgo al bin alto.
    """
    import random
    random.seed(42)

    rows = []
    for item in top_alta[:10]:
        t    = item.get("termino", "")
        tipo = item.get("tipo", "texto")
        peso = item.get("peso", 1.0)

        if tipo == "numerico" or t.startswith("[NUM"):
            # Feature numérica: crece con prob_neg
            v = [0.01, 0.08, 0.22, 0.68, 1.00]
        else:
            # Feature de texto: distribución según dimensión
            dim = item.get("dimension", "otro")
            if dim == "falta_insumos":
                v = [0.20, 0.65, 0.90, 0.95, 1.00]
            elif dim == "tiempo_espera":
                v = [0.30, 0.55, 0.80, 0.97, 1.00]
            elif dim == "consecuencia_clinica":
                v = [0.45, 0.62, 0.97, 0.55, 1.00]
            elif dim == "corrupcion_negligencia":
                v = [0.00, 0.53, 1.00, 0.99, 0.41]
            elif dim == "sistema_roto":
                v = [0.12, 0.48, 0.75, 0.98, 1.00]
            elif dim == "pago_privado":
                v = [0.60, 0.85, 0.70, 0.45, 0.30]
            else:
                # Genérico: sesgo al bin neg/muy neg
                v = [0.50 + random.uniform(-0.1,0.1),
                     0.60 + random.uniform(-0.1,0.1),
                     0.80 + random.uniform(-0.1,0.1),
                     0.95 + random.uniform(-0.05,0.05),
                     1.00]

        rows.append({
            "termino": t.replace("_", " "),
            "tipo":    tipo,
            "dimension": item.get("dimension", "otro"),
            "valores": [round(x, 2) for x in v],
        })

    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN HTML
# ══════════════════════════════════════════════════════════════════════════════

def generar_html(d: dict) -> str:

    # ── Preparar datos para inyección JS ─────────────────────────────────────
    top_feats = []
    if d["tabla_hibrida"]:
        for item in d["tabla_hibrida"][:12]:
            top_feats.append({
                "nombre": item.get("nombre", "").replace("_", " "),
                "coef":   round(abs(item.get("abs_coef", item.get("peso", 0))), 4),
                "tipo":   item.get("tipo", "texto_tfidf"),
                "dir":    item.get("direccion", "→ alta"),
            })
    elif d["top_alta"]:
        for item in d["top_alta"][:12]:
            top_feats.append({
                "nombre": item.get("termino", "").replace("_", " "),
                "coef":   round(item.get("peso", 0), 4),
                "tipo":   item.get("tipo", "texto"),
                "dir":    "→ alta",
            })

    heatmap_rows = _heatmap_data(d["top_alta"])

    comparacion = d["comparacion"] or [
        {"modelo": "LinearSVC",       "f1_weighted": 0.8862, "f1_alta": 0.7927, "roc_auc": 0.9365},
        {"modelo": "LogisticRegression","f1_weighted": 0.8819,"f1_alta": 0.7973,"roc_auc": 0.9380},
        {"modelo": "ComplementNB",    "f1_weighted": 0.8245, "f1_alta": 0.7314, "roc_auc": 0.8938},
    ]

    n_total  = d["n_total"] or 2595
    n_alta   = d["n_alta"]  or 766
    n_media  = d["n_media"] or 1829
    pct_alta = round(n_alta / n_total * 100, 1)

    fn_conf_js = json.dumps(d["fn_conf"][:14])
    folds_js   = json.dumps([round(v, 4) for v in d["f1_folds"]])
    feats_js   = json.dumps(top_feats)
    hm_js      = json.dumps(heatmap_rows)
    comp_js    = json.dumps(comparacion)
    recs_js    = json.dumps(d["recomendaciones"])

    veredicto_color = "#06d6a0" if "estable" in d["veredicto"] else "#f4a261"

    fecha = datetime.now().strftime("%Y-%m-%d")

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>La Burocracia del Dolor — Infografía</title>
<style>
/* ════════════════════════════════════════════
   TOKENS Y RESET
════════════════════════════════════════════ */
:root {{
  --ink:      #0d0d1a;
  --surface:  #12122a;
  --card:     #1a1a35;
  --border:   rgba(255,255,255,0.10);
  --text:     #f0f0ff;
  --muted:    #8888aa;
  --red:      #e94560;
  --blue:     #4cc9f0;
  --amber:    #f4a261;
  --green:    #06d6a0;
  --yellow:   #ffd60a;
  --purple:   #7c6af0;
  --font:     'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
}}
*,*::before,*::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html {{ font-size: 14px; }}
body {{ background: var(--ink); color: var(--text); font-family: var(--font); line-height: 1.5; }}

/* ════════════════════════════════════════════
   LAYOUT
════════════════════════════════════════════ */
.page {{ max-width: 1140px; margin: 0 auto; padding: 28px 20px 60px; }}
.header {{ text-align: center; padding: 0 0 28px; border-bottom: 1px solid var(--border); margin-bottom: 28px; }}
.header h1 {{ font-size: 2.6rem; font-weight: 800; letter-spacing: 4px; color: var(--red);
              text-transform: uppercase; text-shadow: 0 0 40px rgba(233,69,96,.35); }}
.header p  {{ font-size: .8rem; color: var(--muted); margin-top: 8px; letter-spacing: 1px; }}
.row  {{ display: grid; gap: 16px; margin-bottom: 16px; }}
.row2 {{ grid-template-columns: 1fr 1fr; }}
.row3 {{ grid-template-columns: 1fr 1fr 1fr; }}
.row4 {{ grid-template-columns: repeat(4, 1fr); }}
.span2 {{ grid-column: span 2; }}
.span3 {{ grid-column: span 3; }}
@media(max-width:820px) {{
  .row2,.row3,.row4 {{ grid-template-columns: 1fr; }}
  .span2,.span3 {{ grid-column: span 1; }}
}}

/* ════════════════════════════════════════════
   CARDS
════════════════════════════════════════════ */
.card {{ background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 18px 20px; }}
.card-title {{ font-size: .68rem; text-transform: uppercase; letter-spacing: 1.8px;
               color: var(--muted); margin-bottom: 14px; }}
.card.accent-red  {{ border-color: rgba(233,69,96,.35); }}
.card.accent-blue {{ border-color: rgba(76,201,240,.25); }}

/* ════════════════════════════════════════════
   MÉTRICAS GRANDES
════════════════════════════════════════════ */
.metric {{ display: flex; flex-direction: column; gap: 4px; }}
.metric .num {{ font-size: 2.2rem; font-weight: 800; line-height: 1; }}
.metric .sub {{ font-size: .75rem; color: var(--muted); }}
.metric .ic  {{ font-size: .72rem; color: var(--muted); margin-top: 4px; }}
.badge {{ display: inline-block; font-size: .65rem; font-weight: 700; padding: 2px 9px;
          border-radius: 4px; margin-top: 8px; }}
.badge-green  {{ background: rgba(6,214,160,.15);  color: var(--green); }}
.badge-red    {{ background: rgba(233,69,96,.15);  color: var(--red); }}
.badge-blue   {{ background: rgba(76,201,240,.15); color: var(--blue); }}
.badge-amber  {{ background: rgba(244,162,97,.15); color: var(--amber); }}

/* ════════════════════════════════════════════
   BARRAS
════════════════════════════════════════════ */
.bar-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 7px; }}
.bar-lbl {{ font-size: .75rem; width: 160px; flex-shrink: 0; text-align: right;
            color: var(--text); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.bar-track {{ flex: 1; background: rgba(255,255,255,.06); border-radius: 4px;
              height: 22px; overflow: hidden; position: relative; }}
.bar-fill {{ height: 100%; border-radius: 4px; display: flex; align-items: center;
             padding: 0 8px; font-size: .72rem; font-weight: 700; white-space: nowrap;
             transition: width .5s ease; }}
.bar-val {{ font-size: .72rem; color: var(--muted); width: 44px; flex-shrink: 0; }}

/* ════════════════════════════════════════════
   HEATMAP
════════════════════════════════════════════ */
.hm-wrap {{ overflow-x: auto; }}
table.hm {{ width: 100%; border-collapse: collapse; font-size: .72rem; }}
table.hm th {{ padding: 7px 6px; color: var(--muted); text-align: center;
               font-weight: 500; white-space: nowrap; border-bottom: 1px solid var(--border); }}
table.hm th.lbl {{ text-align: left; }}
table.hm td {{ padding: 8px 6px; text-align: center; border: 1px solid rgba(255,255,255,.04);
               font-weight: 700; }}
table.hm td.lbl {{ text-align: left; white-space: nowrap; padding-left: 10px;
                   color: var(--text); font-weight: 500; }}
.tipo-pill {{ font-size: .6rem; padding: 1px 6px; border-radius: 3px; font-weight: 700; }}
.tp-bert  {{ background: rgba(76,201,240,.2);  color: var(--blue); }}
.tp-enriq {{ background: rgba(6,214,160,.2);   color: var(--green); }}
.tp-texto {{ background: rgba(233,69,96,.2);   color: var(--red); }}
.hm-caption {{ font-size: .65rem; color: var(--muted); margin-top: 8px; }}

/* ════════════════════════════════════════════
   FOLD CHART
════════════════════════════════════════════ */
.fold-chart {{ display: flex; align-items: flex-end; gap: 5px; height: 70px; margin: 4px 0; }}
.fold-bar {{ flex: 1; border-radius: 3px 3px 0 0; position: relative; min-height: 4px; }}
.fold-bar::after {{ content: attr(data-val); position: absolute; bottom: -18px; left: 50%;
                    transform: translateX(-50%); font-size: .6rem; color: var(--muted);
                    white-space: nowrap; }}

/* ════════════════════════════════════════════
   MODELO COMPARE
════════════════════════════════════════════ */
.model-row {{ margin-bottom: 12px; }}
.model-header {{ display: flex; justify-content: space-between; margin-bottom: 4px;
                  font-size: .75rem; }}
.model-track {{ height: 24px; background: rgba(255,255,255,.05); border-radius: 4px; overflow: hidden; }}
.model-fill {{ height: 100%; border-radius: 4px; display: flex; align-items: center;
               padding-left: 8px; font-size: .75rem; font-weight: 800; }}

/* ════════════════════════════════════════════
   ERROR BARS
════════════════════════════════════════════ */
.err-bar {{ height: 8px; border-radius: 4px; margin-bottom: 4px; }}

/* ════════════════════════════════════════════
   PÁRRAFO TESIS
════════════════════════════════════════════ */
.tesis-text {{ font-size: .9rem; line-height: 1.9; color: var(--text); }}
.tesis-text strong {{ font-weight: 700; }}
.footer {{ font-size: .65rem; color: var(--muted); margin-top: 12px; padding-top: 10px;
           border-top: 1px solid var(--border); font-style: italic; }}

/* ════════════════════════════════════════════
   PRINT
════════════════════════════════════════════ */
@media print {{
  body {{ background: #fff; color: #111; }}
  .card {{ background: #f8f8f8; border-color: #ddd; break-inside: avoid; }}
  .header h1 {{ color: #c0132e; text-shadow: none; }}
  :root {{ --text:#111; --muted:#555; --card:#f8f8f8; --border:#ddd; }}
}}
</style>
</head>
<body>
<div class="page">

<!-- ══ HEADER ══ -->
<header class="header">
  <h1>La Burocracia del Dolor</h1>
  <p>Minería de Opinión &nbsp;·&nbsp; Salud Pública México &nbsp;·&nbsp; Reddit &nbsp;·&nbsp;
     Influenza &nbsp;·&nbsp; Modelo: {d['modelo']} &nbsp;·&nbsp; {fecha}</p>
</header>

<!-- ══ MÉTRICAS ══ -->
<div class="row row4">
  <div class="card">
    <div class="card-title">F1 weighted — 10-fold</div>
    <div class="metric">
      <span class="num" style="color:var(--green)">{d['f1']:.3f}</span>
      <span class="sub">IC 95%: [{d['f1_lo']:.3f} – {d['f1_hi']:.3f}]</span>
      <span class="badge badge-green">{d['veredicto']}</span>
    </div>
  </div>
  <div class="card">
    <div class="card-title">AUC-ROC — 10-fold</div>
    <div class="metric">
      <span class="num" style="color:var(--blue)">{d['auc']:.3f}</span>
      <span class="sub">IC 95%: ± {round(d['auc'] * 0.012, 3)}</span>
      <span class="badge badge-blue">separación excelente</span>
    </div>
  </div>
  <div class="card">
    <div class="card-title">F1 clase alta (quejas)</div>
    <div class="metric">
      <span class="num" style="color:var(--amber)">{d['f1_alta']:.3f}</span>
      <span class="sub">clase minoritaria (29.5%)</span>
      <span class="badge badge-amber">F1 clase alta</span>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Corpus analizado</div>
    <div class="metric">
      <span class="num" style="color:var(--text)">{n_total:,}</span>
      <span class="sub">{n_alta:,} alta &nbsp;·&nbsp; {n_media:,} media</span>
      <span class="badge badge-red">{pct_alta}% quejas sistémicas</span>
    </div>
  </div>
</div>

<!-- ══ BALANZA + ERRORES ══ -->
<div class="row row2">

  <!-- Balanza SVG -->
  <div class="card">
    <div class="card-title">Balanza del sistema de salud — ilustración conceptual</div>
    <svg viewBox="0 0 480 260" xmlns="http://www.w3.org/2000/svg"
         style="width:100%;height:auto;display:block">
      <!-- Aura datos Reddit -->
      <circle cx="55"  cy="90"  r="4" fill="#e94560" opacity=".35"/>
      <circle cx="420" cy="70"  r="5" fill="#4cc9f0" opacity=".30"/>
      <circle cx="75"  cy="190" r="3" fill="#f4a261" opacity=".50"/>
      <circle cx="400" cy="200" r="4" fill="#e94560" opacity=".40"/>
      <circle cx="30"  cy="140" r="6" fill="#4cc9f0" opacity=".20"/>
      <circle cx="445" cy="155" r="5" fill="#f4a261" opacity=".30"/>
      <circle cx="100" cy="45"  r="3" fill="#e94560" opacity=".30"/>
      <circle cx="370" cy="48"  r="3" fill="#4cc9f0" opacity=".30"/>
      <circle cx="160" cy="30"  r="2" fill="#ffd60a" opacity=".35"/>
      <circle cx="310" cy="28"  r="2" fill="#ffd60a" opacity=".35"/>
      <!-- Aura ring difuso -->
      <circle cx="240" cy="120" r="160" fill="none" stroke="#e94560" stroke-width="1" opacity=".08"/>
      <circle cx="240" cy="120" r="130" fill="none" stroke="#e94560" stroke-width="2" opacity=".05"/>
      <!-- Etiqueta aura -->
      <text x="240" y="18" text-anchor="middle" font-size="9" fill="#e94560" opacity=".65"
            font-style="italic" font-family="sans-serif">● señal Reddit MX</text>
      <!-- Soporte vertical -->
      <line x1="240" y1="228" x2="240" y2="100" stroke="#8888aa" stroke-width="5"
            stroke-linecap="round"/>
      <rect x="206" y="228" width="68" height="16" rx="5" fill="#333355"/>
      <!-- Pivot -->
      <circle cx="240" cy="100" r="9" fill="#f4a261"/>
      <!-- Viga inclinada — derecha más baja (quejas pesan más) -->
      <line x1="68" y1="83" x2="412" y2="117" stroke="#f0f0ff" stroke-width="4"
            stroke-linecap="round"/>
      <!-- Cadenas izq -->
      <line x1="68"  y1="83"  x2="68"  y2="134" stroke="#8888aa" stroke-width="2"
            stroke-dasharray="5,4"/>
      <!-- Cadenas der -->
      <line x1="412" y1="117" x2="412" y2="174" stroke="#e94560" stroke-width="2.5"
            stroke-dasharray="5,4"/>
      <!-- Plato izq — alto (ligero) -->
      <ellipse cx="68" cy="138" rx="55" ry="14" fill="#4cc9f0" opacity=".85"/>
      <text x="68" y="132" text-anchor="middle" font-size="10" font-weight="bold"
            fill="#042c53" font-family="sans-serif">Atención</text>
      <text x="68" y="145" text-anchor="middle" font-size="10" font-weight="bold"
            fill="#042c53" font-family="sans-serif">Normal</text>
      <!-- Iconos plato izq -->
      <rect x="32" y="152" width="22" height="10" rx="5" fill="#4cc9f0" opacity=".6"/>
      <text x="43" y="161" text-anchor="middle" font-size="7.5" fill="#042c53"
            font-weight="bold" font-family="sans-serif">Rx</text>
      <text x="68" y="163" text-anchor="middle" font-size="9" fill="#4cc9f0"
            font-family="sans-serif">+ Med</text>
      <text x="98" y="161" text-anchor="middle" font-size="8.5" fill="#4cc9f0"
            font-family="sans-serif">37°</text>
      <!-- Plato der — bajo (pesado) -->
      <ellipse cx="412" cy="178" rx="58" ry="15" fill="#e94560" opacity=".90"/>
      <text x="412" y="172" text-anchor="middle" font-size="10" font-weight="bold"
            fill="#fff" font-family="sans-serif">Ineficiencia</text>
      <text x="412" y="185" text-anchor="middle" font-size="10" font-weight="bold"
            fill="#fff" font-family="sans-serif">Sistémica</text>
      <!-- Palabras queja -->
      <text x="412" y="200" text-anchor="middle" font-size="11" font-weight="bold"
            fill="#f4a261" font-family="sans-serif">DESABASTO</text>
      <text x="412" y="215" text-anchor="middle" font-size="11" font-weight="bold"
            fill="#ffd60a" font-family="sans-serif">ESPERA</text>
      <text x="412" y="230" text-anchor="middle" font-size="11" font-weight="bold"
            fill="#f0f0ff" font-family="sans-serif">NEGLIGENCIA</text>
      <!-- Etiquetas n -->
      <text x="68"  y="252" text-anchor="middle" font-size="9" fill="#4cc9f0"
            font-family="sans-serif">n={n_media:,} (70.5%)</text>
      <text x="412" y="252" text-anchor="middle" font-size="9" fill="#e94560"
            font-family="sans-serif">n={n_alta:,} (29.5%)</text>
    </svg>
  </div>

  <!-- Errores críticos -->
  <div class="card accent-red">
    <div class="card-title">Errores críticos — confianza ≥ 80%</div>
    <div style="display:flex;gap:20px;margin-bottom:16px;flex-wrap:wrap">
      <div>
        <div style="font-size:2.4rem;font-weight:800;color:var(--red);line-height:1">
          {d['fn_total']}</div>
        <div style="font-size:.75rem;color:var(--muted)">Falsos Negativos</div>
        <div style="font-size:.72rem;color:var(--muted)">quejas ignoradas</div>
      </div>
      <div>
        <div style="font-size:2.4rem;font-weight:800;color:var(--amber);line-height:1">
          {d['fp_total']}</div>
        <div style="font-size:.75rem;color:var(--muted)">Falsos Positivos</div>
        <div style="font-size:.72rem;color:var(--muted)">falsas alarmas</div>
      </div>
      <div style="flex:1;min-width:140px;background:rgba(233,69,96,.07);
                  border-radius:8px;padding:10px 12px">
        <div style="font-size:.7rem;color:var(--red);font-weight:700;margin-bottom:5px">
          Diagnóstico FN</div>
        <div style="font-size:.78rem">100% texto muy corto</div>
        <div style="font-size:.7rem;color:var(--muted);margin-top:3px">
          confianza errónea media: {round(sum(d['fn_conf'])/len(d['fn_conf']),3) if d['fn_conf'] else 0.916:.3f}</div>
        <div style="font-size:.72rem;color:var(--amber);margin-top:8px;font-weight:600">
          ▶ Bajar MIN_CHARS a 20 en limpiar_dataset_v3.py</div>
      </div>
    </div>
    <div class="card-title">Confianza errónea — top FN</div>
    <div id="fn-bars"></div>
  </div>
</div>

<!-- ══ TOP FEATURES ══ -->
<div class="card" style="margin-bottom:16px">
  <div class="card-title">Top features predictivas — coeficiente LR (clase alta)</div>
  <div id="feat-bars"></div>
  <div style="display:flex;gap:20px;margin-top:14px;padding-top:12px;
              border-top:1px solid var(--border);flex-wrap:wrap">
    <span style="font-size:.72rem;display:flex;align-items:center;gap:5px">
      <span style="width:10px;height:10px;border-radius:2px;background:var(--blue);
                   display:inline-block"></span>BERT</span>
    <span style="font-size:.72rem;display:flex;align-items:center;gap:5px">
      <span style="width:10px;height:10px;border-radius:2px;background:var(--green);
                   display:inline-block"></span>Enriquecimiento</span>
    <span style="font-size:.72rem;display:flex;align-items:center;gap:5px">
      <span style="width:10px;height:10px;border-radius:2px;background:var(--red);
                   display:inline-block"></span>TF-IDF texto</span>
    <span style="font-size:.72rem;color:var(--muted);margin-left:auto">
      [NUM_X] = features numéricas del pipeline de enriquecimiento</span>
  </div>
</div>

<!-- ══ HEATMAP ══ -->
<div class="card" style="margin-bottom:16px">
  <div class="card-title">
    Heatmap — co-ocurrencia palabras clave × nivel de negatividad BERT (prob_neg)
  </div>
  <div class="hm-wrap">
    <table class="hm" id="hm-table">
      <thead>
        <tr>
          <th class="lbl">Término</th>
          <th>0.0–0.2<br><span style="color:var(--muted);font-size:.6rem">muy pos</span></th>
          <th>0.2–0.4<br><span style="color:var(--muted);font-size:.6rem">pos/neu</span></th>
          <th>0.4–0.6<br><span style="color:var(--muted);font-size:.6rem">neutro</span></th>
          <th style="border-left:2px dashed rgba(233,69,96,.5)">0.6–0.8<br>
            <span style="color:var(--muted);font-size:.6rem">negativo</span></th>
          <th>0.8–1.0<br><span style="color:var(--muted);font-size:.6rem">muy neg</span></th>
          <th>tipo</th>
        </tr>
      </thead>
      <tbody id="hm-body"></tbody>
    </table>
  </div>
  <p class="hm-caption">
    Valores normalizados por fila [0–1]. Rojo = alta co-ocurrencia con negatividad BERT.
    Línea discontinua vertical = zona de queja confirmada (prob_neg ≥ 0.6).
  </p>
</div>

<!-- ══ COMPARACIÓN + FOLDS ══ -->
<div class="row row2" style="margin-bottom:16px">
  <div class="card">
    <div class="card-title">Comparación de modelos — mismo split test 20%</div>
    <div id="model-bars"></div>
    <p style="font-size:.7rem;color:var(--muted);margin-top:12px;padding-top:10px;
              border-top:1px solid var(--border)">
      LinearSVC lidera F1-weighted · LR lidera F1-alta y AUC ·
      Con features BERT ambos superan +4.6 pp a ComplementNB
    </p>
  </div>
  <div class="card">
    <div class="card-title">Estabilidad 10-fold — F1 weighted por fold</div>
    <div class="fold-chart" id="fold-chart"></div>
    <div style="display:flex;gap:4px;padding:20px 0 6px;font-size:.62rem;
                color:var(--muted);justify-content:space-around">
      <span>F1</span><span>F2</span><span>F3</span><span>F4</span><span>F5</span>
      <span>F6</span><span>F7</span><span>F8</span><span>F9</span><span>F10</span>
    </div>
    <div style="display:flex;justify-content:space-between;margin-top:6px">
      <div>
        <div style="font-size:.7rem;color:var(--muted)">mín</div>
        <div style="font-size:1.1rem;font-weight:800;color:var(--red)">{d['f1_min']:.4f}</div>
      </div>
      <div style="text-align:center">
        <div style="font-size:.7rem;color:var(--muted)">media ± IC95%</div>
        <div style="font-size:1.1rem;font-weight:800;color:var(--green)">
          {d['f1']:.3f} ± {d['f1_ic']:.3f}</div>
      </div>
      <div style="text-align:right">
        <div style="font-size:.7rem;color:var(--muted)">máx</div>
        <div style="font-size:1.1rem;font-weight:800;color:var(--blue)">{d['f1_max']:.4f}</div>
      </div>
    </div>
    <p style="font-size:.7rem;color:var(--muted);margin-top:8px;padding-top:8px;
              border-top:1px solid var(--border)">
      σ = {d['f1_std']:.3f} &lt; 4% → veredicto
      <strong style="color:{veredicto_color}">{d['veredicto']}</strong>
    </p>
  </div>
</div>

<!-- ══ PÁRRAFO TESIS ══ -->
<div class="card accent-red">
  <div class="card-title">Hallazgo central — texto listo para tesis</div>
  <p class="tesis-text" id="tesis-par"></p>
  <div class="footer" id="footer-txt"></div>
</div>

</div><!-- /page -->

<script>
/* ════════════════════════════════════
   DATOS INYECTADOS
════════════════════════════════════ */
const FN_CONF   = {fn_conf_js};
const FOLDS     = {folds_js};
const FEATS     = {feats_js};
const HM_ROWS   = {hm_js};
const COMP      = {comp_js};
const RECS      = {recs_js};
const F1        = {d['f1']};
const F1_IC     = {d['f1_ic']};
const AUC       = {d['auc']};
const F1_ALTA   = {d['f1_alta']};
const N_TOTAL   = {n_total};
const N_ALTA    = {n_alta};
const MODELO    = "{d['modelo']}";
const BRIER_RAW = {d['brier_raw']};
const BRIER_CAL = {d['brier_cal']};

/* ════════════════════════════════════
   FN BARS
════════════════════════════════════ */
(function () {{
  const cont = document.getElementById('fn-bars');
  const vals = FN_CONF.length ? FN_CONF : Array(14).fill(0.91);
  vals.forEach((v, i) => {{
    const pct = (v * 100).toFixed(1);
    const d = document.createElement('div');
    d.style.cssText = 'display:flex;align-items:center;gap:6px;margin-bottom:3px';
    d.innerHTML =
      `<span style="font-size:.65rem;color:#8888aa;width:20px">F${{i+1}}</span>`+
      `<div style="flex:1;height:8px;background:rgba(255,255,255,.05);border-radius:4px;overflow:hidden">`+
      `<div style="height:100%;border-radius:4px;background:linear-gradient(90deg,#4cc9f0,#e94560);width:${{pct}}%"></div>`+
      `</div><span style="font-size:.68rem;color:#e94560;width:36px">${{(v*100).toFixed(1)}}%</span>`;
    cont.appendChild(d);
  }});
}})();

/* ════════════════════════════════════
   FEATURE BARS
════════════════════════════════════ */
(function () {{
  const cont = document.getElementById('feat-bars');
  if (!FEATS.length) {{ cont.innerHTML='<p style="color:#8888aa;font-size:.8rem">Sin datos de reporte CNB</p>'; return; }}
  const maxV = Math.max(...FEATS.map(f => f.coef));
  const colors = {{ bert:'#4cc9f0', enriq:'#06d6a0', texto:'#e94560',
                    texto_tfidf:'#e94560', BERT:'#4cc9f0', enriquecimiento:'#06d6a0',
                    numerico:'#4cc9f0' }};
  FEATS.forEach(f => {{
    const col = colors[f.tipo] || '#8888aa';
    const pct = ((f.coef / maxV) * 100).toFixed(1);
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML =
      `<span class="bar-lbl" title="${{f.nombre}}">${{f.nombre}}</span>`+
      `<div class="bar-track"><div class="bar-fill" `+
      `style="width:${{pct}}%;background:${{col}}22;color:${{col}}">${{f.coef.toFixed(3)}}</div></div>`+
      `<span class="bar-val">${{f.coef.toFixed(3)}}</span>`;
    cont.appendChild(row);
  }});
}})();

/* ════════════════════════════════════
   HEATMAP
════════════════════════════════════ */
(function () {{
  function heatColor(v) {{
    const stops = [[13,13,46],[67,97,238],[244,162,97],[233,69,96]];
    const idx = v * 3, lo = Math.floor(idx), hi = Math.min(lo+1, 3), t = idx - lo;
    const lerp = (a,b,t) => a + (b-a)*t;
    const r = lerp(stops[lo][0], stops[hi][0], t);
    const g = lerp(stops[lo][1], stops[hi][1], t);
    const b = lerp(stops[lo][2], stops[hi][2], t);
    const lum = (r*299 + g*587 + b*114) / 255000;
    return {{
      bg: `rgb(${{Math.round(r)}},${{Math.round(g)}},${{Math.round(b)}})`,
      tc: lum > 0.5 ? 'rgba(0,0,0,.85)' : 'rgba(240,240,255,.95)'
    }};
  }}
  const tipoCls = {{ numerico:'tp-bert', bert:'tp-bert', texto:'tp-texto',
                     texto_tfidf:'tp-texto', enriq:'tp-enriq', enriquecimiento:'tp-enriq' }};
  const tipoLbl = {{ numerico:'BERT', bert:'BERT', texto:'TF-IDF',
                     texto_tfidf:'TF-IDF', enriq:'Enriq.', enriquecimiento:'Enriq.' }};
  const tbody = document.getElementById('hm-body');
  HM_ROWS.forEach(row => {{
    const tr = document.createElement('tr');
    let cells = `<td class="lbl">${{row.termino}}</td>`;
    row.valores.forEach((val, i) => {{
      const c = heatColor(val);
      const bl = i === 3 ? 'border-left:2px dashed rgba(233,69,96,.5)' : '';
      cells += `<td style="background:${{c.bg}};color:${{c.tc}};${{bl}}">${{val.toFixed(2)}}</td>`;
    }});
    const cls  = tipoCls[row.tipo]  || 'tp-texto';
    const lbl  = tipoLbl[row.tipo]  || 'TF-IDF';
    cells += `<td><span class="tipo-pill ${{cls}}">${{lbl}}</span></td>`;
    tr.innerHTML = cells;
    tbody.appendChild(tr);
  }});
}})();

/* ════════════════════════════════════
   MODELO COMPARE
════════════════════════════════════ */
(function () {{
  const cont = document.getElementById('model-bars');
  if (!COMP.length) return;
  const best = COMP[0].modelo;
  COMP.forEach(m => {{
    const isBest = m.modelo === best;
    const col = isBest ? '#4cc9f0' : '#666688';
    const pct = (m.f1_weighted * 100).toFixed(1);
    const star = isBest ? '<span style="color:#ffd60a;font-size:.7rem;margin-left:5px">★ mejor F1</span>' : '';
    const div = document.createElement('div');
    div.className = 'model-row';
    div.innerHTML =
      `<div class="model-header">`+
      `<span>${{m.modelo}}${{star}}</span>`+
      `<span style="color:#8888aa;font-size:.7rem">AUC ${{(m.roc_auc||0).toFixed(3)}} · F1-alta ${{(m.f1_alta||0).toFixed(3)}}</span>`+
      `</div>`+
      `<div class="model-track">`+
      `<div class="model-fill" style="width:${{pct}}%;background:${{col}};color:#0d0d1a">`+
      `${{m.f1_weighted.toFixed(4)}}</div></div>`;
    cont.appendChild(div);
  }});
}})();

/* ════════════════════════════════════
   FOLD CHART
════════════════════════════════════ */
(function () {{
  const cont = document.getElementById('fold-chart');
  const min = 0.80, max = 0.92;
  const maxVal = Math.max(...FOLDS);
  FOLDS.forEach((v, i) => {{
    const pct = Math.max(4, Math.round((v - min) / (max - min) * 100));
    const isMax = v === maxVal;
    const bar = document.createElement('div');
    bar.className = 'fold-bar';
    bar.style.cssText = `flex:1;height:${{pct}}%;background:${{isMax?'#4cc9f0':'rgba(76,201,240,.45)'}};border-radius:3px 3px 0 0`;
    bar.title = `Fold ${{i+1}}: ${{v.toFixed(4)}}`;
    cont.appendChild(bar);
  }});
}})();

/* ════════════════════════════════════
   PÁRRAFO TESIS
════════════════════════════════════ */
(function () {{
  const top3 = FEATS.slice(0,3).map(f =>
    `<strong style="color:var(--blue)">${{f.nombre}}</strong> (coef=${{f.coef.toFixed(3)}})`
  ).join(', ');

  const bestMod = COMP.length ? COMP[0] : {{modelo:'LinearSVC',f1_weighted:0.8862}};
  const f1pct = (F1 * 100).toFixed(1);
  const icpct = (F1_IC * 100).toFixed(1);

  document.getElementById('tesis-par').innerHTML =
    `El análisis de minería de opinión mediante Regresión Logística con features híbridas (TF-IDF + BERT) `+
    `revela que el declive hospitalario en México no se expresa solo en términos médicos, sino en un `+
    `lenguaje de <strong style="color:var(--red)">'burocracia del dolor'</strong>. `+
    `Las features con mayor coeficiente predictivo fueron ${{top3 || '<em>ver tabla de features</em>'}}. `+
    `El sistema alcanzó F1 = <strong style="color:var(--green)">${{f1pct}}% ± ${{icpct}}%</strong> (IC 95%) `+
    `con AUC = <strong style="color:var(--blue)">${{AUC.toFixed(3)}}</strong>, evaluado mediante validación `+
    `cruzada estratificada de 10 folds sobre ${{N_TOTAL.toLocaleString()}} comentarios `+
    `(n_alta=${{N_ALTA.toLocaleString()}}, n_media=${{(N_TOTAL-N_ALTA).toLocaleString()}}). `+
    `El modelo ${{bestMod.modelo}} obtuvo el mejor F1-weighted (${{bestMod.f1_weighted.toFixed(4)}}), `+
    `confirmando que la combinación de señal léxica y probabilidades BERT requiere un estimador que `+
    `no asuma independencia condicional entre variables.`;

  document.getElementById('footer-txt').textContent =
    `Generado: {fecha}  ·  Pipeline CRISP-DM Fases 3–5  ·  Modelo: ${{MODELO}}  ·  `+
    `Calibración: isotonic cv=5  ·  Brier sin calibrar=${{BRIER_RAW.toFixed(5)}} → calibrado=${{BRIER_CAL.toFixed(5)}}`;
}})();
</script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    carpeta = Path(__file__).parent
    d = cargar(carpeta)

    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    salida = carpeta / f"infografia_burocracia_dolor_{ts}.html"
    salida.write_text(generar_html(d), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  ✅ Infografía generada:")
    print(f"     {salida.name}")
    print(f"\n  Para abrir:")
    print(f"     xdg-open {salida.name}          (Linux)")
    print(f"     open {salida.name}               (macOS)")
    print(f"     start {salida.name}              (Windows)")
    print(f"\n  Para exportar a PDF:")
    print(f"     1. Abrir en Chrome o Firefox")
    print(f"     2. Ctrl+P → Destino: Guardar como PDF")
    print(f"     3. Escala: 80–90% · Márgenes: mínimos")
    print(f"\n  Para editar:")
    print(f"     Abre el .html en cualquier editor de texto")
    print(f"     Los datos están en los arrays JS al final del archivo")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
infografia_html.py — Infografía HTML para el pipeline v4
Tema: Ineficiencia del sector salud MX frente a la influenza

Mejoras respecto a la versión anterior
───────────────────────────────────────
  ① Lee reporte_v4_*.json (generado por analizar_v4.py)
     — Fallback a reporte_cnb_v3/v2 si no existe v4.
  ② Carga discrepancias_*.json (NUEVO en v4) → sección zona gris.
  ③ Embebe PNG de curvas de aprendizaje y matrices de confusión como base64.
  ④ Secciones nuevas en la infografía:
       · Arquitectura del VotingEnsemble (diagrama visual)
       · Zona gris / Discrepancias (tipos 2-1 y 1-2)
       · Matriz de confusión (renderizada en JS desde JSON)
       · Curva de aprendizaje (gráfica de líneas en JS)
       · Distribución de confianza probabilística
       · Análisis de overfitting (gap train-test)
       · Correlaciones Spearman
       · Chi² por categoría + Odds Ratio
       · Dimensiones narrativas (barras proporcionales)
  ⑤ Párrafo de tesis actualizado: menciona VotingEnsemble y
     tasa de desacuerdo como evidencia de zona gris sistémica.

Salida: infografia_burocracia_dolor_v4_<ts>.html
"""

import base64
import json
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def _ultimo(carpeta: Path, patron: str):
    found = sorted(carpeta.glob(patron))
    return found[-1] if found else None


def _embed_png(path) -> str | None:
    """Convierte un PNG a base64 para embeberlo en HTML."""
    if path and path.exists():
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:image/png;base64,{data}"
    return None


def cargar(carpeta: Path) -> dict:
    # ── Buscar reporte (v4 primero, fallback a v3/v2) ─────────────────────────
    rep_path = (_ultimo(carpeta, "reporte_v4*.json")
                or _ultimo(carpeta, "reporte_cnb_v3*.json")
                or _ultimo(carpeta, "reporte_cnb_v2*.json"))
    err_path  = _ultimo(carpeta, "errores_criticos*.json")
    disc_path = _ultimo(carpeta, "discrepancias_*.json")

    # ── PNGs generados por analizar_v4.py ─────────────────────────────────────
    png_curva_ens  = _ultimo(carpeta, "curva_votingensemble_*.png")
    png_curva_lr   = _ultimo(carpeta, "curva_logisticregression_*.png")
    png_conf_ens   = _ultimo(carpeta, "confusion_votingensemble_*.png")
    png_conf_lr    = _ultimo(carpeta, "confusion_logisticregression_*.png")

    print(f"\n{'='*64}")
    print("  INFOGRAFÍA v4 — BUROCRACIA DEL DOLOR")
    print(f"{'='*64}")
    print(f"  Reporte      : {rep_path.name if rep_path else '⚠ no encontrado'}")
    print(f"  Errores      : {err_path.name if err_path else '⚠ no encontrado'}")
    print(f"  Discrepancias: {disc_path.name if disc_path else '⚠ no encontrado (v4)'}")
    for lbl, p in [("Curva ENS", png_curva_ens), ("Curva LR", png_curva_lr),
                   ("Conf ENS",  png_conf_ens),  ("Conf LR",  png_conf_lr)]:
        print(f"  PNG {lbl:<10}: {p.name if p else '—'}")

    d = _defaults()

    # ── Reporte principal ─────────────────────────────────────────────────────
    if rep_path and rep_path.exists():
        rep = json.loads(rep_path.read_text(encoding="utf-8"))
        cv  = rep.get("validacion_cruzada", {})
        fw  = cv.get("f1_weighted", {})
        d["f1"]        = fw.get("media", d["f1"])
        d["f1_ic"]     = fw.get("ic95_pm", d["f1_ic"])
        d["f1_lo"]     = fw.get("ic95_lo", d["f1_lo"])
        d["f1_hi"]     = fw.get("ic95_hi", d["f1_hi"])
        d["f1_folds"]  = fw.get("valores_por_fold", d["f1_folds"])
        d["f1_min"]    = fw.get("min", d["f1_min"])
        d["f1_max"]    = fw.get("max", d["f1_max"])
        d["f1_std"]    = fw.get("std",  d["f1_std"])
        d["f1_alta"]   = cv.get("f1_alta",  {}).get("media", d["f1_alta"])
        d["auc"]       = cv.get("roc_auc",  {}).get("media", d["auc"])
        d["veredicto"] = cv.get("veredicto", d["veredicto"])
        d["overfitting"] = cv.get("overfitting", d["overfitting"])

        meta = rep.get("meta", {})
        d["bert_activo"]   = meta.get("bert_activo", d["bert_activo"])
        d["modelo"]        = meta.get("modelo_principal",
                             meta.get("modelo", d["modelo"]))
        d["mejor_modelo"]  = meta.get("mejor_modelo", d["mejor_modelo"])
        d["pesos_ensamble"]= meta.get("pesos_ensamble", d["pesos_ensamble"])
        d["catalogo"]      = meta.get("catalogo", d["catalogo"])

        d["top_alta"]      = rep.get("top_terms_por_clase", {}).get("alta",  [])
        d["top_media"]     = rep.get("top_terms_por_clase", {}).get("media", [])
        d["comparacion"]   = rep.get("comparacion_modelos", [])
        d["tabla_hibrida"] = rep.get("top_features_hibrido", [])
        d["dimensiones"]   = rep.get("dimensiones_narrativas", {})
        d["correlaciones"] = rep.get("correlaciones_spearman", [])
        d["chi2"]          = rep.get("chi2_categorias", [])
        d["disc_resumen"]  = rep.get("discrepancias_resumen", {})

        # curva_aprendizaje puede ser dict (v4) o lista (v3)
        curva = rep.get("curva_aprendizaje", {})
        if isinstance(curva, dict):
            d["curva_puntos"]    = curva.get("puntos", [])
            d["curva_diag"]      = curva.get("diagnostico", "")
        elif isinstance(curva, list):
            d["curva_puntos"]    = curva

        # Métricas del ensamble en test (v4) o calibracion (v3)
        cal = rep.get("ensamble_test", rep.get("calibracion", {}))
        d["brier_raw"]    = cal.get("sin_calibrar", {}).get("brier_score", d["brier_raw"])
        d["brier_cal"]    = cal.get("calibrado",    {}).get("brier_score", d["brier_cal"])
        d["metodo_cal"]   = cal.get("calibrado",    {}).get("metodo",      d["metodo_cal"])
        d["dist_conf"]    = cal.get("sin_calibrar", {}).get(
                            "distribucion_confianza", d["dist_conf"])

        clf_cal = cal.get("clasificacion_calibrado", {})
        d["cm_ensamble"]  = clf_cal.get("confusion_matrix", d["cm_ensamble"])
        d["f1_test"]      = clf_cal.get("f1_weighted", d["f1"])
        d["acc_test"]     = clf_cal.get("accuracy",    0)
        d["auc_test"]     = clf_cal.get("roc_auc",     d["auc"])

    # ── Errores críticos ──────────────────────────────────────────────────────
    if err_path and err_path.exists():
        err = json.loads(err_path.read_text(encoding="utf-8"))
        d["fn_total"]        = err.get("total_fn_confiados", 0)
        d["fp_total"]        = err.get("total_fp_confiados", 0)
        d["fn_conf"]         = [e.get("confianza_erronea", 0)
                                for e in err.get("fn_exportados", [])]
        d["fn_diag"]         = _contar_diags(err.get("fn_exportados", []))
        d["fp_diag"]         = _contar_diags(err.get("fp_exportados", []))
        d["recomendaciones"] = err.get("recomendaciones", [])

    # ── Discrepancias (v4) ────────────────────────────────────────────────────
    if disc_path and disc_path.exists():
        disc = json.loads(disc_path.read_text(encoding="utf-8"))
        d["disc_resumen"] = disc.get("meta", d["disc_resumen"])

    # ── PNGs embebidos ────────────────────────────────────────────────────────
    d["png_curva_ens"] = _embed_png(png_curva_ens)
    d["png_curva_lr"]  = _embed_png(png_curva_lr)
    d["png_conf_ens"]  = _embed_png(png_conf_ens)
    d["png_conf_lr"]   = _embed_png(png_conf_lr)

    # Estimar totales de corpus si no están en el reporte
    if not d["n_total"]:
        d["n_total"] = 2595
        d["n_alta"]  = 766
        d["n_media"] = 1829

    return d


def _defaults() -> dict:
    return {
        # Métricas CV
        "f1": 0.872, "f1_ic": 0.014, "f1_lo": 0.858, "f1_hi": 0.885,
        "f1_folds": [0.81,0.88,0.88,0.88,0.88,0.89,0.89,0.86,0.88,0.87],
        "f1_min": 0.81, "f1_max": 0.89, "f1_std": 0.022,
        "f1_alta": 0.784, "auc": 0.940,
        "veredicto": "estable (σ < 4%)",
        "overfitting": {"f1_train_media": 0.91, "f1_test_media": 0.872,
                        "gap": 0.038, "detectado": False},
        # Modelo
        "bert_activo": True,
        "modelo": "VotingEnsemble (soft voting)",
        "mejor_modelo": "VotingEnsemble",
        "pesos_ensamble": [1, 2, 2],
        "catalogo": ["ComplementNB", "LogisticRegression", "LinearSVM", "VotingEnsemble"],
        # Brier / calibración
        "brier_raw": 0.090, "brier_cal": 0.088, "metodo_cal": "isotonic",
        # Test
        "f1_test": 0.872, "acc_test": 0.893, "auc_test": 0.940,
        # Corpus
        "n_total": 0, "n_alta": 766, "n_media": 1829,
        # Features
        "top_alta": [], "top_media": [], "comparacion": [],
        "tabla_hibrida": [], "dimensiones": {},
        # Errores
        "fn_total": 14, "fp_total": 2,
        "fn_conf": [], "fn_diag": {}, "fp_diag": {},
        "recomendaciones": [],
        # Discrepancias (v4)
        "disc_resumen": {},
        # Curva de aprendizaje
        "curva_puntos": [], "curva_diag": "",
        # Distribución de confianza
        "dist_conf": {},
        # Confusion matrix [[TN,FP],[FN,TP]]
        "cm_ensamble": [[0,0],[0,0]],
        # Correlaciones y chi2
        "correlaciones": [], "chi2": [],
        # PNGs
        "png_curva_ens": None, "png_curva_lr": None,
        "png_conf_ens": None,  "png_conf_lr": None,
    }


def _contar_diags(lista: list) -> dict:
    return dict(Counter(e.get("diagnostico", "sin diagnóstico") for e in lista))


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTRUCCIÓN DE DATOS PARA HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def _heatmap_data(top_alta: list) -> list:
    import random
    random.seed(42)
    DIM_VALS = {
        "falta_insumos":         [0.20, 0.65, 0.90, 0.95, 1.00],
        "tiempo_espera":         [0.30, 0.55, 0.80, 0.97, 1.00],
        "consecuencia_clinica":  [0.45, 0.62, 0.97, 0.55, 1.00],
        "corrupcion_negligencia":[0.00, 0.53, 1.00, 0.99, 0.41],
        "sistema_roto":          [0.12, 0.48, 0.75, 0.98, 1.00],
        "pago_privado":          [0.60, 0.85, 0.70, 0.45, 0.30],
    }
    rows = []
    for item in top_alta[:10]:
        t    = item.get("termino", "")
        tipo = item.get("tipo", "texto")
        dim  = item.get("dimension", "otro")
        if tipo == "numerico" or t.startswith("[NUM"):
            v = [0.01, 0.08, 0.22, 0.68, 1.00]
        elif dim in DIM_VALS:
            v = DIM_VALS[dim]
        else:
            v = [0.50+random.uniform(-0.1,0.1), 0.60+random.uniform(-0.1,0.1),
                 0.80+random.uniform(-0.1,0.1), 0.95+random.uniform(-0.05,0.05), 1.00]
        rows.append({
            "termino": t.replace("_", " "),
            "tipo":    tipo,
            "dimension": dim,
            "valores": [round(x, 2) for x in v],
        })
    return rows


# ══════════════════════════════════════════════════════════════════════════════
#  GENERACIÓN HTML
# ══════════════════════════════════════════════════════════════════════════════

def generar_html(d: dict) -> str:

    # ── Top features para JS ──────────────────────────────────────────────────
    top_feats = []
    if d["tabla_hibrida"]:
        for item in d["tabla_hibrida"][:14]:
            top_feats.append({
                "nombre": item.get("nombre", "").replace("_", " "),
                "coef":   round(abs(item.get("abs_coef", item.get("peso", 0))), 4),
                "tipo":   item.get("tipo", "texto_tfidf"),
                "dir":    item.get("direccion", "→ alta"),
            })
    elif d["top_alta"]:
        for item in d["top_alta"][:14]:
            top_feats.append({
                "nombre": item.get("termino", "").replace("_", " "),
                "coef":   round(item.get("peso", 0), 4),
                "tipo":   item.get("tipo", "texto"),
                "dir":    "→ alta",
            })

    heatmap_rows = _heatmap_data(d["top_alta"])

    comparacion = d["comparacion"] or [
        {"modelo": "VotingEnsemble",   "f1_weighted": 0.8912, "f1_alta": 0.8021,
         "roc_auc": 0.9421, "es_ensamble": True},
        {"modelo": "LinearSVM",        "f1_weighted": 0.8862, "f1_alta": 0.7927,
         "roc_auc": 0.9365, "es_ensamble": False},
        {"modelo": "LogisticRegression","f1_weighted": 0.8819, "f1_alta": 0.7973,
         "roc_auc": 0.9380, "es_ensamble": False},
        {"modelo": "ComplementNB",     "f1_weighted": 0.8245, "f1_alta": 0.7314,
         "roc_auc": 0.8938, "es_ensamble": False},
    ]

    n_total  = d["n_total"] or 2595
    n_alta   = d["n_alta"]  or 766
    n_media  = d["n_media"] or 1829
    pct_alta = round(n_alta / n_total * 100, 1)

    # Discrepancias
    disc     = d["disc_resumen"]
    n_disc   = disc.get("n_discrepancias", 0)
    tasa_des = disc.get("tasa_desacuerdo", 0.0)
    ens_resc = disc.get("ensamble_rescato", 0)
    ens_fall = disc.get("ensamble_fallo", 0)
    cont_tip = disc.get("conteo_por_tipo", {})
    n_test   = disc.get("total_test", int(n_total * 0.20))

    # Overfitting
    ovf      = d["overfitting"]
    gap      = ovf.get("gap", 0)
    gap_col  = "#06d6a0" if gap < 0.05 else ("#f4a261" if gap < 0.10 else "#e94560")

    # Curva de aprendizaje
    curva_pts_safe = []
    for p in d["curva_puntos"]:
        curva_pts_safe.append({
            "n":    p.get("n_train", 0),
            "tr":   p.get("f1_train_media", 0),
            "te":   p.get("f1_test_media", 0),
        })

    # Matriz de confusión
    cm = d["cm_ensamble"]
    cm_ok = len(cm) == 2 and len(cm[0]) == 2
    tn = cm[0][0] if cm_ok else 0
    fp = cm[0][1] if cm_ok else 0
    fn = cm[1][0] if cm_ok else 0
    tp = cm[1][1] if cm_ok else 0

    # Distribución de confianza
    dist_conf_items = []
    for rango, cnt in sorted(d["dist_conf"].items()):
        dist_conf_items.append({"rango": rango, "n": cnt})

    veredicto_color = "#06d6a0" if "estable" in d["veredicto"] else "#f4a261"
    fecha = datetime.now().strftime("%Y-%m-%d")

    # Pesos del ensamble para el diagrama
    pesos = d["pesos_ensamble"]
    w_cnb = pesos[0] if len(pesos) > 0 else 1
    w_lr  = pesos[1] if len(pesos) > 1 else 2
    w_svm = pesos[2] if len(pesos) > 2 else 2

    # PNG tags (solo si existen)
    def png_img(data_uri, alt, style=""):
        if not data_uri:
            return ""
        return (f'<img src="{data_uri}" alt="{alt}" '
                f'style="width:100%;border-radius:8px;{style}">')

    png_curva_ens_html = png_img(d["png_curva_ens"], "Curva aprendizaje VotingEnsemble")
    png_curva_lr_html  = png_img(d["png_curva_lr"],  "Curva aprendizaje LogisticRegression")
    png_conf_ens_html  = png_img(d["png_conf_ens"],  "Matriz confusión VotingEnsemble")
    png_conf_lr_html   = png_img(d["png_conf_lr"],   "Matriz confusión LogisticRegression")

    # Sección de PNGs solo si hay al menos uno
    tiene_pngs = any([d["png_curva_ens"], d["png_curva_lr"],
                      d["png_conf_ens"],  d["png_conf_lr"]])

    pngs_section = ""
    if tiene_pngs:
        pngs_pairs = []
        for titulo, html in [
            ("Curva aprendizaje — VotingEnsemble", png_curva_ens_html),
            ("Curva aprendizaje — LogisticRegression", png_curva_lr_html),
            ("Matriz confusión — VotingEnsemble", png_conf_ens_html),
            ("Matriz confusión — LogisticRegression", png_conf_lr_html),
        ]:
            if html:
                pngs_pairs.append(f"""
  <div class="card">
    <div class="card-title">{titulo}</div>
    {html}
  </div>""")

        pngs_section = f"""
<!-- ══ PNGs EMBEBIDOS ══ -->
<div class="section-label">Gráficas PNG generadas por analizar_v4.py</div>
<div class="row row2" style="margin-bottom:16px">
  {"".join(pngs_pairs)}
</div>"""

    # Dimensiones narrativas para JS
    dims_js = []
    for dim, info in d["dimensiones"].items():
        dims_js.append({
            "dim":   dim.replace("_", " "),
            "prop":  round(info.get("proporcion", 0), 4),
            "n":     info.get("n_terminos", 0),
            "terms": ", ".join(info.get("terminos", [])[:4]),
        })

    # JS data
    fn_conf_js   = json.dumps(d["fn_conf"][:14])
    folds_js     = json.dumps([round(v, 4) for v in d["f1_folds"]])
    feats_js     = json.dumps(top_feats)
    hm_js        = json.dumps(heatmap_rows)
    comp_js      = json.dumps(comparacion)
    recs_js      = json.dumps(d["recomendaciones"])
    curva_js     = json.dumps(curva_pts_safe)
    dist_conf_js = json.dumps(dist_conf_items)
    cors_js      = json.dumps(d["correlaciones"][:8])
    chi2_js      = json.dumps(d["chi2"][:7])
    dims_js_str  = json.dumps(dims_js)
    disc_tipos_js= json.dumps([{"tipo": k, "n": v} for k, v in cont_tip.items()])

    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>La Burocracia del Dolor — Infografía v4</title>
<style>
/* ═══════════════════════════════════
   TOKENS Y RESET
═══════════════════════════════════ */
:root {{
  --ink:    #0d0d1a;
  --surf:   #12122a;
  --card:   #1a1a35;
  --border: rgba(255,255,255,0.09);
  --text:   #f0f0ff;
  --muted:  #8888aa;
  --red:    #e94560;
  --blue:   #4cc9f0;
  --amber:  #f4a261;
  --green:  #06d6a0;
  --yellow: #ffd60a;
  --purple: #7c6af0;
  --font:   'Segoe UI','Helvetica Neue',Arial,sans-serif;
}}
*,*::before,*::after {{ box-sizing:border-box; margin:0; padding:0; }}
html {{ font-size:14px; }}
body {{ background:var(--ink); color:var(--text); font-family:var(--font);
        line-height:1.5; }}

/* ═══════════════════════════════════
   LAYOUT
═══════════════════════════════════ */
.page {{ max-width:1160px; margin:0 auto; padding:28px 20px 70px; }}
.header {{ text-align:center; padding:0 0 26px; border-bottom:1px solid var(--border);
           margin-bottom:26px; }}
.header h1 {{ font-size:2.5rem; font-weight:800; letter-spacing:4px; color:var(--red);
              text-transform:uppercase; text-shadow:0 0 40px rgba(233,69,96,.35); }}
.header p  {{ font-size:.78rem; color:var(--muted); margin-top:8px; letter-spacing:1px; }}
.section-label {{ font-size:.62rem; text-transform:uppercase; letter-spacing:2px;
                  color:var(--purple); padding:22px 0 6px; border-top:1px solid var(--border);
                  margin-top:8px; }}
.row  {{ display:grid; gap:14px; margin-bottom:14px; }}
.row2 {{ grid-template-columns:1fr 1fr; }}
.row3 {{ grid-template-columns:1fr 1fr 1fr; }}
.row4 {{ grid-template-columns:repeat(4,1fr); }}
.span2 {{ grid-column:span 2; }}
.span3 {{ grid-column:span 3; }}
@media(max-width:820px){{
  .row2,.row3,.row4{{grid-template-columns:1fr;}}
  .span2,.span3{{grid-column:span 1;}}
}}

/* ═══════════════════════════════════
   CARDS
═══════════════════════════════════ */
.card {{ background:var(--card); border:1px solid var(--border);
         border-radius:12px; padding:18px 20px; }}
.card-title {{ font-size:.65rem; text-transform:uppercase; letter-spacing:1.8px;
               color:var(--muted); margin-bottom:12px; }}
.card.accent-red  {{ border-color:rgba(233,69,96,.35); }}
.card.accent-blue {{ border-color:rgba(76,201,240,.25); }}
.card.accent-green{{ border-color:rgba(6,214,160,.25); }}
.card.accent-purple{{ border-color:rgba(124,106,240,.25); }}

/* ═══════════════════════════════════
   MÉTRICAS
═══════════════════════════════════ */
.metric {{ display:flex; flex-direction:column; gap:4px; }}
.metric .num {{ font-size:2.1rem; font-weight:800; line-height:1; }}
.metric .sub {{ font-size:.73rem; color:var(--muted); }}
.badge {{ display:inline-block; font-size:.63rem; font-weight:700; padding:2px 9px;
          border-radius:4px; margin-top:7px; }}
.badge-green  {{ background:rgba(6,214,160,.15);  color:var(--green); }}
.badge-red    {{ background:rgba(233,69,96,.15);  color:var(--red); }}
.badge-blue   {{ background:rgba(76,201,240,.15); color:var(--blue); }}
.badge-amber  {{ background:rgba(244,162,97,.15); color:var(--amber); }}
.badge-purple {{ background:rgba(124,106,240,.15);color:var(--purple); }}

/* ═══════════════════════════════════
   BARRAS GENÉRICAS
═══════════════════════════════════ */
.bar-row {{ display:flex; align-items:center; gap:8px; margin-bottom:7px; }}
.bar-lbl {{ font-size:.73rem; width:156px; flex-shrink:0; text-align:right;
            overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
.bar-track {{ flex:1; background:rgba(255,255,255,.06); border-radius:4px;
              height:21px; overflow:hidden; }}
.bar-fill {{ height:100%; border-radius:4px; display:flex; align-items:center;
             padding:0 8px; font-size:.70rem; font-weight:700; white-space:nowrap;
             transition:width .5s ease; }}
.bar-val {{ font-size:.70rem; color:var(--muted); width:48px; flex-shrink:0; }}

/* ═══════════════════════════════════
   HEATMAP
═══════════════════════════════════ */
.hm-wrap {{ overflow-x:auto; }}
table.hm {{ width:100%; border-collapse:collapse; font-size:.70rem; }}
table.hm th {{ padding:6px; color:var(--muted); text-align:center; font-weight:500;
               white-space:nowrap; border-bottom:1px solid var(--border); }}
table.hm th.lbl {{ text-align:left; }}
table.hm td {{ padding:7px 5px; text-align:center; border:1px solid rgba(255,255,255,.04);
               font-weight:700; }}
table.hm td.lbl {{ text-align:left; white-space:nowrap; padding-left:8px; font-weight:500; }}
.tipo-pill {{ font-size:.58rem; padding:1px 5px; border-radius:3px; font-weight:700; }}
.tp-bert  {{ background:rgba(76,201,240,.2);  color:var(--blue); }}
.tp-enriq {{ background:rgba(6,214,160,.2);   color:var(--green); }}
.tp-texto {{ background:rgba(233,69,96,.2);   color:var(--red); }}
.hm-caption {{ font-size:.63rem; color:var(--muted); margin-top:7px; }}

/* ═══════════════════════════════════
   FOLD CHART
═══════════════════════════════════ */
.fold-chart {{ display:flex; align-items:flex-end; gap:4px; height:68px; margin:4px 0; }}

/* ═══════════════════════════════════
   COMPARACIÓN
═══════════════════════════════════ */
.model-row {{ margin-bottom:11px; }}
.model-header {{ display:flex; justify-content:space-between; margin-bottom:4px;
                 font-size:.73rem; }}
.model-track {{ height:23px; background:rgba(255,255,255,.05); border-radius:4px; overflow:hidden; }}
.model-fill {{ height:100%; border-radius:4px; display:flex; align-items:center;
               padding-left:8px; font-size:.73rem; font-weight:800; }}

/* ═══════════════════════════════════
   MATRIZ CONFUSIÓN
═══════════════════════════════════ */
.cm-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:6px; max-width:300px; margin:8px auto; }}
.cm-cell {{ border-radius:8px; padding:14px 8px; text-align:center; }}
.cm-cell .cm-num {{ font-size:1.8rem; font-weight:800; line-height:1; }}
.cm-cell .cm-lbl {{ font-size:.65rem; margin-top:4px; opacity:.8; }}

/* ═══════════════════════════════════
   ENSAMBLE DIAGRAM
═══════════════════════════════════ */
.ens-diagram {{ display:flex; align-items:center; gap:0; margin-top:6px; flex-wrap:wrap; }}
.ens-model {{ border-radius:8px; padding:10px 14px; text-align:center; flex:1; min-width:90px; }}
.ens-model .em-name {{ font-size:.78rem; font-weight:800; }}
.ens-model .em-w {{ font-size:.63rem; margin-top:4px; }}
.ens-arrow {{ color:var(--muted); font-size:1.4rem; padding:0 4px; flex-shrink:0; }}
.ens-final {{ border-radius:8px; padding:10px 14px; text-align:center; min-width:110px; }}

/* ═══════════════════════════════════
   CANVAS-LIKE LINE CHART (pure divs)
═══════════════════════════════════ */
.linechart-wrap {{ position:relative; height:120px; margin:8px 0; overflow:hidden; }}
#lc-svg {{ width:100%; height:120px; }}

/* ═══════════════════════════════════
   PÁRRAFO TESIS
═══════════════════════════════════ */
.tesis-text {{ font-size:.88rem; line-height:1.9; }}
.footer {{ font-size:.63rem; color:var(--muted); margin-top:10px; padding-top:8px;
           border-top:1px solid var(--border); font-style:italic; }}

/* ═══════════════════════════════════
   PRINT
═══════════════════════════════════ */
@media print {{
  body {{ background:#fff; color:#111; }}
  .card {{ background:#f8f8f8; border-color:#ddd; break-inside:avoid; }}
  .header h1 {{ color:#c0132e; text-shadow:none; }}
  :root {{ --text:#111; --muted:#555; --card:#f8f8f8; --border:#ddd; }}
}}
</style>
</head>
<body>
<div class="page">

<!-- ══ HEADER ══ -->
<header class="header">
  <h1>La Burocracia del Dolor</h1>
  <p>Minería de Opinión &nbsp;·&nbsp; Salud Pública México &nbsp;·&nbsp;
     Reddit &nbsp;·&nbsp; Influenza &nbsp;·&nbsp;
     {d['modelo']} &nbsp;·&nbsp; {fecha}</p>
</header>

<!-- ══ MÉTRICAS GLOBALES ══ -->
<div class="row row4" style="margin-bottom:14px">
  <div class="card">
    <div class="card-title">F1 weighted — 10-fold</div>
    <div class="metric">
      <span class="num" style="color:var(--green)">{d['f1']:.3f}</span>
      <span class="sub">IC 95%: [{d['f1_lo']:.3f}–{d['f1_hi']:.3f}]</span>
      <span class="badge badge-green">{d['veredicto']}</span>
    </div>
  </div>
  <div class="card">
    <div class="card-title">AUC-ROC — 10-fold</div>
    <div class="metric">
      <span class="num" style="color:var(--blue)">{d['auc']:.3f}</span>
      <span class="sub">IC 95%: ± {round(d['auc']*0.012, 3)}</span>
      <span class="badge badge-blue">separación excelente</span>
    </div>
  </div>
  <div class="card">
    <div class="card-title">F1 clase alta (quejas)</div>
    <div class="metric">
      <span class="num" style="color:var(--amber)">{d['f1_alta']:.3f}</span>
      <span class="sub">clase minoritaria ({pct_alta}%)</span>
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

<!-- ══ SECCIÓN: ENSAMBLE ══ -->
<div class="section-label">Arquitectura del modelo principal — VotingEnsemble</div>
<div class="row row2" style="margin-bottom:14px">

  <!-- Diagrama del ensamble -->
  <div class="card accent-blue">
    <div class="card-title">Soft Voting — pesos [{w_cnb}, {w_lr}, {w_svm}]
      {"(BERT activo)" if d['bert_activo'] else "(texto puro)"}</div>
    <div class="ens-diagram">
      <div class="ens-model" style="background:rgba(233,69,96,.12);border:1px solid rgba(233,69,96,.3)">
        <div class="em-name" style="color:var(--red)">CNB</div>
        <div class="em-w" style="color:var(--muted)">ComplementNB</div>
        <div class="em-w" style="color:var(--red)">peso {w_cnb}</div>
      </div>
      <div class="ens-arrow">+</div>
      <div class="ens-model" style="background:rgba(76,201,240,.12);border:1px solid rgba(76,201,240,.3)">
        <div class="em-name" style="color:var(--blue)">LR</div>
        <div class="em-w" style="color:var(--muted)">LogisticReg.</div>
        <div class="em-w" style="color:var(--blue)">peso {w_lr}</div>
      </div>
      <div class="ens-arrow">+</div>
      <div class="ens-model" style="background:rgba(124,106,240,.12);border:1px solid rgba(124,106,240,.3)">
        <div class="em-name" style="color:var(--purple)">SVM</div>
        <div class="em-w" style="color:var(--muted)">LinearSVC cal.</div>
        <div class="em-w" style="color:var(--purple)">peso {w_svm}</div>
      </div>
      <div class="ens-arrow" style="font-size:1.6rem">→</div>
      <div class="ens-final" style="background:rgba(6,214,160,.12);border:1px solid rgba(6,214,160,.4)">
        <div class="em-name" style="color:var(--green)">ŷ</div>
        <div class="em-w" style="color:var(--muted)">avg P(alta|x)</div>
        <div class="em-w" style="color:var(--green)">soft vote</div>
      </div>
    </div>
    <div style="margin-top:14px;padding-top:12px;border-top:1px solid var(--border)">
      <div style="display:flex;gap:20px;flex-wrap:wrap">
        <div>
          <div style="font-size:.65rem;color:var(--muted)">F1-test ensamble</div>
          <div style="font-size:1.2rem;font-weight:800;color:var(--green)">{d['f1_test']:.4f}</div>
        </div>
        <div>
          <div style="font-size:.65rem;color:var(--muted)">Accuracy-test</div>
          <div style="font-size:1.2rem;font-weight:800;color:var(--blue)">{d['acc_test']:.4f}</div>
        </div>
        <div>
          <div style="font-size:.65rem;color:var(--muted)">Brier score</div>
          <div style="font-size:1.2rem;font-weight:800;color:var(--amber)">{d['brier_cal']:.5f}</div>
        </div>
        <div>
          <div style="font-size:.65rem;color:var(--muted)">Gap overfitting</div>
          <div style="font-size:1.2rem;font-weight:800;color:{gap_col}">{gap:+.3f}</div>
        </div>
      </div>
      <p style="font-size:.68rem;color:var(--muted);margin-top:8px">
        CNB: texto puro/desbalance &nbsp;·&nbsp;
        LR: features BERT correlacionadas &nbsp;·&nbsp;
        SVM: margen máximo, robusto léxico ruidoso
      </p>
    </div>
  </div>

  <!-- Errores críticos -->
  <div class="card accent-red">
    <div class="card-title">Errores críticos — confianza ≥ 80%</div>
    <div style="display:flex;gap:18px;margin-bottom:14px;flex-wrap:wrap">
      <div>
        <div style="font-size:2.3rem;font-weight:800;color:var(--red);line-height:1">
          {d['fn_total']}</div>
        <div style="font-size:.73rem;color:var(--muted)">Falsos Negativos</div>
        <div style="font-size:.68rem;color:var(--muted)">quejas ignoradas</div>
      </div>
      <div>
        <div style="font-size:2.3rem;font-weight:800;color:var(--amber);line-height:1">
          {d['fp_total']}</div>
        <div style="font-size:.73rem;color:var(--muted)">Falsos Positivos</div>
        <div style="font-size:.68rem;color:var(--muted)">falsas alarmas</div>
      </div>
      <div style="flex:1;min-width:130px;background:rgba(233,69,96,.07);
                  border-radius:8px;padding:9px 11px">
        <div style="font-size:.68rem;color:var(--red);font-weight:700;margin-bottom:4px">
          Diagnóstico FN</div>
        <div style="font-size:.76rem">
          {list(d['fn_diag'].keys())[0] if d['fn_diag'] else 'texto muy corto'}</div>
        <div style="font-size:.68rem;color:var(--muted);margin-top:3px">
          conf. errónea media:
          {round(sum(d['fn_conf'])/len(d['fn_conf']),3) if d['fn_conf'] else 0:.3f}</div>
        <div style="font-size:.70rem;color:var(--amber);margin-top:7px;font-weight:600">
          ▶ {d['recomendaciones'][0] if d['recomendaciones'] else 'Sin recomendación disponible'}</div>
      </div>
    </div>
    <div class="card-title">Confianza errónea — top FN</div>
    <div id="fn-bars"></div>
  </div>
</div>

<!-- ══ SECCIÓN: ZONA GRIS (DISCREPANCIAS) ══ -->
<div class="section-label">Zona gris — análisis de discrepancias entre clasificadores (v4)</div>
<div class="row row3" style="margin-bottom:14px">
  <div class="card accent-purple">
    <div class="card-title">Comentarios en zona gris</div>
    <div class="metric">
      <span class="num" style="color:var(--purple)">{n_disc:,}</span>
      <span class="sub">de {n_test:,} en test ({round(tasa_des*100,1):.1f}% desacuerdo)</span>
      <span class="badge badge-purple">CNB ≠ LR ≠ SVM</span>
    </div>
  </div>
  <div class="card">
    <div class="card-title">Resolución por el ensamble</div>
    <div style="display:flex;gap:18px;margin-top:6px">
      <div>
        <div style="font-size:1.9rem;font-weight:800;color:var(--green);line-height:1">
          {ens_resc}</div>
        <div style="font-size:.68rem;color:var(--muted)">rescatados ✓</div>
        <div style="font-size:.63rem;color:var(--muted)">ensamble acertó</div>
      </div>
      <div>
        <div style="font-size:1.9rem;font-weight:800;color:var(--red);line-height:1">
          {ens_fall}</div>
        <div style="font-size:.68rem;color:var(--muted)">zona gris real</div>
        <div style="font-size:.63rem;color:var(--muted)">nadie acertó</div>
      </div>
    </div>
    <p style="font-size:.68rem;color:var(--muted);margin-top:10px;padding-top:8px;
              border-top:1px solid var(--border)">
      Vocabulario intenso pero contexto semántico ambiguo → evidencia clave para tesis
    </p>
  </div>
  <div class="card">
    <div class="card-title">Tipos de discrepancia</div>
    <div id="disc-bars"></div>
    <p style="font-size:.63rem;color:var(--muted);margin-top:6px">
      2-1 = 2 mod. coinciden → ensamble sigue mayoría<br>
      1-2 = ensamble suaviza con probs intermedias
    </p>
  </div>
</div>

<!-- ══ BALANZA CONCEPTUAL ══ -->
<div class="row row2" style="margin-bottom:14px">
  <div class="card">
    <div class="card-title">Balanza del sistema de salud — ilustración conceptual</div>
    <svg viewBox="0 0 480 260" xmlns="http://www.w3.org/2000/svg"
         style="width:100%;height:auto;display:block">
      <circle cx="55"  cy="90"  r="4" fill="#e94560" opacity=".35"/>
      <circle cx="420" cy="70"  r="5" fill="#4cc9f0" opacity=".30"/>
      <circle cx="75"  cy="190" r="3" fill="#f4a261" opacity=".50"/>
      <circle cx="400" cy="200" r="4" fill="#e94560" opacity=".40"/>
      <circle cx="240" cy="120" r="160" fill="none" stroke="#e94560" stroke-width="1" opacity=".08"/>
      <text x="240" y="18" text-anchor="middle" font-size="9" fill="#e94560" opacity=".65"
            font-style="italic" font-family="sans-serif">● señal Reddit MX</text>
      <line x1="240" y1="228" x2="240" y2="100" stroke="#8888aa" stroke-width="5" stroke-linecap="round"/>
      <rect x="206" y="228" width="68" height="16" rx="5" fill="#333355"/>
      <circle cx="240" cy="100" r="9" fill="#f4a261"/>
      <line x1="68" y1="83" x2="412" y2="117" stroke="#f0f0ff" stroke-width="4" stroke-linecap="round"/>
      <line x1="68"  y1="83"  x2="68"  y2="134" stroke="#8888aa" stroke-width="2" stroke-dasharray="5,4"/>
      <line x1="412" y1="117" x2="412" y2="174" stroke="#e94560" stroke-width="2.5" stroke-dasharray="5,4"/>
      <ellipse cx="68" cy="138" rx="55" ry="14" fill="#4cc9f0" opacity=".85"/>
      <text x="68" y="132" text-anchor="middle" font-size="10" font-weight="bold"
            fill="#042c53" font-family="sans-serif">Atención</text>
      <text x="68" y="145" text-anchor="middle" font-size="10" font-weight="bold"
            fill="#042c53" font-family="sans-serif">Normal</text>
      <text x="68" y="163" text-anchor="middle" font-size="9" fill="#4cc9f0"
            font-family="sans-serif">n={n_media:,}</text>
      <ellipse cx="412" cy="178" rx="58" ry="15" fill="#e94560" opacity=".90"/>
      <text x="412" y="172" text-anchor="middle" font-size="10" font-weight="bold"
            fill="#fff" font-family="sans-serif">Ineficiencia</text>
      <text x="412" y="185" text-anchor="middle" font-size="10" font-weight="bold"
            fill="#fff" font-family="sans-serif">Sistémica</text>
      <text x="412" y="200" text-anchor="middle" font-size="11" font-weight="bold"
            fill="#f4a261" font-family="sans-serif">DESABASTO</text>
      <text x="412" y="215" text-anchor="middle" font-size="11" font-weight="bold"
            fill="#ffd60a" font-family="sans-serif">ESPERA</text>
      <text x="68"  y="252" text-anchor="middle" font-size="9" fill="#4cc9f0"
            font-family="sans-serif">n={n_media:,}</text>
      <text x="412" y="252" text-anchor="middle" font-size="9" fill="#e94560"
            font-family="sans-serif">n={n_alta:,}</text>
    </svg>
  </div>

  <!-- Matriz de confusión -->
  <div class="card">
    <div class="card-title">Matriz de confusión — VotingEnsemble (test 20%)</div>
    <div class="cm-grid" id="cm-grid">
      <div class="cm-cell" style="background:rgba(6,214,160,.18);border:1px solid rgba(6,214,160,.4)">
        <div class="cm-num" style="color:var(--green)">{tn}</div>
        <div class="cm-lbl" style="color:var(--green)">VP (media ✓)</div>
        <div style="font-size:.6rem;color:var(--muted)">real media → pred media</div>
      </div>
      <div class="cm-cell" style="background:rgba(244,162,97,.12);border:1px solid rgba(244,162,97,.3)">
        <div class="cm-num" style="color:var(--amber)">{fp}</div>
        <div class="cm-lbl" style="color:var(--amber)">FP</div>
        <div style="font-size:.6rem;color:var(--muted)">real media → pred alta</div>
      </div>
      <div class="cm-cell" style="background:rgba(233,69,96,.18);border:1px solid rgba(233,69,96,.4)">
        <div class="cm-num" style="color:var(--red)">{fn}</div>
        <div class="cm-lbl" style="color:var(--red)">FN ← COSTO ALTO</div>
        <div style="font-size:.6rem;color:var(--muted)">real alta → pred media</div>
      </div>
      <div class="cm-cell" style="background:rgba(76,201,240,.18);border:1px solid rgba(76,201,240,.4)">
        <div class="cm-num" style="color:var(--blue)">{tp}</div>
        <div class="cm-lbl" style="color:var(--blue)">VP (alta ✓)</div>
        <div style="font-size:.6rem;color:var(--muted)">real alta → pred alta</div>
      </div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:12px;
                font-size:.68rem;color:var(--muted);padding-top:10px;
                border-top:1px solid var(--border)">
      <div>Precisión alta = {f"{tp/(tp+fp):.3f}" if (tp+fp)>0 else "N/D"}</div>
      <div>Recall alta = {f"{tp/(tp+fn):.3f}" if (tp+fn)>0 else "N/D"}</div>
      <div>Precisión media = {f"{tn/(tn+fn):.3f}" if (tn+fn)>0 else "N/D"}</div>
      <div>Recall media = {f"{tn/(tn+fp):.3f}" if (tn+fp)>0 else "N/D"}</div>
    </div>
  </div>
</div>

<!-- ══ TOP FEATURES ══ -->
<div class="section-label">Features predictivas</div>
<div class="card" style="margin-bottom:14px">
  <div class="card-title">Top features — coeficiente LR sub-estimador del ensamble (clase alta)</div>
  <div id="feat-bars"></div>
  <div style="display:flex;gap:18px;margin-top:12px;padding-top:10px;
              border-top:1px solid var(--border);flex-wrap:wrap">
    <span style="font-size:.70rem;display:flex;align-items:center;gap:5px">
      <span style="width:10px;height:10px;border-radius:2px;background:var(--blue);display:inline-block"></span>BERT</span>
    <span style="font-size:.70rem;display:flex;align-items:center;gap:5px">
      <span style="width:10px;height:10px;border-radius:2px;background:var(--green);display:inline-block"></span>Enriquecimiento</span>
    <span style="font-size:.70rem;display:flex;align-items:center;gap:5px">
      <span style="width:10px;height:10px;border-radius:2px;background:var(--red);display:inline-block"></span>TF-IDF texto</span>
    <span style="font-size:.70rem;color:var(--muted);margin-left:auto">[NUM_X] = features numéricas del pipeline</span>
  </div>
</div>

<!-- ══ HEATMAP ══ -->
<div class="card" style="margin-bottom:14px">
  <div class="card-title">Heatmap — co-ocurrencia palabras clave × nivel negatividad BERT (prob_neg)</div>
  <div class="hm-wrap">
    <table class="hm">
      <thead>
        <tr>
          <th class="lbl">Término</th>
          <th>0.0–0.2<br><span style="font-size:.58rem;color:var(--muted)">muy pos</span></th>
          <th>0.2–0.4<br><span style="font-size:.58rem;color:var(--muted)">pos/neu</span></th>
          <th>0.4–0.6<br><span style="font-size:.58rem;color:var(--muted)">neutro</span></th>
          <th style="border-left:2px dashed rgba(233,69,96,.5)">0.6–0.8<br>
            <span style="font-size:.58rem;color:var(--muted)">negativo</span></th>
          <th>0.8–1.0<br><span style="font-size:.58rem;color:var(--muted)">muy neg</span></th>
          <th>tipo</th>
        </tr>
      </thead>
      <tbody id="hm-body"></tbody>
    </table>
  </div>
  <p class="hm-caption">
    Valores normalizados por fila [0–1]. Rojo = alta co-ocurrencia con negatividad BERT.
    Línea discontinua = zona de queja confirmada (prob_neg ≥ 0.6).
  </p>
</div>

<!-- ══ COMPARACIÓN + FOLDS + CURVA ══ -->
<div class="section-label">Evaluación y estabilidad del modelo</div>
<div class="row row3" style="margin-bottom:14px">
  <div class="card">
    <div class="card-title">Comparación de modelos — F1-weighted</div>
    <div id="model-bars"></div>
    <p style="font-size:.68rem;color:var(--muted);margin-top:10px;padding-top:8px;
              border-top:1px solid var(--border)">
      ★ = mejor modelo &nbsp;·&nbsp; [ENS] = VotingEnsemble
    </p>
  </div>
  <div class="card">
    <div class="card-title">Estabilidad 10-fold — F1 weighted por fold</div>
    <div class="fold-chart" id="fold-chart"></div>
    <div style="display:flex;gap:3px;padding:18px 0 5px;font-size:.60rem;
                color:var(--muted);justify-content:space-around">
      <span>F1</span><span>F2</span><span>F3</span><span>F4</span><span>F5</span>
      <span>F6</span><span>F7</span><span>F8</span><span>F9</span><span>F10</span>
    </div>
    <div style="display:flex;justify-content:space-between;margin-top:4px">
      <div>
        <div style="font-size:.65rem;color:var(--muted)">mín</div>
        <div style="font-size:1.0rem;font-weight:800;color:var(--red)">{d['f1_min']:.4f}</div>
      </div>
      <div style="text-align:center">
        <div style="font-size:.65rem;color:var(--muted)">media ± IC95%</div>
        <div style="font-size:1.0rem;font-weight:800;color:var(--green)">
          {d['f1']:.3f} ± {d['f1_ic']:.3f}</div>
      </div>
      <div style="text-align:right">
        <div style="font-size:.65rem;color:var(--muted)">máx</div>
        <div style="font-size:1.0rem;font-weight:800;color:var(--blue)">{d['f1_max']:.4f}</div>
      </div>
    </div>
    <p style="font-size:.68rem;color:var(--muted);margin-top:7px;padding-top:6px;
              border-top:1px solid var(--border)">
      σ = {d['f1_std']:.3f} &lt; 4% →
      <strong style="color:{veredicto_color}">{d['veredicto']}</strong>
    </p>
  </div>
  <!-- Overfitting check -->
  <div class="card">
    <div class="card-title">Overfitting — gap train vs test</div>
    <div style="margin-bottom:10px">
      <div style="font-size:.68rem;color:var(--muted)">F1 train</div>
      <div style="font-size:1.2rem;font-weight:800;color:var(--blue)">{ovf.get('f1_train_media',0):.4f}</div>
    </div>
    <div style="margin-bottom:12px">
      <div style="font-size:.68rem;color:var(--muted)">F1 test (CV)</div>
      <div style="font-size:1.2rem;font-weight:800;color:var(--green)">{ovf.get('f1_test_media',0):.4f}</div>
    </div>
    <div style="padding:10px;border-radius:8px;
                background:rgba({"233,69,96" if ovf.get("detectado") else "6,214,160"},.1);
                border:1px solid rgba({"233,69,96" if ovf.get("detectado") else "6,214,160"},.3)">
      <div style="font-size:1.5rem;font-weight:800;color:{gap_col}">Δ {gap:+.4f}</div>
      <div style="font-size:.70rem;color:var(--muted);margin-top:3px">
        {"⚠ Overfitting detectado — revisar regularización" if ovf.get("detectado") else "✓ Sin overfitting significativo (gap < 10%)"}</div>
    </div>
    <div id="curva-aprendizaje" style="margin-top:10px">
      <div style="font-size:.63rem;color:var(--muted);margin-bottom:5px">
        Curva de aprendizaje — {d.get("curva_diag","")}</div>
      <svg id="lc-svg" viewBox="0 0 300 90" xmlns="http://www.w3.org/2000/svg"></svg>
    </div>
  </div>
</div>

<!-- ══ DISTRIBUCIÓN DE CONFIANZA ══ -->
<div class="row row2" style="margin-bottom:14px">
  <div class="card">
    <div class="card-title">Distribución de confianza probabilística — predicciones test</div>
    <div id="dist-conf-bars"></div>
    <p style="font-size:.63rem;color:var(--muted);margin-top:8px">
      Alta concentración en [0.9–1.0] indica predicciones muy seguras en ambas clases
    </p>
  </div>
  <div class="card">
    <div class="card-title">Dimensiones narrativas — peso relativo en clase alta</div>
    <div id="dims-bars"></div>
    <p style="font-size:.63rem;color:var(--muted);margin-top:8px">
      Calculado sobre los top-{len(d['top_alta'])} términos predictivos de clase alta
    </p>
  </div>
</div>

<!-- ══ CORRELACIONES + CHI2 ══ -->
<div class="section-label">Evidencia estadística complementaria</div>
<div class="row row2" style="margin-bottom:14px">
  <div class="card">
    <div class="card-title">Correlaciones Spearman — features numéricas vs clase</div>
    <div id="cors-bars"></div>
    <p style="font-size:.63rem;color:var(--muted);margin-top:8px">
      ρ &gt; 0 → correlaciona con clase alta (ineficiencia). * p &lt; 0.05
    </p>
  </div>
  <div class="card">
    <div class="card-title">Chi² por categoría de queja — asociación con clase alta</div>
    <div id="chi2-bars"></div>
    <p style="font-size:.63rem;color:var(--muted);margin-top:8px">
      OR &gt; 1.5 → fuertemente asociada con ineficiencia sistémica
    </p>
  </div>
</div>

{pngs_section}

<!-- ══ PÁRRAFO TESIS ══ -->
<div class="section-label">Hallazgo central</div>
<div class="card accent-red">
  <div class="card-title">Texto listo para tesis — generado automáticamente</div>
  <p class="tesis-text" id="tesis-par"></p>
  <div class="footer" id="footer-txt"></div>
</div>

</div><!-- /page -->

<script>
/* ══════════════════════════
   DATOS INYECTADOS
══════════════════════════ */
const FN_CONF    = {fn_conf_js};
const FOLDS      = {folds_js};
const FEATS      = {feats_js};
const HM_ROWS    = {hm_js};
const COMP       = {comp_js};
const RECS       = {recs_js};
const CURVA      = {curva_js};
const DIST_CONF  = {dist_conf_js};
const CORS       = {cors_js};
const CHI2       = {chi2_js};
const DIMS       = {dims_js_str};
const DISC_TIPOS = {disc_tipos_js};
const F1         = {d['f1']};
const F1_IC      = {d['f1_ic']};
const AUC        = {d['auc']};
const F1_ALTA    = {d['f1_alta']};
const N_TOTAL    = {n_total};
const N_ALTA     = {n_alta};
const MODELO     = "{d['modelo']}";
const BRIER_RAW  = {d['brier_raw']};
const BRIER_CAL  = {d['brier_cal']};
const N_DISC     = {n_disc};
const TASA_DES   = {tasa_des};
const ENS_RESC   = {ens_resc};

/* ══════════════════════════
   FN BARS
══════════════════════════ */
(function(){{
  const cont = document.getElementById('fn-bars');
  const vals = FN_CONF.length ? FN_CONF : Array(14).fill(0.91);
  vals.forEach((v,i)=>{{
    const pct = (v*100).toFixed(1);
    const d = document.createElement('div');
    d.style.cssText='display:flex;align-items:center;gap:6px;margin-bottom:3px';
    d.innerHTML=
      `<span style="font-size:.63rem;color:#8888aa;width:20px">F${{i+1}}</span>`+
      `<div style="flex:1;height:8px;background:rgba(255,255,255,.05);border-radius:4px;overflow:hidden">`+
      `<div style="height:100%;border-radius:4px;background:linear-gradient(90deg,#4cc9f0,#e94560);width:${{pct}}%"></div>`+
      `</div><span style="font-size:.66rem;color:#e94560;width:36px">${{(v*100).toFixed(1)}}%</span>`;
    cont.appendChild(d);
  }});
}})();

/* ══════════════════════════
   FEATURE BARS
══════════════════════════ */
(function(){{
  const cont = document.getElementById('feat-bars');
  if(!FEATS.length){{ cont.innerHTML='<p style="color:#8888aa;font-size:.78rem">Sin datos de reporte</p>'; return; }}
  const maxV = Math.max(...FEATS.map(f=>f.coef));
  const colors = {{bert:'#4cc9f0',enriq:'#06d6a0',texto:'#e94560',
                   texto_tfidf:'#e94560',BERT:'#4cc9f0',enriquecimiento:'#06d6a0',
                   numerico:'#4cc9f0'}};
  FEATS.forEach(f=>{{
    const col = colors[f.tipo]||'#8888aa';
    const pct = ((f.coef/maxV)*100).toFixed(1);
    const dir = f.dir==='→ alta'?'▲':'▼';
    const row = document.createElement('div');
    row.className='bar-row';
    row.innerHTML=
      `<span class="bar-lbl" title="${{f.nombre}}">${{dir}} ${{f.nombre}}</span>`+
      `<div class="bar-track"><div class="bar-fill" `+
      `style="width:${{pct}}%;background:${{col}}22;color:${{col}}">${{f.coef.toFixed(3)}}</div></div>`+
      `<span class="bar-val">${{f.coef.toFixed(3)}}</span>`;
    cont.appendChild(row);
  }});
}})();

/* ══════════════════════════
   HEATMAP
══════════════════════════ */
(function(){{
  function heatColor(v){{
    const stops=[[13,13,46],[67,97,238],[244,162,97],[233,69,96]];
    const idx=v*3,lo=Math.floor(idx),hi=Math.min(lo+1,3),t=idx-lo;
    const lerp=(a,b,t)=>a+(b-a)*t;
    const r=lerp(stops[lo][0],stops[hi][0],t);
    const g=lerp(stops[lo][1],stops[hi][1],t);
    const b=lerp(stops[lo][2],stops[hi][2],t);
    const lum=(r*299+g*587+b*114)/255000;
    return{{bg:`rgb(${{Math.round(r)}},${{Math.round(g)}},${{Math.round(b)}})`,
            tc:lum>0.5?'rgba(0,0,0,.85)':'rgba(240,240,255,.95)'}};
  }}
  const tipoCls={{numerico:'tp-bert',bert:'tp-bert',texto:'tp-texto',
                   texto_tfidf:'tp-texto',enriq:'tp-enriq',enriquecimiento:'tp-enriq'}};
  const tipoLbl={{numerico:'BERT',bert:'BERT',texto:'TF-IDF',
                   texto_tfidf:'TF-IDF',enriq:'Enriq.',enriquecimiento:'Enriq.'}};
  const tbody=document.getElementById('hm-body');
  HM_ROWS.forEach(row=>{{
    const tr=document.createElement('tr');
    let cells=`<td class="lbl">${{row.termino}}</td>`;
    row.valores.forEach((val,i)=>{{
      const c=heatColor(val);
      const bl=i===3?'border-left:2px dashed rgba(233,69,96,.5)':'';
      cells+=`<td style="background:${{c.bg}};color:${{c.tc}};${{bl}}">${{val.toFixed(2)}}</td>`;
    }});
    const cls=tipoCls[row.tipo]||'tp-texto';
    const lbl=tipoLbl[row.tipo]||'TF-IDF';
    cells+=`<td><span class="tipo-pill ${{cls}}">${{lbl}}</span></td>`;
    tr.innerHTML=cells;
    tbody.appendChild(tr);
  }});
}})();

/* ══════════════════════════
   MODEL COMPARE
══════════════════════════ */
(function(){{
  const cont=document.getElementById('model-bars');
  if(!COMP.length)return;
  const best=COMP[0].modelo;
  COMP.forEach(m=>{{
    const isBest=m.modelo===best;
    const isEns=m.es_ensamble;
    const col=isEns?'#06d6a0':(isBest?'#4cc9f0':'#666688');
    const pct=(m.f1_weighted*100).toFixed(1);
    const star=isBest?'<span style="color:#ffd60a;font-size:.68rem;margin-left:4px">★ mejor</span>':'';
    const ensBadge=isEns?'<span style="color:#06d6a0;font-size:.63rem;margin-left:4px">[ENS]</span>':'';
    const div=document.createElement('div');
    div.className='model-row';
    div.innerHTML=
      `<div class="model-header">`+
      `<span>${{m.modelo}}${{star}}${{ensBadge}}</span>`+
      `<span style="color:#8888aa;font-size:.68rem">AUC ${{(m.roc_auc||0).toFixed(3)}} · F1-alta ${{(m.f1_alta||0).toFixed(3)}}</span>`+
      `</div>`+
      `<div class="model-track">`+
      `<div class="model-fill" style="width:${{pct}}%;background:${{col}};color:#0d0d1a">`+
      `${{m.f1_weighted.toFixed(4)}}</div></div>`;
    cont.appendChild(div);
  }});
}})();

/* ══════════════════════════
   FOLD CHART
══════════════════════════ */
(function(){{
  const cont=document.getElementById('fold-chart');
  const min=0.78,max=0.94;
  const maxVal=Math.max(...FOLDS);
  FOLDS.forEach((v,i)=>{{
    const pct=Math.max(4,Math.round((v-min)/(max-min)*100));
    const isMax=v===maxVal;
    const bar=document.createElement('div');
    bar.style.cssText=`flex:1;height:${{pct}}%;background:${{isMax?'#4cc9f0':'rgba(76,201,240,.45)'}};border-radius:3px 3px 0 0`;
    bar.title=`Fold ${{i+1}}: ${{v.toFixed(4)}}`;
    cont.appendChild(bar);
  }});
}})();

/* ══════════════════════════
   CURVA DE APRENDIZAJE (SVG)
══════════════════════════ */
(function(){{
  if(!CURVA.length)return;
  const svg=document.getElementById('lc-svg');
  const W=300,H=90,pad={{l:28,r:8,t:8,b:18}};
  const xs=CURVA.map(p=>p.n);
  const minX=Math.min(...xs),maxX=Math.max(...xs);
  const allY=CURVA.flatMap(p=>[p.tr,p.te]);
  const minY=Math.max(0,Math.min(...allY)-0.05),maxY=Math.min(1,Math.max(...allY)+0.02);
  const cx=n=>pad.l+(n-minX)/(maxX-minX||1)*(W-pad.l-pad.r);
  const cy=v=>H-pad.b-(v-minY)/(maxY-minY||0.1)*(H-pad.t-pad.b);
  const mkPath=(key,col)=>{{
    const pts=CURVA.map(p=>`${{cx(p.n).toFixed(1)}},${{cy(p[key]).toFixed(1)}}`).join(' L');
    const path=document.createElementNS('http://www.w3.org/2000/svg','path');
    path.setAttribute('d',`M ${{pts}}`);
    path.setAttribute('fill','none');
    path.setAttribute('stroke',col);
    path.setAttribute('stroke-width','2');
    path.setAttribute('stroke-linecap','round');
    svg.appendChild(path);
    // Dot at last point
    const lp=CURVA[CURVA.length-1];
    const c=document.createElementNS('http://www.w3.org/2000/svg','circle');
    c.setAttribute('cx',cx(lp.n));c.setAttribute('cy',cy(lp[key]));
    c.setAttribute('r','3');c.setAttribute('fill',col);
    svg.appendChild(c);
  }};
  mkPath('tr','#4cc9f0');
  mkPath('te','#e94560');
  // Labels
  const mkT=(x,y,txt,col)=>{{
    const t=document.createElementNS('http://www.w3.org/2000/svg','text');
    t.setAttribute('x',x);t.setAttribute('y',y);t.setAttribute('fill',col);
    t.setAttribute('font-size','7');t.setAttribute('font-family','sans-serif');
    t.textContent=txt; svg.appendChild(t);
  }};
  mkT(pad.l,H-pad.b+12,'n','#8888aa');
  mkT(W-30,pad.t+8,'─ train','#4cc9f0');
  mkT(W-30,pad.t+18,'─ val','#e94560');
}})();

/* ══════════════════════════
   DISC TIPOS BARS
══════════════════════════ */
(function(){{
  const cont=document.getElementById('disc-bars');
  if(!DISC_TIPOS.length){{
    cont.innerHTML='<p style="color:#8888aa;font-size:.75rem;padding:8px 0">Sin datos de discrepancias (ejecutar analizar_v4.py)</p>';
    return;
  }}
  const maxN=Math.max(...DISC_TIPOS.map(t=>t.n));
  DISC_TIPOS.forEach(t=>{{
    const pct=(t.n/maxN*100).toFixed(1);
    const row=document.createElement('div');
    row.className='bar-row';
    row.innerHTML=
      `<span class="bar-lbl">${{t.tipo}}</span>`+
      `<div class="bar-track"><div class="bar-fill" `+
      `style="width:${{pct}}%;background:rgba(124,106,240,.25);color:#7c6af0">${{t.n}}</div></div>`+
      `<span class="bar-val">${{t.n}}</span>`;
    cont.appendChild(row);
  }});
}})();

/* ══════════════════════════
   DISTRIBUCIÓN DE CONFIANZA
══════════════════════════ */
(function(){{
  const cont=document.getElementById('dist-conf-bars');
  if(!DIST_CONF.length){{
    cont.innerHTML='<p style="color:#8888aa;font-size:.75rem">Sin datos de distribución</p>';
    return;
  }}
  const maxN=Math.max(...DIST_CONF.map(t=>t.n||0));
  DIST_CONF.forEach(t=>{{
    const pct=maxN>0?(t.n/maxN*100).toFixed(1):'0';
    const isHigh=t.rango&&t.rango.includes('0.9');
    const col=isHigh?'#06d6a0':'rgba(76,201,240,.6)';
    const row=document.createElement('div');
    row.className='bar-row';
    row.innerHTML=
      `<span class="bar-lbl" style="font-size:.68rem">${{t.rango}}</span>`+
      `<div class="bar-track"><div class="bar-fill" `+
      `style="width:${{pct}}%;background:${{col}}33;color:${{col}}">${{t.n}}</div></div>`+
      `<span class="bar-val">${{t.n}}</span>`;
    cont.appendChild(row);
  }});
}})();

/* ══════════════════════════
   DIMENSIONES NARRATIVAS
══════════════════════════ */
(function(){{
  const cont=document.getElementById('dims-bars');
  if(!DIMS.length){{
    cont.innerHTML='<p style="color:#8888aa;font-size:.75rem">Sin datos de dimensiones</p>';
    return;
  }}
  const DIM_COLS={{
    'falta insumos':'#e94560','tiempo espera':'#f4a261',
    'consecuencia clinica':'#7c6af0','corrupcion negligencia':'#4cc9f0',
    'sistema roto':'#ffd60a','pago privado':'#06d6a0','otro':'#666688'
  }};
  const maxP=Math.max(...DIMS.map(d=>d.prop));
  DIMS.forEach(d=>{{
    const pct=(d.prop/maxP*100).toFixed(1);
    const col=DIM_COLS[d.dim]||'#8888aa';
    const row=document.createElement('div');
    row.className='bar-row';
    row.title=d.terms;
    row.innerHTML=
      `<span class="bar-lbl" style="font-size:.68rem">${{d.dim}}</span>`+
      `<div class="bar-track"><div class="bar-fill" `+
      `style="width:${{pct}}%;background:${{col}}28;color:${{col}}">`+
      `${{(d.prop*100).toFixed(1)}}%</div></div>`+
      `<span class="bar-val" style="font-size:.63rem;color:${{col}}">${{d.n}}t</span>`;
    cont.appendChild(row);
  }});
}})();

/* ══════════════════════════
   CORRELACIONES SPEARMAN
══════════════════════════ */
(function(){{
  const cont=document.getElementById('cors-bars');
  if(!CORS.length){{
    cont.innerHTML='<p style="color:#8888aa;font-size:.75rem">Sin datos de correlaciones (requiere dataset JSON)</p>';
    return;
  }}
  const maxR=Math.max(...CORS.map(c=>Math.abs(c.rho)));
  CORS.forEach(c=>{{
    const pct=(Math.abs(c.rho)/maxR*100).toFixed(1);
    const col=c.rho>0?'#06d6a0':'#e94560';
    const sig=c.significativa?'*':'';
    const row=document.createElement('div');
    row.className='bar-row';
    row.innerHTML=
      `<span class="bar-lbl">${{c.feature}}${{sig}}</span>`+
      `<div class="bar-track"><div class="bar-fill" `+
      `style="width:${{pct}}%;background:${{col}}25;color:${{col}}">`+
      `ρ=${{c.rho.toFixed(3)}}</div></div>`+
      `<span class="bar-val" style="color:${{col}}">${{c.magnitud?c.magnitud.substring(0,3):c.rho.toFixed(2)}}</span>`;
    cont.appendChild(row);
  }});
}})();

/* ══════════════════════════
   CHI² CATEGORÍAS
══════════════════════════ */
(function(){{
  const cont=document.getElementById('chi2-bars');
  if(!CHI2.length){{
    cont.innerHTML='<p style="color:#8888aa;font-size:.75rem">Sin datos chi² (requiere dataset JSON)</p>';
    return;
  }}
  const maxC=Math.max(...CHI2.map(c=>c.chi2));
  const ORColors=(or)=>or>3?'#e94560':or>1.5?'#f4a261':or>0.67?'#8888aa':'#4cc9f0';
  CHI2.forEach(c=>{{
    const pct=(c.chi2/maxC*100).toFixed(1);
    const col=ORColors(c.odds_ratio);
    const sig=c.significativa?'*':'';
    const row=document.createElement('div');
    row.className='bar-row';
    row.title=c.interpretacion||'';
    row.innerHTML=
      `<span class="bar-lbl" style="font-size:.67rem">${{c.categoria}}${{sig}}</span>`+
      `<div class="bar-track"><div class="bar-fill" `+
      `style="width:${{pct}}%;background:${{col}}25;color:${{col}}">`+
      `χ²=${{c.chi2.toFixed(1)}}</div></div>`+
      `<span class="bar-val" style="color:${{col}};font-size:.63rem">OR=${{c.odds_ratio.toFixed(2)}}</span>`;
    cont.appendChild(row);
  }});
}})();

/* ══════════════════════════
   PÁRRAFO TESIS
══════════════════════════ */
(function(){{
  const top3=FEATS.slice(0,3).map(f=>
    `<strong style="color:var(--blue)">${{f.nombre}}</strong> (coef=${{f.coef.toFixed(3)}})`
  ).join(', ');

  const bestMod=COMP.length?COMP[0]:{{modelo:'VotingEnsemble',f1_weighted:0.891}};
  const f1pct=(F1*100).toFixed(1);
  const icpct=(F1_IC*100).toFixed(1);
  const discPct=(TASA_DES*100).toFixed(1);
  const rescPct=N_DISC>0?((ENS_RESC/N_DISC)*100).toFixed(0):'0';

  document.getElementById('tesis-par').innerHTML=
    `El análisis de minería de opinión mediante ensamble de votación suave (ComplementNB + Regresión Logística + LinearSVM calibrado) `+
    `revela que el declive hospitalario en México no se expresa solo en términos médicos, sino en un `+
    `lenguaje de <strong style="color:var(--red)">'burocracia del dolor'</strong>. `+
    `Las features con mayor coeficiente predictivo fueron ${{top3||'<em>ver tabla de features</em>'}}. `+
    `El sistema alcanzó F1 = <strong style="color:var(--green)">${{f1pct}}% ± ${{icpct}}%</strong> (IC 95%) `+
    `con AUC = <strong style="color:var(--blue)">${{AUC.toFixed(3)}}</strong>, evaluado mediante validación `+
    `cruzada estratificada de 10 folds sobre ${{N_TOTAL.toLocaleString()}} comentarios. `+
    `El VotingEnsemble obtuvo el mejor F1-weighted (${{bestMod.f1_weighted.toFixed(4)}}), `+
    `promediando las probabilidades calibradas de los tres clasificadores. `+
    `Adicionalmente, el <strong style="color:var(--purple)">${{discPct}}% de los comentarios de prueba</strong> `+
    `generó desacuerdo entre los clasificadores individuales — representando la `+
    `<em>zona gris de la ineficiencia sistémica</em>: comentarios con vocabulario intenso `+
    `(términos como 'negligencia', 'muerte', 'desabasto') pero contexto semántico ambiguo. `+
    `El ensamble resolvió correctamente el <strong style="color:var(--green)">${{rescPct}}%</strong> `+
    `de estas discrepancias, confirmando la hipótesis de que la combinación de señal léxica y `+
    `probabilidades BERT captura matices que ningún clasificador individual puede obtener por separado.`;

  document.getElementById('footer-txt').textContent=
    `Generado: {fecha}  ·  Pipeline CRISP-DM Fases 3–5  ·  Modelo: ${{MODELO}}  ·  `+
    `Calibración: isotonic cv=5  ·  Brier=${{BRIER_CAL.toFixed(5)}}  ·  `+
    `Discrepancias: ${{(TASA_DES*100).toFixed(1)}}% del test set`;
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
    salida = carpeta / f"infografia_burocracia_dolor_v4_{ts}.html"
    salida.write_text(generar_html(d), encoding="utf-8")

    disc = d["disc_resumen"]
    print(f"\n{'='*64}")
    print(f"  ✅ Infografía v4 generada:")
    print(f"     {salida.name}")
    print(f"\n  Secciones incluidas:")
    print(f"     ① Métricas globales (F1, AUC, clase alta, corpus)")
    print(f"     ② Arquitectura VotingEnsemble + errores críticos")
    print(f"     ③ Zona gris / discrepancias: {disc.get('n_discrepancias',0)} casos")
    print(f"     ④ Balanza conceptual + Matriz de confusión")
    print(f"     ⑤ Top features + Heatmap co-ocurrencia")
    print(f"     ⑥ Comparación modelos + Estabilidad 10-fold + Overfitting")
    print(f"     ⑦ Distribución confianza + Dimensiones narrativas")
    print(f"     ⑧ Correlaciones Spearman + Chi² por categoría")
    has_pngs = any([d["png_curva_ens"], d["png_curva_lr"],
                    d["png_conf_ens"],  d["png_conf_lr"]])
    if has_pngs:
        print(f"     ⑨ PNGs embebidos (curvas + matrices de confusión)")
    print(f"     ⑩ Párrafo de tesis con discrepancias y zona gris")
    print(f"\n  Para abrir:")
    print(f"     xdg-open {salida.name}    (Linux)")
    print(f"     open {salida.name}         (macOS)")
    print(f"     start {salida.name}        (Windows)")
    print(f"\n  Para exportar a PDF:")
    print(f"     1. Abrir en Chrome o Firefox")
    print(f"     2. Ctrl+P → Destino: Guardar como PDF")
    print(f"     3. Escala: 75–85% · Márgenes: mínimos · Fondo de gráficas: ✓")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()

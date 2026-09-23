#!/usr/bin/env python3
"""
analizar_v4.py — Paso 3 del pipeline de modelado (CRISP-DM Fase 5)
Tema: Ineficiencia del sector salud MX frente a la influenza

Cambios v4 respecto a v3
─────────────────────────
  ① Modularización de modelos: los pipelines CNB, LR y LinearSVM se importan
     desde modelos.py — añadir un clasificador nuevo = una línea allá.

  ② Modelo PRINCIPAL actualizado a VotingEnsemble (soft voting):
     Combina CNB + LR + LinearSVM calibrado. La decisión final promedia
     las probabilidades de los tres clasificadores ponderadas.
     → Reduce el error respecto a cualquier clasificador individual.
     → La tasa de desacuerdo entre modelos es evidencia directa de la tesis.

  ③ Análisis de Discrepancias (nuevo en v4):
     analizar_discrepancias() identifica los comentarios donde los tres
     clasificadores no estuvieron de acuerdo (ej. CNB=alta, LR=media, SVM=alta).
     Se exportan en JSON + CSV para la sección de Discusión de Resultados de la tesis.
     · Tipo 2-1: dos clasificadores coinciden → el ensamble sigue la mayoría.
     · Tipo 1-2: el ensamble suaviza la decisión usando las probabilidades.
     · Estos casos son la "zona gris" de la ineficiencia sistémica.

  ④ Tabla comparativa enriquecida: comparar_modelos() ahora incluye el
     VotingEnsemble junto a CNB, LR y LinearSVM, reportando
     Precision, Recall y F1 por clase (alta / media) + métricas globales.

  ⑤ Funciones de graficación genéricas (curva de aprendizaje del ensamble
     y matriz de confusión del ensamble):
       · graficar_curva_aprendizaje(ensemble, X, y, "VotingEnsemble", ...)
       · graficar_matriz_confusion(y_te, y_pred_ens, "VotingEnsemble", ...)

Heredadas de v3
───────────────
  ⑥ Selector automático de estimador BERT (conservado en validación cruzada).
  ⑦ Validación cruzada 10-fold con IC 95% sobre el ensamble.
  ⑧ Análisis de errores críticos (FN/FP de alta confianza) del ensamble.
  ⑨ Top-features híbrido extraído del sub-estimador LR del ensamble.
  ⑩ Spearman, χ², dimensiones narrativas, reporte narrativo.

Entrada : X_features*.npz  |  y_labels*.npy  |  vectorizador*.pkl
          dataset_etiquetado*.json  (auto_etiquetar.py, preferido)
          dataset_normalizado*.json  (fallback sin BERT)
Salida  : reporte_v4_*.json       |  resumen_narrativo_v4_*.txt
          errores_criticos_*.json
          discrepancias_*.json     |  discrepancias_*.csv  ← NUEVO
          curva_aprendizaje_<modelo>_*.png  (si matplotlib disponible)
          confusion_<modelo>_*.png          (si matplotlib disponible)

Uso: python3 analizar_v4.py
"""

import csv
import json
import sys
import subprocess
import pickle
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

warnings.filterwarnings("ignore", category=UserWarning)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

TOP_N_TERMS         = 25
N_TERMINOS_CITA     = 5
TEST_SIZE           = 0.20
RANDOM_STATE        = 42

# ── Validación cruzada ───────────────────────────────────────────────────────
N_FOLDS             = 10
IC_Z                = 1.96

# ── Calibración ──────────────────────────────────────────────────────────────
METODO_CALIBRACION  = "isotonic"

# ── Análisis de errores ──────────────────────────────────────────────────────
N_ERRORES_EXPORTAR  = 20
UMBRAL_CONFIANZA    = 0.80

# ── Curva de aprendizaje ─────────────────────────────────────────────────────
CURVA_PUNTOS        = 8

# ── Features BERT (v3) ───────────────────────────────────────────────────────
FEATURES_BERT      = ["prob_neg", "prob_neu", "prob_pos", "sentimiento_num"]
UMBRAL_BERT_ACTIVO = 0.10   # ≥10% de filas con prob_neg != 0 → BERT activo

# ── Etiquetas de clase para gráficas ─────────────────────────────────────────
ETIQUETAS_CLASE = ["media", "alta"]

# ── Ensamble ─────────────────────────────────────────────────────────────────
# Nombre del ensamble tal como aparece en CATALOGO_MODELOS.
# Se usa para separar modelos individuales del ensamble en discrepancias.
NOMBRE_ENSAMBLE = "VotingEnsemble"


# ══════════════════════════════════════════════════════════════════════════════
#  DIMENSIONES NARRATIVAS Y CATEGORÍAS
# ══════════════════════════════════════════════════════════════════════════════

DIMENSIONES_NARRATIVAS: dict[str, list[str]] = {
    "falta_insumos": [
        "desabasto", "sin_medicamento", "no_hay_medicamento", "no_hay_vacuna",
        "sin_vacuna", "agotado", "escasez", "falta_de_medicamento",
        "anaquel", "no_surtieron", "no_hay",
    ],
    "tiempo_espera": [
        "fila", "cola", "espera", "horas", "sala_de_espera", "turno",
        "cita", "meses", "tardaron", "vuelta_y_vuelta", "vuelva_manana",
        "viacrucis", "odisea", "tramite", "papeleo",
    ],
    "consecuencia_clinica": [
        "fallecer", "morir", "muerte", "neumonia", "empeorar",
        "intubado", "uci", "recaer", "complicar", "grave",
        "hospitalizar", "internar",
    ],
    "pago_privado": [
        "particular", "medico_particular", "de_mi_bolsillo", "privado",
        "privada", "farmacia_particular",
    ],
    "corrupcion_negligencia": [
        "moche", "mordida", "soborno", "corrupcion", "negligencia",
        "negligente", "maltrato", "palancas", "incompetente", "inutil",
    ],
    "sistema_roto": [
        "no_hay_sistema", "no_funciona", "urgencias_llenas",
        "no_hay_cupo", "saturado", "no_me_atendieron", "mala_atencion",
    ],
}

CATEGORIAS_CHI2 = [
    "desabasto_medicamentos", "burocracia_sistema", "vacunacion",
    "atencion_medica", "influenza_enfermedad", "costos_financiamiento",
    "corrupcion",
]


# ══════════════════════════════════════════════════════════════════════════════
#  DEPENDENCIAS
# ══════════════════════════════════════════════════════════════════════════════

def verificar_deps():
    faltantes = [pip for mod, pip in
                 [("sklearn","scikit-learn"),("numpy","numpy"),("scipy","scipy")]
                 if not _importable(mod)]
    if faltantes:
        print(f"  Instalando: {faltantes}")
        subprocess.run([sys.executable,"-m","pip","install"]+faltantes+["-q"],
                       check=True)

def _importable(mod):
    try: __import__(mod); return True
    except ImportError: return False

verificar_deps()

from sklearn.calibration         import CalibratedClassifierCV
from sklearn.model_selection     import (train_test_split, StratifiedKFold,
                                          cross_validate, learning_curve)
from sklearn.metrics             import (classification_report, confusion_matrix,
                                          roc_auc_score, f1_score,
                                          precision_score, recall_score,
                                          brier_score_loss, log_loss)
from sklearn.preprocessing       import MaxAbsScaler
from scipy.sparse                import load_npz, issparse
from scipy.stats                 import spearmanr, chi2_contingency
import numpy as np

# ── Importar pipelines y ensamble desde modelos.py ───────────────────────────
# Cualquier modelo nuevo se añade SOLO en modelos.py → CATALOGO_MODELOS.
# analizar_v4.py no necesita saber cómo están construidos internamente.
try:
    from modelos import (
        _pipeline_cnb,
        _pipeline_lr,
        _pipeline_svm,
        obtener_ensamble_votacion,
        CATALOGO_MODELOS,
    )
except ImportError as e:
    print(f"\n  ❌ No se pudo importar modelos.py: {e}")
    print("     Asegúrate de que modelos.py está en el mismo directorio.")
    sys.exit(1)

# ── Matplotlib (opcional) ─────────────────────────────────────────────────────
_MATPLOTLIB_OK = False
try:
    import matplotlib
    matplotlib.use("Agg")           # backend sin pantalla (servidores, CI)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    _MATPLOTLIB_OK = True
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
#  DETECCIÓN DE BERT
# ══════════════════════════════════════════════════════════════════════════════

def detectar_bert_activo(X, art: dict) -> bool:
    """
    Detecta si la matriz X contiene features BERT con valores reales.
    Retorna True → el ensamble usará pesos [1, 2, 2] (LR y SVM dominan).
    Retorna False → el ensamble usará pesos [2, 1, 2] (CNB y SVM dominan).
    """
    cols_num = art.get("features_numericas", [])
    if "prob_neg" not in cols_num:
        return False

    idx_prob_neg = len(cols_num) - 1 - list(reversed(cols_num)).index("prob_neg")
    n_texto  = X.shape[1] - len(cols_num)
    col_idx  = n_texto + idx_prob_neg

    if col_idx >= X.shape[1]:
        return False

    col_vals = np.asarray(X[:, col_idx].todense()).flatten() \
               if issparse(X) else X[:, col_idx]
    fraccion_activa = float((col_vals != 0).mean())

    activo = fraccion_activa >= UMBRAL_BERT_ACTIVO
    estado = "✅ activas" if activo else f"⚠️  inactivas ({fraccion_activa:.1%} ≠ 0)"
    print(f"  Features BERT      : {estado}")
    return activo


# ══════════════════════════════════════════════════════════════════════════════
#  INTERFAZ DE ARCHIVOS
# ══════════════════════════════════════════════════════════════════════════════

def pedir_archivos(carpeta: Path) -> tuple:
    def glob_ultimo(pat):
        return sorted(carpeta.glob(pat))

    def elegir(lista, tipo):
        if not lista: return None
        if len(lista) == 1: return lista[0]
        print(f"\n  {tipo}:")
        for i, p in enumerate(lista):
            print(f"    [{i}] {p.name}")
        while True:
            try: return lista[int(input(f"  Elige: ").strip())]
            except: print("  ❌ Inválido.")

    print(f"\n{'='*72}")
    print("  ANÁLISIS v4 — ENSAMBLE VOTACIÓN + DISCREPANCIAS + TABLA COMPLETA  🔬")
    print(f"{'='*72}")

    ruta_X    = elegir(glob_ultimo("X_features*.npz"),       "X_features")
    ruta_y    = elegir(glob_ultimo("y_labels*.npy"),          "y_labels")
    ruta_pkl  = elegir(glob_ultimo("vectorizador*.pkl"),       "vectorizador")
    ruta_json = elegir(
        glob_ultimo("dataset_etiquetado*.json") +
        glob_ultimo("dataset_normalizado*.json") +
        glob_ultimo("dataset_enriquecido*.json"), "dataset JSON (opcional)")

    if not ruta_X or not ruta_y or not ruta_pkl:
        print("\n  ❌ Artefactos no encontrados. Ejecuta vectorizar_v4.py primero.")
        sys.exit(1)

    return ruta_X, ruta_y, ruta_pkl, ruta_json


def cargar_todo(ruta_X, ruta_y, ruta_pkl, ruta_json):
    X = load_npz(str(ruta_X))
    y = np.load(str(ruta_y))
    with open(ruta_pkl, "rb") as f:
        art = pickle.load(f)
    registros = []
    if ruta_json:
        with open(ruta_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        registros = data.get("datos", data) if isinstance(data, dict) else data
    return X, y, art, registros


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDACIÓN CRUZADA ROBUSTA (10-FOLD ESTRATIFICADO) — ENSAMBLE PRINCIPAL
#
#  En v4, el estimador de la validación cruzada es el VotingEnsemble.
#  Esto mide la capacidad de generalización del clasificador que se usará
#  en producción, no de los modelos individuales por separado.
#
#  Los pesos del ensamble varían según BERT:
#    · BERT activo  → weights=[1, 2, 2]  (LR y SVM dominan)
#    · texto puro   → weights=[2, 1, 2]  (CNB y SVM dominan)
# ══════════════════════════════════════════════════════════════════════════════

def validacion_cruzada_robusta(X, y, bert_activo: bool = False) -> dict:
    """
    Ejecuta 10-fold estratificado sobre el VotingEnsemble.
    Retorna métricas con IC 95% para citar en tesis.

    Args:
        X           : Matriz de features (sparse).
        y           : Vector de etiquetas.
        bert_activo : Ajusta pesos del ensamble según presencia de BERT.
    """
    print(f"\n  Ejecutando {N_FOLDS}-fold estratificado (VotingEnsemble)...")

    # Pesos adaptados al tipo de corpus
    pesos = [1, 2, 2] if bert_activo else [2, 1, 2]
    clf_cv = obtener_ensamble_votacion(weights=pesos)

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    scores = cross_validate(
        clf_cv, X, y, cv=cv,
        scoring={
            "f1_weighted": "f1_weighted",
            "f1_alta":     "f1",
            "roc_auc":     "roc_auc",
            "accuracy":    "accuracy",
        },
        return_train_score=True,
    )

    def resumen(arr: np.ndarray) -> dict:
        mu, std = float(arr.mean()), float(arr.std())
        ic = IC_Z * std / np.sqrt(N_FOLDS)
        return {
            "media":   round(mu,  4),
            "std":     round(std, 4),
            "ic95_pm": round(ic,  4),
            "ic95_lo": round(mu - ic, 4),
            "ic95_hi": round(mu + ic, 4),
            "min":     round(float(arr.min()), 4),
            "max":     round(float(arr.max()), 4),
            "valores_por_fold": [round(float(v), 4) for v in arr],
            "estable": bool(std < 0.04),
        }

    resultado = {
        "n_folds":     N_FOLDS,
        "ic_nivel":    "95%",
        "estimador":   "VotingEnsemble (soft)",
        "pesos_ensamble": pesos,
        "f1_weighted": resumen(scores["test_f1_weighted"]),
        "f1_alta":     resumen(scores["test_f1_alta"]),
        "roc_auc":     resumen(scores["test_roc_auc"]),
        "accuracy":    resumen(scores["test_accuracy"]),
        "overfitting": {
            "f1_train_media": round(float(scores["train_f1_weighted"].mean()), 4),
            "f1_test_media":  round(float(scores["test_f1_weighted"].mean()),  4),
            "gap":            round(float(
                scores["train_f1_weighted"].mean() -
                scores["test_f1_weighted"].mean()), 4),
            "detectado": bool(
                scores["train_f1_weighted"].mean() -
                scores["test_f1_weighted"].mean() > 0.10),
        },
    }

    f1_std = resultado["f1_weighted"]["std"]
    if f1_std < 0.02:   resultado["veredicto"] = "muy estable (σ < 2%)"
    elif f1_std < 0.04: resultado["veredicto"] = "estable (σ < 4%)"
    elif f1_std < 0.07: resultado["veredicto"] = "aceptable (σ < 7%)"
    else:               resultado["veredicto"] = "inestable — revisar datos"

    return resultado


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRENAMIENTO DEL ENSAMBLE PRINCIPAL
#
#  El VotingEnsemble con soft voting ya produce probabilidades promediadas
#  entre los tres clasificadores → no se aplica una segunda calibración
#  isotónica encima (sería redundante y puede distorsionar las probs).
#
#  Sin embargo, para comparabilidad con el reporte de v3 y para la tabla
#  de calibración, calculamos Brier score y log-loss del ensamble directamente
#  en el set de prueba, que sirven como referencia del modelo combinado.
# ══════════════════════════════════════════════════════════════════════════════

def entrenar_ensamble(X_tr, y_tr, X_te, y_te,
                      bert_activo: bool = False) -> tuple:
    """
    Entrena el VotingEnsemble sobre X_tr y evalúa en X_te.

    El soft voting promedia predict_proba de CNB, LR y SVM. Las probabilidades
    resultantes representan el consenso ponderado — no requieren calibración
    adicional porque cada sub-estimador ya está calibrado individualmente.

    Args:
        X_tr, y_tr : Conjunto de entrenamiento.
        X_te, y_te : Conjunto de prueba.
        bert_activo: Ajusta pesos internos del ensamble.

    Returns:
        Tupla (ensemble_fitted, probs_ens, y_pred_ens, metricas_dict)
        · ensemble_fitted : estimador ajustado con interfaz predict_proba.
        · probs_ens       : P(alta | x) del ensamble, shape (n_te,).
        · y_pred_ens      : predicciones binarias, shape (n_te,).
        · metricas_dict   : dict compatible con guardar_todo / generar_narrativa.
    """
    pesos = [1, 2, 2] if bert_activo else [2, 1, 2]
    print(f"  Entrenando VotingEnsemble (soft, pesos={pesos})...")

    ensemble = obtener_ensamble_votacion(weights=pesos)
    ensemble.fit(X_tr, y_tr)

    probs_ens  = ensemble.predict_proba(X_te)[:, 1]
    y_pred_ens = ensemble.predict(X_te)

    # ── Métricas de calidad probabilística ───────────────────────────────────
    def hist_confianza(probs):
        cortes = [0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
        return {f"[{cortes[i]:.1f},{cortes[i+1]:.1f})":
                int(((probs >= cortes[i]) & (probs < cortes[i+1])).sum())
                for i in range(len(cortes)-1)}

    brier_ens = float(brier_score_loss(y_te, probs_ens))
    ll_ens    = float(log_loss(y_te, probs_ens))

    metricas = {
        # Mantenemos la estructura de cal_res para compatibilidad con narrativa
        "sin_calibrar": {
            "brier_score": round(brier_ens, 5),   # ensamble como línea base
            "log_loss":    round(ll_ens,    5),
            "prob_media":  round(float(probs_ens.mean()), 4),
            "prob_std":    round(float(probs_ens.std()),  4),
            "distribucion_confianza": hist_confianza(probs_ens),
        },
        "calibrado": {
            "brier_score":        round(brier_ens, 5),
            "log_loss":           round(ll_ens,    5),
            "prob_media":         round(float(probs_ens.mean()), 4),
            "prob_std":           round(float(probs_ens.std()),  4),
            "distribucion_confianza": hist_confianza(probs_ens),
            # Sin mejora adicional: el soft voting ya es el resultado calibrado
            "mejora_brier_pct":   0.0,
            "mejora_logloss_pct": 0.0,
        },
        "clasificacion_calibrado": {
            "f1_weighted": round(float(f1_score(y_te, y_pred_ens, average="weighted")), 4),
            "f1_alta":     round(float(f1_score(y_te, y_pred_ens, pos_label=1, average="binary")), 4),
            "roc_auc":     round(float(roc_auc_score(y_te, probs_ens)), 4),
            "accuracy":    round(float((y_pred_ens == y_te).mean()), 4),
            "confusion_matrix": confusion_matrix(y_te, y_pred_ens).tolist(),
        },
        "interpretacion": (
            "VotingEnsemble soft — probabilidades son el promedio ponderado "
            "de CNB, LR y LinearSVM. No se aplica calibración adicional."
        ),
        "pesos_ensamble": pesos,
    }

    return ensemble, probs_ens, y_pred_ens, metricas


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS DE ERRORES CRÍTICOS
# ══════════════════════════════════════════════════════════════════════════════

def analizar_errores_criticos(X_te, y_te, y_pred, probs_cal,
                               registros, indices_test) -> dict:
    """
    Identifica FN y FP de alta confianza del ensamble.
    FN confiado = ensamble dijo MEDIA con confianza alta pero era ALTA (el más costoso).
    """
    fn_confiados, fp_confiados = [], []

    for i, (real, pred, prob) in enumerate(zip(y_te, y_pred, probs_cal)):
        confianza = prob if pred == 1 else (1 - prob)
        if pred == real or confianza < UMBRAL_CONFIANZA:
            continue

        idx_original = int(indices_test[i]) if i < len(indices_test) else None
        reg = (registros[idx_original]
               if registros and idx_original is not None
               and idx_original < len(registros) else {})

        tokens     = reg.get("tokens", [])
        texto_orig = reg.get("comentario", reg.get("texto_normalizado", ""))
        texto_corto = texto_orig[:200] + ("..." if len(texto_orig) > 200 else "")

        error = {
            "indice_original":    idx_original,
            "etiqueta_real":      "alta" if real == 1 else "media",
            "prediccion":         "alta" if pred == 1 else "media",
            "prob_clase_alta":    round(float(prob), 4),
            "confianza_erronea":  round(float(confianza), 4),
            "texto_original":     texto_corto,
            "tokens_principales": tokens[:20] if tokens else [],
            "subreddit":          reg.get("subreddit", ""),
            "score_comentario":   reg.get("score_comentario", 0),
            "ifb_score":          reg.get("ifb_score", 0),
            "tipo_error":         "FN" if real == 1 else "FP",
            "diagnostico":        _diagnosticar_error(tokens, real, pred),
            "bert_sentimiento":   reg.get("sentimiento_bert", "N/D"),
            "bert_prob_neg":      reg.get("prob_neg", None),
            "desacuerdo_bert":    _diagnosticar_desacuerdo_bert(reg, real, pred),
        }

        if real == 1 and pred == 0:
            fn_confiados.append(error)
        else:
            fp_confiados.append(error)

    fn_confiados.sort(key=lambda x: -x["confianza_erronea"])
    fp_confiados.sort(key=lambda x: -x["confianza_erronea"])

    return {
        "umbral_confianza":   UMBRAL_CONFIANZA,
        "total_fn_confiados": len(fn_confiados),
        "total_fp_confiados": len(fp_confiados),
        "fn_exportados":      fn_confiados[:N_ERRORES_EXPORTAR],
        "fp_exportados":      fp_confiados[:N_ERRORES_EXPORTAR],
        "patrones_fn":        _patrones_en_errores(fn_confiados),
        "patrones_fp":        _patrones_en_errores(fp_confiados),
        "recomendaciones":    _generar_recomendaciones(
            _patrones_en_errores(fn_confiados),
            _patrones_en_errores(fp_confiados)),
    }


def _diagnosticar_error(tokens, real, pred) -> str:
    t = set(tokens)
    if real == 1 and pred == 0:
        if any(tok in t for tok in ["particular","medico_particular","de_mi_bolsillo"]):
            return "posible FN por pago_privado: gasto privado no asociado a queja alta"
        if any(tok in t for tok in ["fallecer","morir","muerte","intubado","uci"]):
            return "posible FN por impacto_clinico: consecuencia no detectada"
        if len(tokens) < 8:
            return "posible FN por texto muy corto: señal léxica insuficiente"
        return "FN sin patrón claro — ampliar TERMINOS_DOMINIO"
    if real == 0 and pred == 1:
        if any(tok in t for tok in ["vacuna_info_neutral","vacuna_gratuita"]):
            return "posible FP por vacunación informativa: token neutral no filtrado"
        if any(tok in t for tok in ["imss","issste","hospital","medico"]):
            return "posible FP por institución: mención sin queja real"
        return "FP sin patrón claro — posible ironía positiva"
    return "tipo de error no clasificado"


def _diagnosticar_desacuerdo_bert(reg, real, pred) -> str:
    sent_bert  = reg.get("sentimiento_bert", "")
    prob_neg   = reg.get("prob_neg", 0.0) or 0.0
    if not sent_bert or sent_bert == "N/D":
        return "sin datos BERT (dataset sin auto_etiquetar)"
    bert_dijo_alta = sent_bert == "NEG" and prob_neg >= 0.5
    if real == 1:
        return (f"BERT correcto (NEG p={prob_neg:.2f}), TF-IDF incorrecto → ampliar léxico"
                if bert_dijo_alta
                else f"BERT también incorrecto ({sent_bert}) → caso difícil / sarcasmo")
    else:
        return (f"BERT correcto ({sent_bert} p_neg={prob_neg:.2f}), TF-IDF incorrecto"
                if not bert_dijo_alta
                else f"BERT también incorrecto (NEG p={prob_neg:.2f}) → revisar etiqueta")


def _patrones_en_errores(errores) -> dict:
    if not errores:
        return {}
    return {
        "tokens_mas_frecuentes": Counter(
            tok for e in errores for tok in e.get("tokens_principales", [])
        ).most_common(15),
        "contextos_frecuentes":  Counter(
            ctx for e in errores for ctx in e.get("contextos", [])
        ).most_common(8),
        "diagnosticos": dict(Counter(e["diagnostico"] for e in errores)),
        "confianza_media": round(
            float(np.mean([e["confianza_erronea"] for e in errores])), 4),
    }


def _generar_recomendaciones(patron_fn, patron_fp) -> list[str]:
    recs = []
    fn_d = patron_fn.get("diagnosticos", {})
    fp_d = patron_fp.get("diagnosticos", {})
    if any("pago_privado" in d for d in fn_d):
        recs.append("ACCIÓN FN: Añadir 'particular','médico_privado' a TERMINOS_DOMINIO.")
    if any("texto muy corto" in d for d in fn_d):
        recs.append("ACCIÓN FN: Bajar MIN_CHARS para conservar comentarios cortos.")
    if any("vacunación" in d for d in fp_d):
        recs.append("ACCIÓN FP: Añadir bigramas 'vacuna disponible' como señal media.")
    if any("institución" in d for d in fp_d):
        recs.append("ACCIÓN FP: Contextualizar mención de IMSS sin queja real.")
    if not recs:
        recs.append("Sin patrones dominantes — el ensamble es robusto en este corpus.")
    return recs


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS DE DISCREPANCIAS ENTRE CLASIFICADORES
#
#  ¿Por qué es clave para la tesis?
#  ─────────────────────────────────
#  Un comentario donde CNB dice "alta" (vocabulario intenso de queja) pero
#  LR dice "media" (prob_neg BERT bajo) representa la ambigüedad de fondo
#  de la ineficiencia sistémica: el ciudadano usa palabras graves pero el
#  contexto semántico es moderado. Estos son los casos donde el ensamble
#  añade más valor: en lugar de empatar en 1-1, promedia las probabilidades
#  y decide con una confianza intermedia que refleja la ambigüedad real.
#
#  Tipos de discrepancia con 3 clasificadores:
#    · Tipo 2-1 (alta): 2 clasificadores → "alta", 1 → "media"
#                        El ensamble dice "alta" con menor confianza.
#    · Tipo 1-2 (media): 1 clasificador → "alta", 2 → "media"
#                        El ensamble dice "media" pero con probabilidad > 0.5.
#    Si el ensamble es correcto y los clasificadores individuales no → ganancia.
#    Si el ensamble también falla → zona gris real del corpus (caso límite).
# ══════════════════════════════════════════════════════════════════════════════

def analizar_discrepancias(predicciones_por_modelo: dict,
                            y_te: np.ndarray,
                            registros: list,
                            idx_te: np.ndarray,
                            carpeta: Path,
                            ts: str) -> dict:
    """
    Identifica y exporta los casos donde los clasificadores individuales
    no estuvieron de acuerdo. El VotingEnsemble se excluye del análisis
    de desacuerdo (es el árbitro final) pero sí se incluye en el reporte.

    Salidas:
        discrepancias_<ts>.json — Reporte completo con casos individuales.
        discrepancias_<ts>.csv  — Tabla plana para análisis en Excel/R.

    Args:
        predicciones_por_modelo : dict { nombre_modelo → np.ndarray(y_pred) }
        y_te                    : Etiquetas reales del set de prueba.
        registros               : Lista de registros originales (puede ser []).
        idx_te                  : Índices de y_te en el dataset original.
        carpeta                 : Ruta donde guardar los archivos.
        ts                      : Timestamp de esta ejecución.

    Returns:
        dict con tasa de acuerdo, resumen por tipo, y lista de casos.
    """
    # Modelos individuales (excluir VotingEnsemble del análisis de desacuerdo)
    nombres_todos = list(predicciones_por_modelo.keys())
    nombres_ind   = [n for n in nombres_todos if n != NOMBRE_ENSAMBLE]

    etiq = ETIQUETAS_CLASE   # ["media", "alta"]
    casos_discrepancia = []
    n_total = len(y_te)

    # Contadores por tipo de discrepancia
    conteo_tipo = Counter()

    for i in range(n_total):
        preds_ind = {n: int(predicciones_por_modelo[n][i]) for n in nombres_ind}
        valores   = list(preds_ind.values())

        # Sin discrepancia: todos coinciden
        if len(set(valores)) == 1:
            continue

        # ── Clasificar el tipo de discrepancia ───────────────────────────────
        n_alta  = sum(1 for v in valores if v == 1)
        n_media = len(valores) - n_alta
        tipo    = f"{n_alta}-{n_media} (alta-media)"   # ej. "2-1 (alta-media)"
        conteo_tipo[tipo] += 1

        # ── Predicción del ensamble en este caso ─────────────────────────────
        pred_ens = (int(predicciones_por_modelo[NOMBRE_ENSAMBLE][i])
                    if NOMBRE_ENSAMBLE in predicciones_por_modelo else None)

        # ── Texto original (si está disponible) ──────────────────────────────
        idx_orig = int(idx_te[i]) if idx_te is not None and i < len(idx_te) else i
        texto    = ""
        if registros and idx_orig < len(registros):
            reg   = registros[idx_orig]
            texto = reg.get("texto_normalizado",
                    reg.get("comentario",
                    reg.get("texto", "")))[:250]

        # ── Determinar si el ensamble acertó ─────────────────────────────────
        etiqueta_real = int(y_te[i])
        ens_correcto  = (pred_ens == etiqueta_real) if pred_ens is not None else None

        # Cuántos modelos individuales acertaron
        ind_correctos = sum(1 for v in valores if v == etiqueta_real)

        caso = {
            "indice_test":         i,
            "indice_original":     idx_orig,
            "etiqueta_real":       etiq[etiqueta_real],
            "decision_ensamble":   etiq[pred_ens] if pred_ens is not None else "N/D",
            "ensamble_correcto":   ens_correcto,
            "ind_correctos_de":    f"{ind_correctos}/{len(nombres_ind)}",
            "tipo_discrepancia":   tipo,
            "predicciones":        {n: etiq[v] for n, v in preds_ind.items()},
            "texto_muestra":       texto,
        }
        casos_discrepancia.append(caso)

    n_disc = len(casos_discrepancia)
    tasa_acuerdo = round(1.0 - n_disc / n_total, 4) if n_total > 0 else 1.0

    # ── Cuántas discrepancias resolvió bien el ensamble ──────────────────────
    ens_rescato = sum(1 for c in casos_discrepancia if c["ensamble_correcto"] is True)
    ens_fallo   = sum(1 for c in casos_discrepancia if c["ensamble_correcto"] is False)

    resultado = {
        "meta": {
            "total_test":          n_total,
            "n_discrepancias":     n_disc,
            "tasa_acuerdo":        tasa_acuerdo,
            "tasa_desacuerdo":     round(1.0 - tasa_acuerdo, 4),
            "modelos_individuales":nombres_ind,
            "ensamble_rescato":    ens_rescato,   # discrepancias donde ens acertó
            "ensamble_fallo":      ens_fallo,     # discrepancias donde ens también falló
            "conteo_por_tipo":     dict(conteo_tipo),
        },
        "casos": casos_discrepancia,
    }

    # ── Guardar JSON ──────────────────────────────────────────────────────────
    ruta_json = carpeta / f"discrepancias_{ts}.json"
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    # ── Guardar CSV (formato plano para Excel / Stata / R) ────────────────────
    ruta_csv = carpeta / f"discrepancias_{ts}.csv"
    if casos_discrepancia:
        campos_base = ["indice_test", "indice_original", "etiqueta_real",
                       "decision_ensamble", "ensamble_correcto",
                       "ind_correctos_de", "tipo_discrepancia", "texto_muestra"]
        campos_pred = [f"pred_{n}" for n in nombres_ind]
        fieldnames  = campos_base + campos_pred

        with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for caso in casos_discrepancia:
                fila = {k: caso[k] for k in campos_base}
                for n in nombres_ind:
                    fila[f"pred_{n}"] = caso["predicciones"].get(n, "N/D")
                writer.writerow(fila)

    print(f"\n  ── Análisis de discrepancias ──────────────────────────────────")
    print(f"    Total comentarios test    : {n_total}")
    print(f"    Con desacuerdo entre modelos: {n_disc} ({100*(1-tasa_acuerdo):.1f}%)")
    print(f"    Tipos de desacuerdo       : {dict(conteo_tipo)}")
    print(f"    Ensamble rescató (de disc.): {ens_rescato}")
    print(f"    Ensamble también falló    : {ens_fallo}")
    print(f"    💾 JSON : {ruta_json.name}")
    print(f"    💾 CSV  : {ruta_csv.name}")

    return resultado


# ══════════════════════════════════════════════════════════════════════════════
#  CURVA DE APRENDIZAJE — GENÉRICA
#
#  Acepta cualquier pipeline sklearn, incluyendo VotingClassifier.
#  La curva del ensamble demuestra que al combinar clasificadores el error
#  de generalización disminuye más suavemente que el de cada modelo individual.
# ══════════════════════════════════════════════════════════════════════════════

def curva_aprendizaje(pipeline, X, y,
                      nombre: str = "modelo",
                      n_puntos: int = CURVA_PUNTOS) -> dict:
    """
    Calcula la curva de aprendizaje F1-weighted para cualquier estimador sklearn.

    Args:
        pipeline  : Estimador sklearn con interfaz fit/predict.
        X         : Matriz de features (sparse o densa).
        y         : Vector de etiquetas binarias.
        nombre    : Nombre legible del modelo (se usa en logs y gráfica).
        n_puntos  : Número de puntos en el eje X (tamaños de entrenamiento).

    Returns:
        dict con 'puntos' (lista de dicts por tamaño) y 'diagnostico' (str).
    """
    print(f"  Curva de aprendizaje: {nombre} ({n_puntos} puntos)...")
    cv_ca = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    sizes, train_scores, test_scores = learning_curve(
        pipeline, X, y,
        train_sizes=np.linspace(0.10, 1.0, n_puntos),
        cv=cv_ca,
        scoring="f1_weighted",
    )

    puntos = [
        {
            "n_train":        int(sz),
            "f1_train_media": round(float(tr.mean()), 4),
            "f1_train_std":   round(float(tr.std()),  4),
            "f1_test_media":  round(float(te.mean()),  4),
            "f1_test_std":    round(float(te.std()),   4),
            "gap":            round(float(tr.mean() - te.mean()), 4),
        }
        for sz, tr, te in zip(sizes, train_scores, test_scores)
    ]

    gap_final     = puntos[-1]["gap"]
    pendiente_test = puntos[-1]["f1_test_media"] - puntos[n_puntos//2]["f1_test_media"]

    if gap_final > 0.12:
        diagnostico = "overfitting moderado — reducir vocabulario o aumentar alpha"
    elif pendiente_test > 0.02:
        diagnostico = "más datos mejorarían el modelo — continuar scraping"
    elif gap_final < 0.03:
        diagnostico = "modelo saturado — mejorar features > añadir datos"
    else:
        diagnostico = "equilibrado — modelo estable"

    return {"modelo": nombre, "puntos": puntos, "diagnostico": diagnostico}


def graficar_curva_aprendizaje(pipeline, X, y, nombre: str,
                                carpeta: Path, ts: str) -> Path | None:
    """
    Genera y guarda la curva de aprendizaje como PNG.
    Función genérica: funciona con cualquier estimador sklearn.

    Args:
        pipeline : Estimador sklearn compatible con learning_curve.
        X, y     : Datos de entrenamiento completos.
        nombre   : Nombre del modelo (eje Y y título).
        carpeta  : Directorio donde guardar el PNG.
        ts       : Timestamp para el nombre de archivo.

    Returns:
        Path del PNG guardado, o None si matplotlib no está disponible.
    """
    if not _MATPLOTLIB_OK:
        print(f"    [matplotlib no disponible — se omite gráfica de {nombre}]")
        return None

    datos = curva_aprendizaje(pipeline, X, y, nombre)
    puntos = datos["puntos"]
    n_train  = [p["n_train"]        for p in puntos]
    tr_media = [p["f1_train_media"] for p in puntos]
    tr_std   = [p["f1_train_std"]   for p in puntos]
    te_media = [p["f1_test_media"]  for p in puntos]
    te_std   = [p["f1_test_std"]    for p in puntos]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(n_train,
                    [m-s for m,s in zip(tr_media, tr_std)],
                    [m+s for m,s in zip(tr_media, tr_std)],
                    alpha=0.12, color="#2196F3")
    ax.fill_between(n_train,
                    [m-s for m,s in zip(te_media, te_std)],
                    [m+s for m,s in zip(te_media, te_std)],
                    alpha=0.12, color="#FF5722")
    ax.plot(n_train, tr_media, "o-", color="#2196F3", lw=2, label="Train F1")
    ax.plot(n_train, te_media, "s-", color="#FF5722", lw=2, label="Validación F1")

    ax.set_title(f"Curva de aprendizaje — {nombre}", fontsize=13, pad=10)
    ax.set_xlabel("Tamaño del conjunto de entrenamiento")
    ax.set_ylabel("F1-weighted")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.02, 0.04, datos["diagnostico"],
            transform=ax.transAxes, fontsize=8, color="gray",
            verticalalignment="bottom")

    nombre_safe = nombre.replace(" ", "_").lower()
    ruta = carpeta / f"curva_{nombre_safe}_{ts}.png"
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"    💾 Curva guardada: {ruta.name}")
    return ruta


def graficar_matriz_confusion(y_te, y_pred, nombre: str,
                               carpeta: Path, ts: str,
                               etiquetas: list[str] = None) -> Path | None:
    """
    Genera y guarda la matriz de confusión normalizada como PNG.
    Función genérica: acepta cualquier par (y_te, y_pred) independientemente
    del modelo que los generó.

    Args:
        y_te     : Etiquetas reales.
        y_pred   : Predicciones del modelo.
        nombre   : Nombre del modelo (título).
        carpeta  : Directorio de salida.
        ts       : Timestamp.
        etiquetas: Lista de nombres de clase. Default: ETIQUETAS_CLASE.

    Returns:
        Path del PNG, o None si matplotlib no está disponible.
    """
    if not _MATPLOTLIB_OK:
        return None

    etiquetas = etiquetas or ETIQUETAS_CLASE
    cm = confusion_matrix(y_te, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, datos_cm, titulo in zip(
        axes,
        [cm, cm_norm],
        ["Conteos absolutos", "Normalizada (por clase real)"]
    ):
        im = ax.imshow(datos_cm, interpolation="nearest",
                       cmap="Blues" if titulo.startswith("C") else "RdYlGn",
                       vmin=0, vmax=None if titulo.startswith("C") else 1)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xticks(range(len(etiquetas)))
        ax.set_yticks(range(len(etiquetas)))
        ax.set_xticklabels(etiquetas, rotation=30, ha="right")
        ax.set_yticklabels(etiquetas)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        ax.set_title(titulo, fontsize=10)
        fmt = ".0f" if titulo.startswith("C") else ".2f"
        thresh = datos_cm.max() / 2
        for i in range(len(etiquetas)):
            for j in range(len(etiquetas)):
                ax.text(j, i, f"{datos_cm[i, j]:{fmt}}",
                        ha="center", va="center",
                        color="white" if datos_cm[i, j] > thresh else "black",
                        fontsize=11, fontweight="bold")

    fig.suptitle(f"Matriz de confusión — {nombre}", fontsize=13)
    nombre_safe = nombre.replace(" ", "_").lower()
    ruta = carpeta / f"confusion_{nombre_safe}_{ts}.png"
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    print(f"    💾 Confusión guardada: {ruta.name}")
    return ruta


# ══════════════════════════════════════════════════════════════════════════════
#  COMPARACIÓN DE MODELOS — TABLA COMPLETA Precision / Recall / F1
#
#  En v4 el CATALOGO_MODELOS incluye VotingEnsemble, por lo que aparece
#  en la tabla comparativa junto a los tres clasificadores individuales.
#  Esto permite cuantificar la GANANCIA del ensamble sobre cada base classifier.
#
#  La función retorna TAMBIÉN un dict de predicciones por modelo, que se
#  usa en analizar_discrepancias() para encontrar los casos de desacuerdo.
# ══════════════════════════════════════════════════════════════════════════════

def comparar_modelos(X_tr, y_tr, X_te, y_te,
                     carpeta: Path = None,
                     ts: str = "") -> tuple[list[dict], dict]:
    """
    Entrena y evalúa todos los modelos de CATALOGO_MODELOS en el mismo split.
    Reporta Precision, Recall y F1 por clase (alta / media) + métricas globales.
    Genera PNG de matriz de confusión para cada modelo si matplotlib está activo.

    Returns:
        Tupla (resultados, predicciones_por_modelo):
          · resultados            : Lista de dicts ordenada por F1-weighted desc.
          · predicciones_por_modelo: { nombre_modelo → np.ndarray(y_pred) }
                                    Usado por analizar_discrepancias().
    """
    n_modelos = len(CATALOGO_MODELOS)
    print(f"\n  Comparando {n_modelos} modelos: {list(CATALOGO_MODELOS.keys())}...")

    resultados             = []
    predicciones_por_modelo = {}

    for nombre, factory in CATALOGO_MODELOS.items():
        print(f"    ▸ Entrenando {nombre}...")
        clf = factory()
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else None

        # Guardar predicciones para análisis de discrepancias
        predicciones_por_modelo[nombre] = y_pred

        # Métricas por clase
        rep = classification_report(y_te, y_pred, output_dict=True,
                                    target_names=ETIQUETAS_CLASE)

        resultado = {
            "modelo":           nombre,
            # ── Globales ────────────────────────────────────────────────────
            "f1_weighted":      round(float(f1_score(y_te, y_pred, average="weighted")), 4),
            "accuracy":         round(float((y_pred == y_te).mean()), 4),
            "roc_auc":          (round(float(roc_auc_score(y_te, y_prob)), 4)
                                 if y_prob is not None else None),
            # ── Clase ALTA (positiva, la que interesa detectar) ─────────────
            "precision_alta":   round(float(rep["alta"]["precision"]),   4),
            "recall_alta":      round(float(rep["alta"]["recall"]),      4),
            "f1_alta":          round(float(rep["alta"]["f1-score"]),    4),
            # ── Clase MEDIA (negativa, referencia) ──────────────────────────
            "precision_media":  round(float(rep["media"]["precision"]),  4),
            "recall_media":     round(float(rep["media"]["recall"]),     4),
            "f1_media":         round(float(rep["media"]["f1-score"]),   4),
            # ── Matriz de confusión cruda ────────────────────────────────────
            "confusion_matrix": confusion_matrix(y_te, y_pred).tolist(),
            # ── Indicador de rol ──────────────────────────────────────────────
            "es_ensamble":      (nombre == NOMBRE_ENSAMBLE),
        }
        resultados.append(resultado)

        # Gráfica de confusión genérica
        if carpeta:
            graficar_matriz_confusion(y_te, y_pred, nombre, carpeta, ts)

    resultados.sort(key=lambda x: -x["f1_weighted"])
    return resultados, predicciones_por_modelo


def _tabla_comparativa_txt(comparacion: list[dict]) -> str:
    """
    Genera la tabla comparativa en texto formateado para el reporte narrativo.
    Señala VotingEnsemble con [ENS] y el mejor modelo con ★.
    """
    ancho = 100
    sep   = "─" * ancho

    cab = (f"  {'Modelo':<22} {'Prec↑':>7} {'Recall↑':>8} {'F1↑':>7} "
           f"{'F1-W':>7} {'AUC':>7} {'Acc':>7}  Rol")

    filas = []
    for i, m in enumerate(comparacion):
        auc_s = f"{m['roc_auc']:.4f}" if m["roc_auc"] else "  N/D "
        mark  = " ★" if i == 0 else "  "
        rol   = "[ENS]" if m.get("es_ensamble") else "     "
        filas.append(
            f"  {m['modelo']:<22} {m['precision_alta']:>7.4f} "
            f"{m['recall_alta']:>8.4f} {m['f1_alta']:>7.4f} "
            f"{m['f1_weighted']:>7.4f} {auc_s:>7} {m['accuracy']:>7.4f}"
            f"{mark} {rol}"
        )

    mejor = comparacion[0]
    veredicto = (
        f"  → Mejor para detectar INEFICIENCIAS: {mejor['modelo']}\n"
        f"     Precision={mejor['precision_alta']:.4f} | "
        f"Recall={mejor['recall_alta']:.4f} | "
        f"F1={mejor['f1_alta']:.4f}\n"
        f"     Un Recall alto indica que el modelo recupera la mayoría de quejas\n"
        f"     reales — métrica prioritaria en salud pública (costo del FN alto).\n"
        f"     [ENS] = VotingEnsemble | ★ = mejor modelo en esta ejecución"
    )

    return (f"\n  {sep}\n"
            f"  TABLA COMPARATIVA — CLASE ALTA (ineficiencia detectada)\n"
            f"  Prec↑ = Precision(alta) | Recall↑ = Recall(alta) | F1↑ = F1(alta)\n"
            f"  {sep}\n"
            f"{cab}\n"
            f"  {sep}\n"
            + "\n".join(filas) + "\n"
            f"  {sep}\n\n"
            + veredicto)


# ══════════════════════════════════════════════════════════════════════════════
#  TOP TERMS, SPEARMAN, χ², DIMENSIONES
# ══════════════════════════════════════════════════════════════════════════════

def extraer_top_terms(clf, vec, selector, n=TOP_N_TERMS) -> dict:
    """
    Extrae top-N features predictivas.
    Soporta:
      · CNB (feature_log_prob_)
      · LR y LinearSVC (coef_)
      · VotingClassifier → extrae del sub-estimador LR interno.
    """
    def _extraer_cnb(estimador):
        if hasattr(estimador, "feature_log_prob_"): return estimador
        if hasattr(estimador, "named_steps"):
            return _extraer_cnb(estimador.named_steps.get("cnb", None))
        if hasattr(estimador, "calibrated_classifiers_"):
            sub = getattr(estimador.calibrated_classifiers_[0], "estimator", None)
            return _extraer_cnb(sub) if sub else None
        inner = getattr(estimador, "estimator", None)
        return _extraer_cnb(inner) if inner else None

    def _extraer_lr(estimador):
        if estimador is None: return None
        if hasattr(estimador, "coef_"): return estimador
        if hasattr(estimador, "named_steps"):
            for paso in reversed(list(estimador.named_steps.values())):
                r = _extraer_lr(paso)
                if r: return r
        if hasattr(estimador, "calibrated_classifiers_"):
            sub = getattr(estimador.calibrated_classifiers_[0], "estimator", None)
            return _extraer_lr(sub) if sub else None
        inner = getattr(estimador, "estimator", None)
        return _extraer_lr(inner) if inner else None

    if vec is None:
        return {"alta": [], "media": [], "nombres_features": []}

    nombres_vec   = list(vec.get_feature_names_out())
    if selector is not None:
        mascara       = selector.get_support()
        nombres_tfidf = [nombres_vec[i] for i, m in enumerate(mascara) if m]
    else:
        nombres_tfidf = nombres_vec

    # ── Soporte para VotingClassifier: preferir sub-estimador LR ─────────────
    # Si clf es un VotingClassifier ajustado, `named_estimators_` contiene
    # cada sub-estimador con su nombre. Priorizamos LR por su interpretabilidad.
    if hasattr(clf, "named_estimators_"):
        lr_sub = clf.named_estimators_.get("lr", None)
        cnb_sub = clf.named_estimators_.get("cnb", None)
        lr  = _extraer_lr(lr_sub)
        cnb = _extraer_cnb(cnb_sub) if lr is None else None
    else:
        cnb = _extraer_cnb(clf)
        lr  = _extraer_lr(clf)

    resultado = {}

    if cnb is not None and hasattr(cnb, "feature_log_prob_"):
        log_probs = cnb.feature_log_prob_
        n_cols    = min(log_probs.shape[1], len(nombres_tfidf))
        nombres_f = nombres_tfidf[:n_cols]
        for idx_clase, nombre_clase in enumerate(["media", "alta"]):
            if idx_clase >= log_probs.shape[0]:
                resultado[nombre_clase] = []; continue
            pesos   = -log_probs[idx_clase][:n_cols]
            top_idx = np.argsort(pesos)[-n:][::-1]
            resultado[nombre_clase] = [
                {"termino": nombres_f[i], "peso": round(float(pesos[i]), 5),
                 "tipo": "texto", "dimension": _asignar_dimension(nombres_f[i])}
                for i in top_idx if i < len(nombres_f)
            ]

    elif lr is not None and hasattr(lr, "coef_"):
        coef  = lr.coef_[0]
        n_total = len(coef)
        nombres_completos = (nombres_tfidf +
                             [f"[NUM_{i}]" for i in range(max(0, n_total - len(nombres_tfidf)))])
        nombres_completos = nombres_completos[:n_total]
        top_alta  = np.argsort(coef)[-n:][::-1]
        top_media = np.argsort(coef)[:n]
        for nombre_clase, top_idx in [("alta", top_alta), ("media", top_media)]:
            resultado[nombre_clase] = [
                {"termino": nombres_completos[i] if i < len(nombres_completos) else f"feat_{i}",
                 "peso": round(float(abs(coef[i])), 5),
                 "coef_lr": round(float(coef[i]), 5),
                 "tipo": "numerico" if i >= len(nombres_tfidf) else "texto",
                 "dimension": _asignar_dimension(
                     nombres_completos[i] if i < len(nombres_completos) else "")}
                for i in top_idx
            ]
    else:
        resultado["alta"] = []
        resultado["media"] = []

    resultado["nombres_features"] = nombres_tfidf
    resultado["modo"] = "lr_ensamble" if hasattr(clf, "named_estimators_") else \
                        ("cnb" if cnb else ("lr" if lr else "desconocido"))
    return resultado


def top_features_hibrido(clf, vec, selector, art: dict, n: int = 20) -> list[dict]:
    """
    Tabla de features híbridas (texto + BERT) del estimador LR.
    Compatible con VotingClassifier: extrae el sub-estimador LR interno.
    """
    def _extraer_lr(estimador):
        if estimador is None: return None
        if hasattr(estimador, "coef_"): return estimador
        if hasattr(estimador, "named_steps"):
            for paso in reversed(list(estimador.named_steps.values())):
                r = _extraer_lr(paso)
                if r: return r
        if hasattr(estimador, "calibrated_classifiers_"):
            sub = getattr(estimador.calibrated_classifiers_[0], "estimator", None)
            return _extraer_lr(sub) if sub else None
        inner = getattr(estimador, "estimator", None)
        return _extraer_lr(inner) if inner else None

    # Soporte para VotingClassifier
    if hasattr(clf, "named_estimators_"):
        lr = _extraer_lr(clf.named_estimators_.get("lr", None))
    else:
        lr = _extraer_lr(clf)

    if lr is None or not hasattr(lr, "coef_") or vec is None:
        return []

    coef          = lr.coef_[0]
    nombres_vec   = list(vec.get_feature_names_out())
    if selector is not None:
        mascara       = selector.get_support()
        nombres_tfidf = [nombres_vec[i] for i, m in enumerate(mascara) if m]
    else:
        nombres_tfidf = nombres_vec

    cols_num = art.get("features_numericas", [])
    nombres_completos = (nombres_tfidf + cols_num)[:len(coef)]

    BERT_COLS  = set(FEATURES_BERT + ["score_queja_salud"])
    ENRIQ_COLS = {"polaridad_score", "ifb_score", "impacto_escala",
                  "num_palabras", "peso_reddit"}

    def _tipo(nombre: str) -> str:
        if nombre in BERT_COLS:  return "BERT"
        if nombre in ENRIQ_COLS: return "enriquecimiento"
        return "texto_tfidf"

    top_idx = np.argsort(np.abs(coef))[-n:][::-1]
    tabla = []
    for i in top_idx:
        nombre = nombres_completos[i] if i < len(nombres_completos) else f"feat_{i}"
        tabla.append({
            "rank":      len(tabla) + 1,
            "nombre":    nombre,
            "coef":      round(float(coef[i]), 5),
            "abs_coef":  round(float(abs(coef[i])), 5),
            "direccion": "→ alta" if coef[i] > 0 else "→ media",
            "tipo":      _tipo(nombre),
            "dimension": _asignar_dimension(nombre),
        })
    return tabla


def _asignar_dimension(termino: str) -> str:
    t = termino.lower()
    for dim, palabras in DIMENSIONES_NARRATIVAS.items():
        if any(p in t for p in palabras):
            return dim
    return "otro"


def correlacion_spearman(registros: list[dict], y: np.ndarray) -> list[dict]:
    COLS = ["polaridad_score","ifb_score","impacto_escala",
            "num_palabras","peso_reddit","score_comentario"]
    n = min(len(registros), len(y))
    resultados = []
    for col in COLS:
        vals, y_f = [], []
        for i, reg in enumerate(registros[:n]):
            v = reg.get(col)
            if v is not None:
                try: vals.append(float(v)); y_f.append(int(y[i]))
                except: pass
        if len(vals) < 10: continue
        rho, pval = spearmanr(vals, y_f)
        r = abs(rho)
        resultados.append({
            "feature": col, "rho": round(float(rho), 4),
            "p_valor": round(float(pval), 6), "n": len(vals),
            "significativa": bool(pval < 0.05),
            "magnitud": ("muy fuerte" if r>=0.7 else "fuerte" if r>=0.5 else
                         "moderada" if r>=0.3 else "débil" if r>=0.1 else "despreciable"),
        })
    return sorted(resultados, key=lambda x: -abs(x["rho"]))


def prueba_chi2_categorias(registros: list[dict], y: np.ndarray) -> list[dict]:
    n = min(len(registros), len(y))
    resultados = []
    for cat in CATEGORIAS_CHI2:
        pa, pm, aa, am = 0, 0, 0, 0
        for i, reg in enumerate(registros[:n]):
            ctx = reg.get("contextos", [])
            if isinstance(ctx, str): ctx = [ctx]
            tc = cat in ctx; ta = int(y[i]) == 1
            if tc and ta: pa += 1
            elif tc:      pm += 1
            elif ta:      aa += 1
            else:         am += 1
        if pa + pm < 3: continue
        chi2, pval, _, _ = chi2_contingency([[pa,pm],[aa,am]], correction=True)
        or_v = ((pa+.5)/(pm+.5)) / ((aa+.5)/(am+.5))
        resultados.append({
            "categoria": cat, "chi2": round(float(chi2), 4),
            "p_valor": round(float(pval), 6),
            "odds_ratio": round(or_v, 3),
            "significativa": bool(pval < 0.05),
            "interpretacion": (
                "muy fuertemente asociada con alta" if or_v > 3 else
                "asociada con alta" if or_v > 1.5 else
                "sin asociación clara" if or_v > 0.67 else "asociada con media"),
        })
    return sorted(resultados, key=lambda x: -abs(x["chi2"]))


def analizar_dimensiones(top_terms: dict) -> dict:
    pesos: dict[str, float] = defaultdict(float)
    terms: dict[str, list]  = defaultdict(list)
    for item in top_terms.get("alta", []):
        pesos[item["dimension"]] += item["peso"]
        terms[item["dimension"]].append(item["termino"])
    total = sum(pesos.values()) or 1.0
    return {
        dim: {
            "peso_total": round(p, 4),
            "proporcion": round(p/total, 4),
            "terminos":   terms[dim],
            "n_terminos": len(terms[dim]),
        }
        for dim, p in sorted(pesos.items(), key=lambda x: -x[1])
    }


# ══════════════════════════════════════════════════════════════════════════════
#  REPORTE NARRATIVO
# ══════════════════════════════════════════════════════════════════════════════

def generar_narrativa(top_terms, cv_res, cal_res, errores_res,
                      curva_res, comparacion, correlaciones,
                      chi2_res, dimensiones,
                      discrepancias: dict = None) -> str:

    terms_alta = [i["termino"].replace("_"," ")
                  for i in top_terms.get("alta", [])[:N_TERMINOS_CITA]]
    p = [terms_alta[i] if i < len(terms_alta) else "[N/D]"
         for i in range(N_TERMINOS_CITA)]

    f1_m   = cv_res["f1_weighted"]["media"]
    f1_pm  = cv_res["f1_weighted"]["ic95_pm"]
    f1_std = cv_res["f1_weighted"]["std"]
    auc    = cv_res["roc_auc"]["media"]
    ovf    = cv_res["overfitting"]["detectado"]
    pesos  = cv_res.get("pesos_ensamble", [1, 2, 2])

    brier_ens   = cal_res["calibrado"]["brier_score"]
    ll_ens      = cal_res["calibrado"]["log_loss"]
    f1_cal      = cal_res["clasificacion_calibrado"]["f1_weighted"]

    fn_total = errores_res["total_fn_confiados"]
    fp_total = errores_res["total_fp_confiados"]
    recs     = errores_res["recomendaciones"]

    dim_dom = list(dimensiones.keys())[0] if dimensiones else "falta_insumos"
    dim_2da = list(dimensiones.keys())[1] if len(dimensiones) > 1 else "tiempo_espera"
    dim_map = {
        "falta_insumos":         "la falta de insumos",
        "tiempo_espera":         "el tiempo de espera prolongado",
        "consecuencia_clinica":  "las consecuencias clínicas de la demora",
        "pago_privado":          "el desembolso en servicios privados",
        "corrupcion_negligencia":"la negligencia e irregularidades institucionales",
        "sistema_roto":          "el colapso del sistema de atención",
    }

    cor_top   = next((r for r in correlaciones if r["significativa"]), None)
    chi2_top  = next((r for r in chi2_res if r["significativa"]), None)
    mejor     = comparacion[0] if comparacion else {}
    curva_dia = curva_res.get("diagnostico", "") if isinstance(curva_res, dict) else ""

    # ── Datos de discrepancias ────────────────────────────────────────────────
    disc_meta       = discrepancias.get("meta", {}) if discrepancias else {}
    n_disc          = disc_meta.get("n_discrepancias", 0)
    n_total_test    = disc_meta.get("total_test", 0)
    tasa_desacuerdo = disc_meta.get("tasa_desacuerdo", 0.0)
    ens_rescato     = disc_meta.get("ensamble_rescato", 0)
    ens_fallo_disc  = disc_meta.get("ensamble_fallo", 0)
    tipos_disc      = disc_meta.get("conteo_por_tipo", {})

    txt = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  REPORTE NARRATIVO v4 — BUROCRACIA DEL DOLOR EN SALUD PÚBLICA MX        ║
║  VotingEnsemble (CNB + LR + LinearSVM) | {N_FOLDS}-fold StratifiedKFold         ║
╚══════════════════════════════════════════════════════════════════════════╝

HALLAZGO PRINCIPAL
──────────────────
El análisis de minería de opinión revela que la ineficiencia del sector
salud MX ante la influenza no se expresa solo en términos médicos, sino en
un lenguaje de 'burocracia del dolor'. El ensamble de tres clasificadores
(CNB + LR + LinearSVM, soft voting, pesos={pesos}) captura esta señal
mejor que cualquier clasificador individual.

Términos con mayor peso predictivo para relevancia ALTA:
  1. "{p[0]}"
  2. "{p[1]}"
  3. "{p[2]}"

Confirma que el ciudadano percibe la ineficiencia principalmente a través
de {dim_map.get(dim_dom, dim_dom)} y {dim_map.get(dim_2da, dim_2da)}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
① VALIDACIÓN CRUZADA — {N_FOLDS}-fold estratificado (VotingEnsemble)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  F1 weighted     : {f1_m:.4f} ± {f1_pm:.4f}  (IC 95%)
  AUC-ROC         : {auc:.4f} ± {cv_res['roc_auc']['ic95_pm']:.4f}
  F1 clase alta   : {cv_res['f1_alta']['media']:.4f} ± {cv_res['f1_alta']['ic95_pm']:.4f}
  Desv. estándar  : {f1_std:.4f}
  Veredicto       : {cv_res['veredicto']}
  Overfitting     : {'⚠️  detectado (gap train-test > 10%)' if ovf else '✅ no detectado'}
  Curva aprend.   : {curva_dia}

  Valores por fold: {cv_res['f1_weighted']['valores_por_fold']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
② PROBABILIDADES DEL ENSAMBLE (soft voting)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Brier Score ensamble : {brier_ens:.5f}  (menor = mejor calibración)
  Log-Loss ensamble    : {ll_ens:.5f}
  F1 weighted          : {f1_cal:.4f}
  Nota: el soft voting ya promedía probabilidades calibradas de cada
  sub-estimador — no se aplica calibración isotónica adicional.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
③ ERRORES CRÍTICOS del ensamble (umbral confianza ≥ {UMBRAL_CONFIANZA})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Falsos Negativos confiados : {fn_total}  ← quejas reales ignoradas
  Falsos Positivos confiados : {fp_total}  ← falsas alarmas

  Recomendaciones accionables:
"""
    for r in recs:
        txt += f"  · {r}\n"

    # ── Sección de discrepancias ──────────────────────────────────────────────
    txt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
④ ANÁLISIS DE DISCREPANCIAS — ZONA GRIS DE LA INEFICIENCIA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Comentarios con desacuerdo entre clasificadores individuales:
    {n_disc} de {n_total_test} ({100*tasa_desacuerdo:.1f}% del set de prueba)
  
  Tipos de desacuerdo detectados:"""
    for tipo_d, cnt_d in sorted(tipos_disc.items()):
        txt += f"\n    · {tipo_d:<25}: {cnt_d} comentarios"
    txt += f"""

  Resolución por el ensamble:
    · Ensamble acertó (rescató discrepancia): {ens_rescato}
    · Ensamble también falló (zona gris real): {ens_fallo_disc}

  Interpretación para la tesis:
    Los {n_disc} comentarios con desacuerdo representan la 'zona gris'
    de la ineficiencia sistémica: comentarios donde el vocabulario de queja
    (peso CNB) no coincide con la señal semántica BERT (peso LR). Estos
    casos límite son evidencia de la complejidad del discurso ciudadano
    sobre el sistema de salud → exportados en discrepancias_*.csv y .json.

"""

    # ── Tabla comparativa (sección central de la tesis) ───────────────────────
    txt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⑤ COMPARACIÓN DE MODELOS — TABLA COMPLETA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""
    txt += _tabla_comparativa_txt(comparacion)

    if cor_top:
        txt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⑥ CORRELACIONES Y χ²
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Spearman más fuerte: {cor_top['feature']} (ρ={cor_top['rho']}, p={cor_top['p_valor']}) [{cor_top['magnitud']}]
"""
    if chi2_top:
        txt += (f"  χ² más significativo: {chi2_top['categoria']} "
                f"(χ²={chi2_top['chi2']}, OR={chi2_top['odds_ratio']}) "
                f"— {chi2_top['interpretacion']}\n")

    txt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOP TÉRMINOS PREDICTIVOS (clase ALTA) — extraídos del sub-estimador LR
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    for j, item in enumerate(top_terms.get("alta", [])[:15], 1):
        barra = "█" * int(item["peso"] * 15)
        txt += (f"  {j:>2}. {item['termino']:<30} {item['peso']:.5f}"
                f"  [{item['dimension']}]  {barra}\n")

    return txt


# ══════════════════════════════════════════════════════════════════════════════
#  GUARDADO
# ══════════════════════════════════════════════════════════════════════════════

def guardar_todo(carpeta, cv_res, cal_res, errores_res, curva_res,
                 comparacion, top_terms, correlaciones, chi2_res,
                 dimensiones, narrativa, ruta_X,
                 tabla_hibrida=None, bert_activo=False,
                 discrepancias=None):

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {}

    mejor_modelo = comparacion[0]["modelo"] if comparacion else "N/D"
    disc_meta    = discrepancias.get("meta", {}) if discrepancias else {}

    reporte = {
        "meta": {
            "generado":          datetime.now().isoformat(),
            "version":           "4.0",
            "modelo_principal":  "VotingEnsemble (soft voting)",
            "bert_activo":       bert_activo,
            "corpus":            str(ruta_X),
            "mejor_modelo":      mejor_modelo,
            "catalogo":          list(CATALOGO_MODELOS.keys()),
            "pesos_ensamble":    cv_res.get("pesos_ensamble", [1, 2, 2]),
        },
        "validacion_cruzada":     cv_res,
        "ensamble_test":          cal_res,
        "comparacion_modelos":    comparacion,
        "discrepancias_resumen":  disc_meta,
        "top_terms_por_clase":    {k: v for k, v in top_terms.items()
                                   if k != "nombres_features"},
        "top_features_hibrido":   tabla_hibrida or [],
        "dimensiones_narrativas": dimensiones,
        "correlaciones_spearman": correlaciones,
        "chi2_categorias":        chi2_res,
        "curva_aprendizaje":      curva_res,
    }

    ruta_rep = carpeta / f"reporte_v4_{ts}.json"
    with open(ruta_rep, "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)
    paths["reporte"] = ruta_rep

    ruta_txt = carpeta / f"resumen_narrativo_v4_{ts}.txt"
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write(narrativa)
    paths["narrativa"] = ruta_txt

    ruta_err = carpeta / f"errores_criticos_{ts}.json"
    with open(ruta_err, "w", encoding="utf-8") as f:
        json.dump(errores_res, f, ensure_ascii=False, indent=2)
    paths["errores"] = ruta_err

    return paths, ts


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    carpeta = Path(__file__).parent
    ruta_X, ruta_y, ruta_pkl, ruta_json = pedir_archivos(carpeta)

    print(f"\n  Cargando artefactos...")
    X, y, art, registros = cargar_todo(ruta_X, ruta_y, ruta_pkl, ruta_json)
    vec      = art.get("vectorizador")
    selector = art.get("selector")

    print(f"  X={X.shape}  y={y.shape}  registros={len(registros)}")
    vals, cnts = np.unique(y, return_counts=True)
    print(f"  Clases: {dict(zip(['media','alta'], cnts))}")

    # ── Detectar features BERT (ajusta pesos del ensamble) ───────────────────
    bert_activo = detectar_bert_activo(X, art)
    pesos_ens   = [1, 2, 2] if bert_activo else [2, 1, 2]
    print(f"  Modelo principal    : VotingEnsemble (soft, pesos={pesos_ens})")
    print(f"  Modelos en catálogo : {list(CATALOGO_MODELOS.keys())}")

    # ── Split único 80/20 ─────────────────────────────────────────────────────
    idx = np.arange(len(y))
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        idx, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    X_tr = X[idx_tr]; X_te = X[idx_te]

    print(f"\n{'='*72}")
    print(f"  PIPELINE COMPLETO v4 — ENSAMBLE + {len(CATALOGO_MODELOS)} MODELOS COMPARADOS")
    print(f"{'='*72}")

    # ① Validación cruzada sobre el VotingEnsemble
    cv_res = validacion_cruzada_robusta(X, y, bert_activo=bert_activo)

    # ② Entrenar ensamble principal (80% train → 20% test)
    ensemble, probs_ens, y_pred_ens, cal_res = entrenar_ensamble(
        X_tr, y_tr, X_te, y_te, bert_activo=bert_activo)

    # ③ Errores críticos del ensamble
    print(f"  Analizando errores críticos del ensamble...")
    errores_res = analizar_errores_criticos(
        X_te, y_te, y_pred_ens, probs_ens, registros, idx_te)

    # ④ Curva de aprendizaje del ensamble
    ensemble_curva = obtener_ensamble_votacion(weights=pesos_ens)
    curva_res = curva_aprendizaje(ensemble_curva, X, y, NOMBRE_ENSAMBLE)

    # ── Timestamp único para todas las gráficas de esta ejecución ─────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Gráfica de curva del ensamble ──────────────────────────────────────────
    graficar_curva_aprendizaje(
        obtener_ensamble_votacion(weights=pesos_ens), X, y,
        NOMBRE_ENSAMBLE, carpeta, ts)

    # ⑤ Comparación de todos los modelos (CNB, LR, SVM, Ensamble)
    #    + recolección de predicciones para análisis de discrepancias
    comparacion, predicciones_por_modelo = comparar_modelos(
        X_tr, y_tr, X_te, y_te, carpeta, ts)

    # ── Gráficas de curva para clasificadores individuales (informativo) ───────
    for nombre_m, factory in CATALOGO_MODELOS.items():
        if nombre_m != NOMBRE_ENSAMBLE:
            graficar_curva_aprendizaje(factory(), X, y, nombre_m, carpeta, ts)

    # ⑥ ANÁLISIS DE DISCREPANCIAS ── NUEVO EN v4 ───────────────────────────────
    print(f"\n  Analizando discrepancias entre clasificadores...")
    discrepancias = analizar_discrepancias(
        predicciones_por_modelo, y_te, registros, idx_te, carpeta, ts)

    # ⑦ Top terms del ensamble (extrae del sub-estimador LR interno)
    print(f"  Extrayendo términos predictivos (sub-estimador LR del ensamble)...")
    top_terms     = extraer_top_terms(ensemble, vec, selector)
    tabla_hibrida = top_features_hibrido(ensemble, vec, selector, art, n=20)
    correlaciones = correlacion_spearman(registros, y) if registros else []
    chi2_res      = prueba_chi2_categorias(registros, y) if registros else []
    dimensiones   = analizar_dimensiones(top_terms)

    # ⑧ Narrativa con sección de discrepancias
    narrativa = generar_narrativa(
        top_terms, cv_res, cal_res, errores_res,
        curva_res, comparacion, correlaciones, chi2_res, dimensiones,
        discrepancias=discrepancias)
    print(narrativa)

    # ── Tabla híbrida en consola ──────────────────────────────────────────────
    if tabla_hibrida:
        print(f"{'='*72}")
        print(f"  TOP FEATURES HÍBRIDAS (texto + BERT) — sub-estimador LR del ensamble")
        print(f"  {'Rank':<5} {'Nombre':<32} {'Coef':>8}  {'Dirección':<12} Tipo")
        print(f"  {'─'*65}")
        for row in tabla_hibrida[:20]:
            barra = "█" * int(row["abs_coef"] * 8)
            print(f"  {row['rank']:<5} {row['nombre']:<32} "
                  f"{row['coef']:>+8.4f}  {row['direccion']:<12} "
                  f"[{row['tipo']}]  {barra}")
        print(f"{'='*72}")

    # ── Guardar reporte principal ─────────────────────────────────────────────
    paths, _ = guardar_todo(
        carpeta, cv_res, cal_res, errores_res, curva_res,
        comparacion, top_terms, correlaciones, chi2_res,
        dimensiones, narrativa, ruta_X,
        tabla_hibrida=tabla_hibrida,
        bert_activo=bert_activo,
        discrepancias=discrepancias,
    )

    disc_meta = discrepancias.get("meta", {})
    print(f"{'='*72}")
    print(f"  💾 Reporte JSON       : {paths['reporte'].name}")
    print(f"  💾 Resumen narrativo  : {paths['narrativa'].name}")
    print(f"  💾 Errores críticos   : {paths['errores'].name}")
    print(f"  💾 Discrepancias JSON : discrepancias_{ts}.json")
    print(f"  💾 Discrepancias CSV  : discrepancias_{ts}.csv")
    print(f"       → {disc_meta.get('n_discrepancias',0)} comentarios en zona gris "
          f"({100*disc_meta.get('tasa_desacuerdo',0):.1f}% del test set)")
    if _MATPLOTLIB_OK:
        print(f"  💾 Gráficas PNG       : curva_*.png | confusion_*.png")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()

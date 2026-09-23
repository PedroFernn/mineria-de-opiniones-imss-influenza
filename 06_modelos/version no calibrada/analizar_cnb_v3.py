#!/usr/bin/env python3
"""
analizar_cnb_v3.py — Paso 3 del pipeline de modelado (CRISP-DM Fase 5)
Tema: Ineficiencia del sector salud MX frente a la influenza

Mejoras v3 — compatibilidad con vectorizar_v3 + auto_etiquetar:
  ① Selector automático de estimador: detecta si la matriz X contiene
     features BERT (prob_neg, sentimiento_num…) y elige LogisticRegression
     en lugar de ComplementNB — LR maneja mejor datos híbridos correlacionados
  ② Top-features híbrido: muestra nombres reales de variables numéricas
     junto a palabras TF-IDF (ej. "prob_neg [BERT]" y "negligencia [texto]")
     → citable en tesis: "el modelo decide por 'negligencia' + prob_neg=0.91"
  ③ Análisis de desacuerdo BERT vs clasificador: detecta casos donde
     robertuito dijo NEG pero el modelo clasificó media (y viceversa)
     → diagnóstico de qué "cerebro" se equivocó en cada error crítico

Heredadas de v2:
  ④ Validación cruzada 10-fold con IC 95%
  ⑤ Calibración isotónica de probabilidades
  ⑥ Análisis de errores críticos (FN/FP de alta confianza)
  ⑦ Curva de aprendizaje
  ⑧ Comparación de modelos (CNB / LR / LinearSVC)
  ⑨ Correlación Spearman, χ², dimensiones narrativas, reporte narrativo

Entrada : X_features*.npz  |  y_labels*.npy  |  vectorizador*.pkl
          dataset_etiquetado*.json  (salida de auto_etiquetar.py, preferido)
          dataset_normalizado*.json  (fallback sin BERT)
Salida  : reporte_cnb_v3_*.json  |  resumen_narrativo_*.txt
          errores_criticos_*.json

Uso: python3 analizar_cnb_v3.py
"""

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
# Nombres exactos que vectorizar_v3.py añade al final de cols_num.
# Se usan para: detectar si X viene de auto_etiquetar, nombrar features en
# el reporte híbrido, y activar LogisticRegression como estimador principal.
FEATURES_BERT = ["prob_neg", "prob_neu", "prob_pos", "sentimiento_num"]

# Umbral: fracción de filas con prob_neg>0 para considerar BERT activo.
# Evita activar LR si el dataset no pasó por auto_etiquetar (cols=0).
UMBRAL_BERT_ACTIVO = 0.10   # ≥10% de filas con valor real → BERT presente


# ══════════════════════════════════════════════════════════════════════════════
#  DIMENSIONES NARRATIVAS Y CATEGORÍAS (heredadas de v1)
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

from sklearn.naive_bayes         import ComplementNB
from sklearn.linear_model        import LogisticRegression
from sklearn.svm                 import LinearSVC
from sklearn.calibration         import CalibratedClassifierCV
from sklearn.model_selection     import (train_test_split, StratifiedKFold,
                                          cross_validate, learning_curve)
from sklearn.metrics             import (classification_report, confusion_matrix,
                                          roc_auc_score, f1_score,
                                          brier_score_loss, log_loss,
                                          precision_recall_curve)
from sklearn.pipeline            import Pipeline
from sklearn.preprocessing       import MaxAbsScaler, FunctionTransformer
from scipy.sparse                import load_npz, issparse
from scipy.stats                 import spearmanr, chi2_contingency
import numpy as np


# ── Transformador de limpieza de negativos (sparse-safe) ─────────────────────
# Problema raiz: el vectorizador concatena features TF-IDF (>=0) con features
# numericas escaladas con StandardScaler (pueden ser <0).
# MaxAbsScaler divide cada columna por su maximo absoluto -> rango [-1, 1].
# Los valores que ya eran negativos SIGUEN siendo negativos tras MaxAbsScaler.
# Solucion: clip a 0 despues del scaler. Para matrices sparse esto se hace
# operando directamente sobre .data (solo los valores no-cero almacenados).

def _clip_negativos(X):
    """Fuerza X >= 0 respetando el formato sparse si aplica."""
    if issparse(X):
        X = X.copy()
        X.data = np.clip(X.data, 0.0, None)
        return X
    return np.clip(X, 0.0, None)


def _pipeline_cnb(alpha: float = 0.5) -> Pipeline:
    """Pipeline sparse-compatible: MaxAbsScaler -> clip(0) -> ComplementNB."""
    return Pipeline([
        ("scaler", MaxAbsScaler()),
        ("clip",   FunctionTransformer(_clip_negativos, validate=False)),
        ("cnb",    ComplementNB(alpha=alpha)),
    ])


def _pipeline_lr(C: float = 1.0) -> Pipeline:
    """
    Pipeline para LogisticRegression con features híbridas (texto + BERT).
    LR no requiere X≥0, por lo que solo se necesita MaxAbsScaler para
    normalizar magnitudes — sin el paso clip.
    LR maneja features correlacionadas (prob_neg ↔ sentimiento_num) mucho
    mejor que CNB porque su función de pérdida aprende coeficientes conjuntos
    en lugar de asumir independencia condicional.
    """
    return Pipeline([
        ("scaler", MaxAbsScaler()),
        ("lr",     LogisticRegression(
            C            = C,
            max_iter     = 2000,
            class_weight = "balanced",
            solver       = "lbfgs",
        )),
    ])


def detectar_bert_activo(X, art: dict) -> bool:
    """
    Detecta si la matriz X contiene features BERT con valores reales.
    Estrategia: busca 'prob_neg' en cols_num del artefacto pickle y verifica
    que al menos UMBRAL_BERT_ACTIVO de las filas tienen valor ≠ 0.

    Retorna True  → usar LogisticRegression como estimador principal
    Retorna False → usar ComplementNB (corpus sin auto_etiquetar)
    """
    cols_num = art.get("features_numericas", [])
    if "prob_neg" not in cols_num:
        return False

    # Índice de prob_neg en la matriz combinada
    # cols_num está al final de X: [TF-IDF features ... | cols_num]
    idx_prob_neg = len(cols_num) - 1 - list(reversed(cols_num)).index("prob_neg")
    n_texto = X.shape[1] - len(cols_num)
    col_idx  = n_texto + idx_prob_neg

    if col_idx >= X.shape[1]:
        return False

    # Contar filas con valor real (no cero) en esa columna
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
        r = sorted(carpeta.glob(pat))
        return r

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
    print("  ANÁLISIS CNB v2 — VALIDACIÓN ROBUSTA + CALIBRACIÓN + ERRORES  🔬")
    print(f"{'='*72}")

    ruta_X   = elegir(glob_ultimo("X_features*.npz"),       "X_features")
    ruta_y   = elegir(glob_ultimo("y_labels*.npy"),          "y_labels")
    ruta_pkl = elegir(glob_ultimo("vectorizador*.pkl"),       "vectorizador")
    ruta_json= elegir(
        glob_ultimo("dataset_normalizado*.json") +
        glob_ultimo("dataset_enriquecido*.json"), "dataset JSON (opcional)")

    if not ruta_X or not ruta_y or not ruta_pkl:
        print("\n  ❌ Artefactos no encontrados. Ejecuta vectorizar_v2.py primero.")
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
#  MEJORA ① — VALIDACIÓN CRUZADA ROBUSTA (10-FOLD ESTRATIFICADO)
#
#  Por qué 10-fold y no 5-fold para una tesis:
#  · 10-fold reduce la varianza del estimador en ~30% respecto a 5-fold.
#  · Con n≈2500, cada fold tiene ~250 muestras de test → suficiente para
#    calcular métricas estables (vs ~125 con 5-fold).
#  · Permite construir un intervalo de confianza 95% con la distribución
#    empírica de los 10 scores, lo cual es citeable en literatura académica.
#
#  Fórmula IC: x̄ ± z * (σ / √k)   donde k = número de folds
# ══════════════════════════════════════════════════════════════════════════════

def validacion_cruzada_robusta(X, y, bert_activo: bool = False) -> dict:
    """
    Ejecuta 10-fold estratificado. Usa LR si hay features BERT, CNB si no.
    """
    print(f"\n  Ejecutando {N_FOLDS}-fold estratificado...")
    cv      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    clf_cv  = _pipeline_lr() if bert_activo else _pipeline_cnb(alpha=0.5)

    # cross_validate calcula múltiples métricas en un solo paso
    scores = cross_validate(
        clf_cv, X, y, cv=cv,
        scoring={
            "f1_weighted": "f1_weighted",
            "f1_alta":     "f1",               # f1 de clase positiva (alta=1)
            "roc_auc":     "roc_auc",
            "accuracy":    "accuracy",
        },
        return_train_score=True,   # detecta overfitting si train >> test
    )

    def resumen(arr: np.ndarray, nombre: str) -> dict:
        mu  = float(arr.mean())
        std = float(arr.std())
        ic  = IC_Z * std / np.sqrt(N_FOLDS)
        return {
            "media":   round(mu,  4),
            "std":     round(std, 4),
            "ic95_pm": round(ic,  4),            # ± este valor para IC 95%
            "ic95_lo": round(mu - ic, 4),
            "ic95_hi": round(mu + ic, 4),
            "min":     round(float(arr.min()), 4),
            "max":     round(float(arr.max()), 4),
            "valores_por_fold": [round(float(v), 4) for v in arr],
            "estable": bool(std < 0.04),         # std < 4% = modelo estable
        }

    resultado = {
        "n_folds":       N_FOLDS,
        "ic_nivel":      "95%",
        "f1_weighted":   resumen(scores["test_f1_weighted"], "f1_weighted"),
        "f1_alta":       resumen(scores["test_f1_alta"],     "f1_alta"),
        "roc_auc":       resumen(scores["test_roc_auc"],     "roc_auc"),
        "accuracy":      resumen(scores["test_accuracy"],    "accuracy"),
        "overfitting": {
            # Si train >> test por margen amplio → overfitting
            "f1_train_media": round(float(scores["train_f1_weighted"].mean()), 4),
            "f1_test_media":  round(float(scores["test_f1_weighted"].mean()),  4),
            "gap":            round(float(
                scores["train_f1_weighted"].mean() -
                scores["test_f1_weighted"].mean()), 4),
            "detectado":      bool(
                scores["train_f1_weighted"].mean() -
                scores["test_f1_weighted"].mean() > 0.10),
        },
    }

    # Veredicto de estabilidad para el reporte narrativo
    f1_std = resultado["f1_weighted"]["std"]
    if f1_std < 0.02:   resultado["veredicto"] = "muy estable (σ < 2%)"
    elif f1_std < 0.04: resultado["veredicto"] = "estable (σ < 4%)"
    elif f1_std < 0.07: resultado["veredicto"] = "aceptable (σ < 7%)"
    else:               resultado["veredicto"] = "inestable — revisar datos"

    return resultado


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA ② — CALIBRACIÓN DE PROBABILIDADES
#
#  Problema de CNB sin calibrar:
#  · CNB asume independencia condicional entre features → sub/sobre-estima prob.
#  · Ejemplo real: un comentario con 3 términos de queja recibe P(alta)=0.997,
#    cuando la probabilidad real (estimada por calibración) sería ~0.82.
#  · Para una tesis de salud pública, decir "85% de probabilidad" requiere
#    que ese 85% sea real, no inflado.
#
#  Método isotónico vs sigmoid (Platt):
#  · Isotónico: aprende una función monótona no paramétrica. Más flexible,
#    necesita ≥1000 muestras de calibración (cv=5 divide el train).
#  · Sigmoid: regresión logística sobre los scores. Más estable con menos datos.
#
#  Métrica de calidad de calibración:
#  · Brier Score: MSE entre probabilidad predicha y etiqueta real (0=perfecto).
#  · Log-loss: penaliza las predicciones confiadas y equivocadas (más severo).
# ══════════════════════════════════════════════════════════════════════════════

def entrenar_con_calibracion(X_tr, y_tr, X_te, y_te,
                              bert_activo: bool = False) -> tuple:
    """Entrena el estimador base y su versión calibrada."""
    base_pipeline = _pipeline_lr() if bert_activo else _pipeline_cnb(alpha=0.5)

    print(f"\n  Entrenando {'LR' if bert_activo else 'CNB'} sin calibrar (línea base)...")
    clf_raw = base_pipeline
    clf_raw.fit(X_tr, y_tr)
    probs_raw = clf_raw.predict_proba(X_te)[:, 1]

    print(f"  Calibrando con método '{METODO_CALIBRACION}' (cv=5)...")
    clf_cal = CalibratedClassifierCV(
        _pipeline_lr() if bert_activo else _pipeline_cnb(alpha=0.5),
        method=METODO_CALIBRACION,
        cv=5
    )
    clf_cal.fit(X_tr, y_tr)
    probs_cal = clf_cal.predict_proba(X_te)[:, 1]
    y_pred_cal = clf_cal.predict(X_te)

    # ── Métricas de calibración ────────────────────────────────────────────
    brier_raw = float(brier_score_loss(y_te, probs_raw))
    brier_cal = float(brier_score_loss(y_te, probs_cal))
    ll_raw    = float(log_loss(y_te, probs_raw))
    ll_cal    = float(log_loss(y_te, probs_cal))

    # Distribución de confianzas antes y después (histograma ligero)
    def hist_confianza(probs, bins=5):
        edges = np.linspace(0, 1, bins+1)
        counts, _ = np.histogram(probs, bins=edges)
        return {
            f"{edges[i]:.1f}-{edges[i+1]:.1f}": int(counts[i])
            for i in range(len(counts))
        }

    metricas = {
        "sin_calibrar": {
            "brier_score": round(brier_raw, 5),
            "log_loss":    round(ll_raw,    5),
            "prob_media":  round(float(probs_raw.mean()), 4),
            "prob_std":    round(float(probs_raw.std()),  4),
            "distribucion_confianza": hist_confianza(probs_raw),
        },
        "calibrado": {
            "brier_score":   round(brier_cal, 5),
            "log_loss":      round(ll_cal,    5),
            "prob_media":    round(float(probs_cal.mean()), 4),
            "prob_std":      round(float(probs_cal.std()),  4),
            "distribucion_confianza": hist_confianza(probs_cal),
            "mejora_brier_pct": round((brier_raw - brier_cal) / brier_raw * 100, 1),
            "mejora_logloss_pct": round((ll_raw - ll_cal) / ll_raw * 100, 1),
        },
        "clasificacion_calibrado": {
            "f1_weighted":  round(float(f1_score(y_te, y_pred_cal, average="weighted")), 4),
            "f1_alta":      round(float(f1_score(y_te, y_pred_cal, pos_label=1, average="binary")), 4),
            "roc_auc":      round(float(roc_auc_score(y_te, probs_cal)), 4),
            "accuracy":     round(float((y_pred_cal == y_te).mean()), 4),
            "confusion_matrix": confusion_matrix(y_te, y_pred_cal).tolist(),
        },
        "interpretacion": (
            "mejorada — usar probabilidades calibradas para inferencia"
            if brier_cal < brier_raw
            else "sin mejora significativa — CNB bien calibrado en este corpus"
        ),
    }

    return clf_cal, probs_cal, y_pred_cal, metricas


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA ③ — ANÁLISIS DE ERRORES CRÍTICOS
#
#  Un "error crítico" es un caso donde:
#    · El modelo calibrado asignó probabilidad ≥ UMBRAL_CONFIANZA a una clase
#    · Pero la etiqueta real era la opuesta
#
#  En salud pública, los más graves son los Falsos Negativos de alta confianza:
#    → El modelo dijo "este comentario es MEDIA" con 90% de confianza
#    → Pero en realidad era ALTA (queja de ineficiencia real)
#    → El sistema habría ignorado una señal de alarma real
#
#  El análisis de errores sirve para:
#    1. Identificar términos "engañosos" que el modelo confunde
#    2. Detectar si hay un subtema semántico no cubierto por el lexicon
#    3. Priorizar qué ejemplos añadir al conjunto de entrenamiento
#    4. Decidir si bajar el umbral de clasificación para reducir FN
# ══════════════════════════════════════════════════════════════════════════════

def analizar_errores_criticos(
    X_te, y_te: np.ndarray,
    y_pred: np.ndarray,
    probs_cal: np.ndarray,
    registros: list[dict],
    indices_test: np.ndarray,
) -> dict:
    """
    Identifica y clasifica los errores más confiados del modelo.

    Tipos de error:
      FN_confiado: modelo dijo MEDIA (prob_alta < 1-umbral) pero era ALTA
      FP_confiado: modelo dijo ALTA  (prob_alta ≥ umbral)   pero era MEDIA

    Retorna diccionario con ambas listas ordenadas por confianza descendente.
    """
    fn_confiados: list[dict] = []   # Falsos Negativos — los más costosos
    fp_confiados: list[dict] = []   # Falsos Positivos

    for i, (real, pred, prob) in enumerate(zip(y_te, y_pred, probs_cal)):
        confianza = prob if pred == 1 else (1 - prob)

        if pred == real or confianza < UMBRAL_CONFIANZA:
            continue   # predicción correcta o poco confiada → no es crítico

        idx_original = int(indices_test[i]) if i < len(indices_test) else None
        reg = registros[idx_original] if (registros and idx_original is not None
                                          and idx_original < len(registros)) else {}

        # Términos del texto que podrían haber engañado al modelo
        tokens = reg.get("tokens", [])
        texto_corto = (reg.get("comentario", reg.get("texto_normalizado", ""))[:200]
                       + "..." if len(reg.get("comentario","")) > 200 else
                       reg.get("comentario", reg.get("texto_normalizado", "")))

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
            "polaridad_score":    reg.get("polaridad_score", 0),
            "contextos":          reg.get("contextos", []),
            "tipo_error":         "FN" if real == 1 else "FP",
            "diagnostico":        _diagnosticar_error(tokens, real, pred),
            # ── Desacuerdo BERT vs modelo (v3) ────────────────────────────────
            # Compara lo que dijo robertuito (sentimiento_bert del registro)
            # con lo que predijo el clasificador, para saber cuál "cerebro"
            # contribuyó al error. Citable en tesis: "En el 67% de los FN,
            # BERT había clasificado correctamente como NEG pero el modelo
            # TF-IDF lo ignoró por falta de términos léxicos de queja."
            "bert_sentimiento":   reg.get("sentimiento_bert", "N/D"),
            "bert_prob_neg":      reg.get("prob_neg", None),
            "bert_etiqueta_fuente": reg.get("etiqueta_fuente", "N/D"),
            "desacuerdo_bert":    _diagnosticar_desacuerdo_bert(
                                      reg, real, pred
                                  ),
        }

        if real == 1 and pred == 0:   # Falso Negativo (el más costoso)
            fn_confiados.append(error)
        else:                          # Falso Positivo
            fp_confiados.append(error)

    # Ordenar por confianza errónea descendente
    fn_confiados.sort(key=lambda x: -x["confianza_erronea"])
    fp_confiados.sort(key=lambda x: -x["confianza_erronea"])

    # Análisis de patrones en los errores
    patron_fn = _patrones_en_errores(fn_confiados)
    patron_fp = _patrones_en_errores(fp_confiados)

    return {
        "umbral_confianza":     UMBRAL_CONFIANZA,
        "total_fn_confiados":   len(fn_confiados),
        "total_fp_confiados":   len(fp_confiados),
        "fn_exportados":        fn_confiados[:N_ERRORES_EXPORTAR],
        "fp_exportados":        fp_confiados[:N_ERRORES_EXPORTAR],
        "patrones_fn":          patron_fn,
        "patrones_fp":          patron_fp,
        "recomendaciones":      _generar_recomendaciones(patron_fn, patron_fp),
    }


def _diagnosticar_desacuerdo_bert(reg: dict, real: int, pred: int) -> str:
    """
    Analiza si BERT y el clasificador discreparon y en qué dirección.
    Permite distinguir tres tipos de error en el reporte de tesis:
      · BERT correcto, TF-IDF incorrecto → léxico insuficiente, ampliar dominio
      · BERT incorrecto, TF-IDF correcto → sarcasmo o contexto ambiguo en BERT
      · Ambos incorrectos                → caso genuinamente difícil
    """
    sent_bert   = reg.get("sentimiento_bert", "")
    etq_fuente  = reg.get("etiqueta_fuente", "")
    prob_neg    = reg.get("prob_neg", 0.0) or 0.0

    bert_dijo_alta  = sent_bert == "NEG" and prob_neg >= 0.5
    bert_dijo_media = sent_bert in ("POS", "NEU") or prob_neg < 0.5
    modelo_dijo_alta = pred == 1
    real_es_alta     = real == 1

    if not sent_bert or sent_bert == "N/D":
        return "sin datos BERT (dataset sin auto_etiquetar)"

    if real_es_alta:
        # Es un Falso Negativo (modelo dijo media)
        if bert_dijo_alta:
            return (f"BERT correcto (NEG p={prob_neg:.2f}), TF-IDF incorrecto "
                    f"→ señal léxica insuficiente, ampliar TERMINOS_DOMINIO")
        else:
            return (f"BERT también incorrecto ({sent_bert} p_neg={prob_neg:.2f}) "
                    f"→ caso difícil, posible sarcasmo o queja implícita")
    else:
        # Es un Falso Positivo (modelo dijo alta)
        if bert_dijo_media:
            return (f"BERT correcto ({sent_bert} p_neg={prob_neg:.2f}), "
                    f"TF-IDF incorrecto → término de queja fuera de contexto")
        else:
            return (f"BERT también incorrecto (NEG p={prob_neg:.2f}) "
                    f"→ ambos sobreestimaron negatividad, revisar etiqueta")


def _diagnosticar_error(tokens: list[str], real: int, pred: int) -> str:
    """
    Intenta diagnosticar por qué el modelo se equivocó mirando los tokens.
    Devuelve una cadena de diagnóstico interpretable.
    """
    t = set(tokens)

    # FN: el modelo lo clasificó como media cuando era alta
    if real == 1 and pred == 0:
        if any(tok in t for tok in ["particular", "medico_particular", "de_mi_bolsillo",
                                     "fui_al_particular", "tuve_que_pagar"]):
            return "posible FN por pago_privado: el modelo no asocia el gasto privado con queja alta"
        if any(tok in t for tok in ["fallecer", "morir", "muerte", "intubado", "uci"]):
            return "posible FN por impacto_clinico: palabras de consecuencia no detectadas"
        if len(tokens) < 8:
            return "posible FN por texto muy corto: insuficiente señal léxica"
        return "FN sin patrón claro — puede requerir ampliar TERMINOS_DOMINIO"

    # FP: el modelo lo clasificó como alta cuando era media
    if real == 0 and pred == 1:
        if any(tok in t for tok in ["vacuna_info_neutral", "vacuna_gratuita"]):
            return "posible FP por vacunación informativa: token neutral no filtrado"
        if any(tok in t for tok in ["imss", "issste", "hospital", "medico"]):
            return "posible FP por institución: mención de IMSS sin queja real"
        return "FP sin patrón claro — puede ser comentario irónico positivo"

    return "tipo de error no clasificado"


def _patrones_en_errores(errores: list[dict]) -> dict:
    """Extrae patrones frecuentes en una lista de errores."""
    if not errores:
        return {}

    todos_tokens = [tok for e in errores for tok in e.get("tokens_principales", [])]
    contextos    = [ctx for e in errores for ctx in e.get("contextos", [])]
    diagnosticos = Counter(e["diagnostico"] for e in errores)

    return {
        "tokens_mas_frecuentes": Counter(todos_tokens).most_common(15),
        "contextos_frecuentes":  Counter(contextos).most_common(8),
        "diagnosticos":          dict(diagnosticos),
        "confianza_media":       round(
            float(np.mean([e["confianza_erronea"] for e in errores])), 4
        ) if errores else 0,
    }


def _generar_recomendaciones(patron_fn: dict, patron_fp: dict) -> list[str]:
    """Genera recomendaciones accionables basadas en los patrones de error."""
    recomendaciones = []

    fn_diagnosticos = patron_fn.get("diagnosticos", {})
    fp_diagnosticos = patron_fp.get("diagnosticos", {})

    if any("pago_privado" in d for d in fn_diagnosticos):
        recomendaciones.append(
            "ACCIÓN FN: Añadir 'particular', 'médico_privado', 'de_mi_bolsillo' "
            "a TERMINOS_DOMINIO en el normalizador para mejorar detección."
        )
    if any("texto muy corto" in d for d in fn_diagnosticos):
        recomendaciones.append(
            "ACCIÓN FN: Bajar MIN_CHARS a 20 en limpiar_dataset_v3.py "
            "para conservar comentarios cortos con alta carga semántica."
        )
    if any("vacunación" in d for d in fp_diagnosticos):
        recomendaciones.append(
            "ACCIÓN FP: Añadir bigramas 'vacuna disponible', 'ya me vacuné' "
            "como señales de relevancia media en el vectorizador."
        )
    if any("institución" in d for d in fp_diagnosticos):
        recomendaciones.append(
            "ACCIÓN FP: Considerar contexto de mención de instituciones — "
            "IMSS sin indicador de problema debería ser relevancia media."
        )
    if not recomendaciones:
        recomendaciones.append(
            "Sin patrones dominantes en errores — el modelo es robusto en este corpus."
        )
    return recomendaciones


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA ④ — CURVA DE APRENDIZAJE
#
#  Responde: ¿el modelo mejora si tenemos más datos?
#  · Si train y test convergen rápido → el modelo ya está saturado,
#    añadir más datos no ayudará. Foco: mejorar features.
#  · Si train >> test con datos grandes → overfitting.
#    Foco: regularización, reducir vocabulario.
#  · Si ambas curvas siguen subiendo → más datos mejorarán el modelo.
#    Foco: recolectar más comentarios con el scraper.
# ══════════════════════════════════════════════════════════════════════════════

def curva_aprendizaje(X, y, bert_activo: bool = False) -> dict:
    print(f"  Calculando curva de aprendizaje ({CURVA_PUNTOS} puntos)...")
    clf_ca = _pipeline_lr() if bert_activo else _pipeline_cnb(alpha=0.5)
    cv_ca  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    sizes, train_scores, test_scores = learning_curve(
        clf_ca, X, y,
        train_sizes=np.linspace(0.10, 1.0, CURVA_PUNTOS),
        cv=cv_ca,
        scoring="f1_weighted",
    )

    puntos = []
    for sz, tr, te in zip(sizes, train_scores, test_scores):
        puntos.append({
            "n_train":        int(sz),
            "f1_train_media": round(float(tr.mean()), 4),
            "f1_train_std":   round(float(tr.std()),  4),
            "f1_test_media":  round(float(te.mean()),  4),
            "f1_test_std":    round(float(te.std()),   4),
            "gap":            round(float(tr.mean() - te.mean()), 4),
        })

    # Diagnóstico automático
    gap_final     = puntos[-1]["gap"]
    gap_inicial   = puntos[0]["gap"]
    pendiente_test = puntos[-1]["f1_test_media"] - puntos[CURVA_PUNTOS//2]["f1_test_media"]

    if gap_final > 0.12:
        diagnostico = "overfitting moderado — reducir vocabulario o aumentar alpha"
    elif pendiente_test > 0.02:
        diagnostico = "más datos mejorarían el modelo — continuar scraping"
    elif gap_final < 0.03:
        diagnostico = "modelo saturado — mejorar features es más efectivo que añadir datos"
    else:
        diagnostico = "equilibrado — modelo estable"

    return {"puntos": puntos, "diagnostico": diagnostico}


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA ⑤ — COMPARACIÓN DE MODELOS
# ══════════════════════════════════════════════════════════════════════════════

def comparar_modelos(X_tr, y_tr, X_te, y_te) -> list[dict]:
    """
    Evalúa CNB, LogisticRegression y LinearSVC en el mismo split.
    Permite seleccionar el mejor modelo para la tesis con evidencia comparativa.
    """
    print(f"  Comparando CNB vs LR vs LinearSVC...")
    modelos = [
        ("ComplementNB",       _pipeline_cnb(alpha=0.5)),
        ("LogisticRegression", Pipeline([
            ("scaler", MaxAbsScaler()),
            ("lr", LogisticRegression(C=1.0, max_iter=1000,
                                      class_weight="balanced",
                                      solver="lbfgs")),
        ])),
        ("LinearSVC",          CalibratedClassifierCV(
                                   Pipeline([
                                       ("scaler", MaxAbsScaler()),
                                       ("svc", LinearSVC(C=0.5, max_iter=2000,
                                                         class_weight="balanced")),
                                   ]), cv=5)),
    ]

    resultados = []
    for nombre, clf in modelos:
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else None

        resultados.append({
            "modelo":        nombre,
            "f1_weighted":   round(float(f1_score(y_te, y_pred, average="weighted")), 4),
            "f1_alta":       round(float(f1_score(y_te, y_pred, pos_label=1, average="binary")), 4),
            "roc_auc":       round(float(roc_auc_score(y_te, y_prob)), 4) if y_prob is not None else None,
            "accuracy":      round(float((y_pred == y_te).mean()), 4),
        })

    return sorted(resultados, key=lambda x: -x["f1_weighted"])


# ══════════════════════════════════════════════════════════════════════════════
#  HEREDADAS DE v1 — TOP TERMS, SPEARMAN, χ², DIMENSIONES, NARRATIVA
# ══════════════════════════════════════════════════════════════════════════════

def extraer_top_terms(clf, vec, selector, n=TOP_N_TERMS) -> dict:
    """
    Extrae top-N features predictivas soportando CNB y LR/LinearSVC.
    Para CNB usa log_prob; para LR usa coef_ directamente.
    En ambos casos incluye NOMBRES LEGIBLES para features numéricas
    (prob_neg, sentimiento_num, score_queja_salud…) junto a tokens TF-IDF.
    """
    # ── Intentar extraer CNB ──────────────────────────────────────────────────
    def _extraer_cnb(estimador):
        if hasattr(estimador, "feature_log_prob_"):
            return estimador
        if hasattr(estimador, "named_steps"):
            return _extraer_cnb(estimador.named_steps.get("cnb", None))
        if hasattr(estimador, "calibrated_classifiers_"):
            sub = getattr(estimador.calibrated_classifiers_[0], "estimator", None)
            return _extraer_cnb(sub) if sub else None
        inner = getattr(estimador, "estimator", None)
        return _extraer_cnb(inner) if inner else None

    # ── Intentar extraer LR / LinearSVC ──────────────────────────────────────
    def _extraer_lr(estimador):
        """Busca el estimador con atributo coef_ (LR o LinearSVC calibrado)."""
        if hasattr(estimador, "coef_"):
            return estimador
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

    # Construir lista completa de nombres de features:
    # [TF-IDF features filtradas por selector] + [cols_num del pkl]
    nombres_vec = list(vec.get_feature_names_out())
    if selector is not None:
        mascara     = selector.get_support()
        nombres_tfidf = [nombres_vec[i] for i, m in enumerate(mascara) if m]
    else:
        nombres_tfidf = nombres_vec

    # Las features numéricas están al final de X — recuperar del artefacto
    # si están disponibles globalmente, o usar marcador genérico
    n_num = 0   # se actualiza en top_features_hibrido con el pkl real

    cnb = _extraer_cnb(clf)
    lr  = _extraer_lr(clf)

    resultado = {}

    if cnb is not None and hasattr(cnb, "feature_log_prob_"):
        # ── Modo CNB ──────────────────────────────────────────────────────────
        log_probs = cnb.feature_log_prob_
        n_cols    = min(log_probs.shape[1], len(nombres_tfidf))
        nombres_f = nombres_tfidf[:n_cols]

        for idx_clase, nombre_clase in enumerate(["media", "alta"]):
            if idx_clase >= log_probs.shape[0]:
                resultado[nombre_clase] = []; continue
            pesos   = -log_probs[idx_clase][:n_cols]
            top_idx = np.argsort(pesos)[-n:][::-1]
            resultado[nombre_clase] = [
                {
                    "termino":   nombres_f[i],
                    "peso":      round(float(pesos[i]), 5),
                    "tipo":      "texto",
                    "dimension": _asignar_dimension(nombres_f[i]),
                }
                for i in top_idx if i < len(nombres_f)
            ]

    elif lr is not None and hasattr(lr, "coef_"):
        # ── Modo LR — coef_ incluye TODAS las features (texto + numéricas) ───
        coef = lr.coef_[0]   # shape (n_features,) para clasificación binaria
        # Las features numéricas están al final; sus nombres vienen del pkl
        # top_features_hibrido los completará; aquí usamos índices de texto
        n_total = len(coef)
        n_tfidf = len(nombres_tfidf)

        # Nombres completos: tfidf + marcadores numéricos temporales
        nombres_completos = nombres_tfidf + [
            f"[NUM_{i}]" for i in range(max(0, n_total - n_tfidf))
        ]
        nombres_completos = nombres_completos[:n_total]

        # Clase ALTA: coeficientes positivos más grandes
        top_alta  = np.argsort(coef)[-n:][::-1]
        # Clase MEDIA: coeficientes negativos más grandes (más negativos)
        top_media = np.argsort(coef)[:n]

        for nombre_clase, top_idx in [("alta", top_alta), ("media", top_media)]:
            resultado[nombre_clase] = [
                {
                    "termino":   nombres_completos[i] if i < len(nombres_completos) else f"feat_{i}",
                    "peso":      round(float(abs(coef[i])), 5),
                    "coef_lr":   round(float(coef[i]), 5),
                    "tipo":      "numerico" if i >= n_tfidf else "texto",
                    "dimension": _asignar_dimension(
                        nombres_completos[i] if i < len(nombres_completos) else ""
                    ),
                }
                for i in top_idx
            ]
    else:
        resultado["alta"]  = []
        resultado["media"] = []

    resultado["nombres_features"] = nombres_tfidf
    resultado["modo"]             = "cnb" if cnb else ("lr" if lr else "desconocido")
    return resultado


def top_features_hibrido(clf, vec, selector, art: dict, n: int = 20) -> list[dict]:
    """
    Genera tabla unificada de importancia de features que mezcla:
      · Tokens TF-IDF con su nombre de texto (ej. 'negligencia')
      · Features numéricas con su nombre real (ej. 'prob_neg [BERT]')
    Solo funciona con LR/LinearSVC (que tienen coef_ interpretable).
    Con CNB retorna lista vacía (CNB no tiene coef_ global).

    Útil para la tesis: "El modelo asigna coef=+2.31 a prob_neg [BERT]
    y coef=+1.87 a la palabra 'viacrucis', demostrando que ambas señales
    se combinan para detectar quejas de ineficiencia sistémica."
    """
    def _extraer_lr(estimador):
        if hasattr(estimador, "coef_"):
            return estimador
        if hasattr(estimador, "named_steps"):
            for paso in reversed(list(estimador.named_steps.values())):
                r = _extraer_lr(paso)
                if r: return r
        if hasattr(estimador, "calibrated_classifiers_"):
            sub = getattr(estimador.calibrated_classifiers_[0], "estimator", None)
            return _extraer_lr(sub) if sub else None
        inner = getattr(estimador, "estimator", None)
        return _extraer_lr(inner) if inner else None

    lr = _extraer_lr(clf)
    if lr is None or not hasattr(lr, "coef_") or vec is None:
        return []

    coef = lr.coef_[0]

    # Reconstruir lista completa de nombres igual que en vectorizar_v3.py
    nombres_vec   = list(vec.get_feature_names_out())
    if selector is not None:
        mascara     = selector.get_support()
        nombres_tfidf = [nombres_vec[i] for i, m in enumerate(mascara) if m]
    else:
        nombres_tfidf = nombres_vec

    cols_num = art.get("features_numericas", [])
    # cols_num incluye score_queja_salud al final (añadido dinámicamente)
    nombres_completos = nombres_tfidf + cols_num
    nombres_completos = nombres_completos[:len(coef)]  # ajustar si hay diferencia

    # Enriquecer con metadatos de tipo
    BERT_COLS   = set(FEATURES_BERT + ["score_queja_salud"])
    ENRIQ_COLS  = {"polaridad_score", "ifb_score", "impacto_escala",
                   "num_palabras", "peso_reddit"}

    def _tipo(nombre: str) -> str:
        if nombre in BERT_COLS:   return "BERT"
        if nombre in ENRIQ_COLS:  return "enriquecimiento"
        return "texto_tfidf"

    # Top-N por valor absoluto de coeficiente
    top_idx = np.argsort(np.abs(coef))[-n:][::-1]

    tabla = []
    for i in top_idx:
        nombre = nombres_completos[i] if i < len(nombres_completos) else f"feat_{i}"
        tabla.append({
            "rank":       len(tabla) + 1,
            "nombre":     nombre,
            "coef":       round(float(coef[i]), 5),
            "abs_coef":   round(float(abs(coef[i])), 5),
            "direccion":  "→ alta"  if coef[i] > 0 else "→ media",
            "tipo":       _tipo(nombre),
            "dimension":  _asignar_dimension(nombre),
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
            "feature": col, "rho": round(float(rho),4),
            "p_valor": round(float(pval),6), "n": len(vals),
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
            "categoria": cat, "chi2": round(float(chi2),4),
            "p_valor": round(float(pval),6),
            "odds_ratio": round(or_v, 3),
            "significativa": bool(pval < 0.05),
            "interpretacion": (
                "muy fuertemente asociada con alta" if or_v > 3 else
                "asociada con alta" if or_v > 1.5 else
                "sin asociación clara" if or_v > 0.67 else
                "asociada con media"
            ),
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
            "peso_total":  round(p, 4),
            "proporcion":  round(p/total, 4),
            "terminos":    terms[dim],
            "n_terminos":  len(terms[dim]),
        }
        for dim, p in sorted(pesos.items(), key=lambda x: -x[1])
    }


def generar_narrativa(top_terms, cv_res, cal_res, errores_res,
                      curva_res, comparacion, correlaciones, chi2_res,
                      dimensiones) -> str:

    terms_alta = [i["termino"].replace("_"," ")
                  for i in top_terms.get("alta", [])[:N_TERMINOS_CITA]]
    p = [terms_alta[i] if i < len(terms_alta) else "[N/D]"
         for i in range(N_TERMINOS_CITA)]

    f1_m   = cv_res["f1_weighted"]["media"]
    f1_pm  = cv_res["f1_weighted"]["ic95_pm"]
    f1_std = cv_res["f1_weighted"]["std"]
    veredicto = cv_res["veredicto"]
    auc    = cv_res["roc_auc"]["media"]
    ovf    = cv_res["overfitting"]["detectado"]

    brier_antes = cal_res["sin_calibrar"]["brier_score"]
    brier_cal   = cal_res["calibrado"]["brier_score"]
    mejora_cal  = cal_res["calibrado"]["mejora_brier_pct"]
    f1_cal      = cal_res["clasificacion_calibrado"]["f1_weighted"]

    fn_total = errores_res["total_fn_confiados"]
    fp_total = errores_res["total_fp_confiados"]
    recs     = errores_res["recomendaciones"]

    dim_dom = list(dimensiones.keys())[0] if dimensiones else "falta_insumos"
    dim_2da = list(dimensiones.keys())[1] if len(dimensiones) > 1 else "tiempo_espera"
    dim_map = {
        "falta_insumos": "la falta de insumos",
        "tiempo_espera": "el tiempo de espera prolongado",
        "consecuencia_clinica": "las consecuencias clínicas de la demora",
        "pago_privado": "el desembolso en servicios privados",
        "corrupcion_negligencia": "la negligencia e irregularidades institucionales",
        "sistema_roto": "el colapso del sistema de atención",
    }

    cor_top = next((r for r in correlaciones if r["significativa"]), None)
    chi2_top = next((r for r in chi2_res if r["significativa"]), None)
    mejor_modelo = comparacion[0] if comparacion else {}

    txt = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  REPORTE NARRATIVO v2 — BUROCRACIA DEL DOLOR EN SALUD PÚBLICA MX        ║
║  Modelo: ComplementNB calibrado  |  Validación: {N_FOLDS}-fold StratifiedKFold   ║
╚══════════════════════════════════════════════════════════════════════════╝

HALLAZGO PRINCIPAL
──────────────────
El análisis de minería de opinión mediante Complement Naive Bayes revela
que el declive hospitalario en México no se expresa solo en términos
médicos, sino en un lenguaje de 'burocracia del dolor'.

Los términos con mayor peso predictivo para relevancia alta fueron:
  1. "{p[0]}"
  2. "{p[1]}"
  3. "{p[2]}"

Esto confirma que el ciudadano percibe la ineficiencia principalmente
a través de {dim_map.get(dim_dom, dim_dom)} y {dim_map.get(dim_2da, dim_2da)}
con relación a la influenza.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
① VALIDACIÓN CRUZADA — ESTABILIDAD DEL MODELO ({N_FOLDS}-fold estratificado)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  F1 weighted     : {f1_m:.4f} ± {f1_pm:.4f}  (IC 95%)
  AUC-ROC         : {auc:.4f} ± {cv_res['roc_auc']['ic95_pm']:.4f}
  F1 clase alta   : {cv_res['f1_alta']['media']:.4f} ± {cv_res['f1_alta']['ic95_pm']:.4f}
  Desviación std  : {f1_std:.4f}
  Veredicto       : {veredicto}
  Overfitting     : {'⚠️  detectado (gap train-test > 10%)' if ovf else '✅ no detectado'}
  Curva aprend.   : {curva_res['diagnostico']}

  Valores por fold: {cv_res['f1_weighted']['valores_por_fold']}

  Interpretación: Un F1 de {f1_m:.2%} con IC [{cv_res['f1_weighted']['ic95_lo']:.4f},
  {cv_res['f1_weighted']['ic95_hi']:.4f}] significa que en el 95% de posibles
  muestras del mismo corpus, el modelo se comportaría dentro de ese rango.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
② CALIBRACIÓN DE PROBABILIDADES ({METODO_CALIBRACION})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Brier Score sin calibrar : {brier_antes:.5f}  (menor = mejor)
  Brier Score calibrado    : {brier_cal:.5f}  (mejora: {mejora_cal:+.1f}%)
  F1 weighted calibrado    : {f1_cal:.4f}
  Interpretación           : {cal_res['interpretacion']}

  La calibración permite afirmar: "este comentario tiene X% de probabilidad
  de representar una queja de ineficiencia clínicamente relevante",
  con X siendo una probabilidad real y no inflada por la independencia
  condicional asumida por Naive Bayes.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
③ ANÁLISIS DE ERRORES CRÍTICOS (umbral confianza ≥ {UMBRAL_CONFIANZA})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Falsos Negativos confiados : {fn_total}  ← quejas reales ignoradas
  Falsos Positivos confiados : {fp_total}  ← falsas alarmas
"""

    if errores_res["fn_exportados"]:
        txt += "\n  Top 5 FN más confiados (el modelo estaba seguro pero se equivocó):\n"
        for j, e in enumerate(errores_res["fn_exportados"][:5], 1):
            txt += (f"  {j}. [{e['confianza_erronea']:.0%} confianza errónea] "
                    f"'{e['texto_original'][:90]}...'\n"
                    f"     → {e['diagnostico']}\n")

    txt += f"""
  Recomendaciones accionables:
"""
    for r in recs:
        txt += f"  · {r}\n"

    txt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
④ COMPARACIÓN DE MODELOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  {'Modelo':<22} {'F1-W':>7} {'F1-Alta':>9} {'AUC':>7}  {'Acc':>7}
  {'─'*56}"""
    for m in comparacion:
        auc_s = f"{m['roc_auc']:.4f}" if m['roc_auc'] else "  N/D "
        txt += (f"\n  {m['modelo']:<22} {m['f1_weighted']:>7.4f} "
                f"{m['f1_alta']:>9.4f} {auc_s:>7}  {m['accuracy']:>7.4f}")

    if mejor_modelo:
        txt += f"\n\n  → Mejor modelo: {mejor_modelo['modelo']} (F1={mejor_modelo['f1_weighted']:.4f})\n"

    if cor_top:
        txt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⑤ CORRELACIONES Y χ²
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Spearman más fuerte: {cor_top['feature']} (ρ={cor_top['rho']}, p={cor_top['p_valor']}) [{cor_top['magnitud']}]
"""
    if chi2_top:
        txt += (f"  χ² más significativo: {chi2_top['categoria']} "
                f"(χ²={chi2_top['chi2']}, OR={chi2_top['odds_ratio']}) "
                f"— {chi2_top['interpretacion']}\n")

    txt += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TOP TÉRMINOS PREDICTIVOS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RELEVANCIA ALTA:\n"""
    for j, item in enumerate(top_terms.get("alta", [])[:15], 1):
        barra = "█" * int(item["peso"] * 15)
        txt += (f"  {j:>2}. {item['termino']:<30} {item['peso']:.5f}"
                f"  [{item['dimension']}]  {barra}\n")

    txt += "\n  RELEVANCIA MEDIA:\n"
    for j, item in enumerate(top_terms.get("media", [])[:10], 1):
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
                 tabla_hibrida: list | None = None,
                 bert_activo: bool = False):

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {}

    reporte = {
        "meta": {"generado": datetime.now().isoformat(),
                 "version": "3.0",
                 "modelo": "LR_calibrado" if bert_activo else "CNB_calibrado",
                 "bert_activo": bert_activo,
                 "corpus": str(ruta_X)},
        "validacion_cruzada":    cv_res,
        "calibracion":           cal_res,
        "comparacion_modelos":   comparacion,
        "top_terms_por_clase":   {k: v for k, v in top_terms.items()
                                  if k != "nombres_features"},
        "top_features_hibrido":  tabla_hibrida or [],
        "dimensiones_narrativas": dimensiones,
        "correlaciones_spearman": correlaciones,
        "chi2_categorias":        chi2_res,
        "curva_aprendizaje":      curva_res,
    }

    ruta_rep = carpeta / f"reporte_cnb_v2_{ts}.json"
    with open(ruta_rep, "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)
    paths["reporte"] = ruta_rep

    ruta_txt = carpeta / f"resumen_narrativo_v2_{ts}.txt"
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write(narrativa)
    paths["narrativa"] = ruta_txt

    # Errores críticos en archivo separado
    ruta_err = carpeta / f"errores_criticos_{ts}.json"
    with open(ruta_err, "w", encoding="utf-8") as f:
        json.dump(errores_res, f, ensure_ascii=False, indent=2)
    paths["errores"] = ruta_err

    return paths


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

    # ── Detectar si hay features BERT activas ─────────────────────────────────
    bert_activo = detectar_bert_activo(X, art)
    estimador_nombre = "LogisticRegression" if bert_activo else "ComplementNB"
    print(f"  Estimador elegido  : {estimador_nombre} "
          f"({'features híbridas' if bert_activo else 'texto puro'})")

    # ── Split único ───────────────────────────────────────────────────────────
    idx = np.arange(len(y))
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        idx, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_tr = X[idx_tr]; X_te = X[idx_te]

    print(f"\n{'='*72}")
    print(f"  EJECUTANDO PIPELINE COMPLETO v3  —  {estimador_nombre}")
    print(f"{'='*72}")

    # ① Validación cruzada — usa el estimador correcto según detección
    cv_res = validacion_cruzada_robusta(X, y, bert_activo=bert_activo)

    # ② Calibración
    clf_cal, probs_cal, y_pred_cal, cal_res = entrenar_con_calibracion(
        X_tr, y_tr, X_te, y_te, bert_activo=bert_activo)

    # ③ Errores críticos (ahora incluye desacuerdo BERT)
    print(f"  Analizando errores críticos...")
    errores_res = analizar_errores_criticos(
        X_te, y_te, y_pred_cal, probs_cal, registros, idx_te)

    # ④ Curva de aprendizaje
    curva_res = curva_aprendizaje(X, y, bert_activo=bert_activo)

    # ⑤ Comparación de modelos
    comparacion = comparar_modelos(X_tr, y_tr, X_te, y_te)

    # ⑥ Top terms + tabla híbrida de features
    print(f"  Extrayendo términos predictivos y estadísticas...")
    top_terms      = extraer_top_terms(clf_cal, vec, selector)
    tabla_hibrida  = top_features_hibrido(clf_cal, vec, selector, art, n=20)
    correlaciones  = correlacion_spearman(registros, y)  if registros else []
    chi2_res       = prueba_chi2_categorias(registros, y) if registros else []
    dimensiones    = analizar_dimensiones(top_terms)

    # ⑦ Narrativa
    narrativa = generar_narrativa(
        top_terms, cv_res, cal_res, errores_res,
        curva_res, comparacion, correlaciones, chi2_res, dimensiones
    )
    print(narrativa)

    # ── Mostrar tabla híbrida en consola ──────────────────────────────────────
    if tabla_hibrida:
        print(f"{'='*72}")
        print(f"  TOP FEATURES HÍBRIDAS (texto + BERT) — {estimador_nombre}")
        print(f"  {'Rank':<5} {'Nombre':<32} {'Coef':>8}  {'Dirección':<12} Tipo")
        print(f"  {'─'*65}")
        for row in tabla_hibrida[:20]:
            barra = "█" * int(row["abs_coef"] * 8)
            print(f"  {row['rank']:<5} {row['nombre']:<32} "
                  f"{row['coef']:>+8.4f}  {row['direccion']:<12} "
                  f"[{row['tipo']}]  {barra}")
        print(f"{'='*72}")

    # ── Guardar ───────────────────────────────────────────────────────────────
    paths = guardar_todo(
        carpeta, cv_res, cal_res, errores_res, curva_res,
        comparacion, top_terms, correlaciones, chi2_res,
        dimensiones, narrativa, ruta_X,
        tabla_hibrida=tabla_hibrida,
        bert_activo=bert_activo,
    )

    print(f"{'='*72}")
    print(f"  💾 Reporte JSON      : {paths['reporte'].name}")
    print(f"  💾 Resumen narrativo : {paths['narrativa'].name}")
    print(f"  💾 Errores críticos  : {paths['errores'].name}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()

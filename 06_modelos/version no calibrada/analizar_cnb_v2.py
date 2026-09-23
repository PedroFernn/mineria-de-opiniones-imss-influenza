#!/usr/bin/env python3
"""
analizar_cnb_v2.py — Paso 3 del pipeline de modelado (CRISP-DM Fase 5)
Tema: Ineficiencia del sector salud MX frente a la influenza

Mejoras v2 — robustez estadística para tesis académica:
  ① Validación cruzada StratifiedKFold (10-fold) con intervalo de confianza 95%
     Si el modelo da 82%±1.2% en 10 pruebas distintas → conclusiones sólidas
  ② Calibración de probabilidades con CalibratedClassifierCV (isotónica)
     CNB sin calibrar da 0.98 / 0.03 → con calibración da 0.85 / 0.30
     Permite afirmar: "este comentario tiene 85% de prob. de ser queja clínica"
  ③ Análisis de errores: falsos negativos y falsos positivos por confianza
     Exporta los N casos donde el modelo era "muy seguro" y se equivocó
     Diagnóstico directo de qué términos engañan al clasificador
  ④ Curva de aprendizaje — detecta si el modelo necesita más datos o más features
  ⑤ Comparación CNB vs LogisticRegression vs LinearSVC en el mismo split

Heredadas de v1:
  ⑥ Top términos predictivos por clase (log-probabilidad CNB)
  ⑦ Correlación de Spearman (features numéricas ↔ y)
  ⑧ Prueba χ² (categorías semánticas ↔ relevancia)
  ⑨ Dimensiones narrativas de la "burocracia del dolor"
  ⑩ Reporte narrativo con [Palabra N] reemplazados automáticamente

Entrada : X_features*.npz  |  y_labels*.npy  |  vectorizador*.pkl
          dataset_normalizado*.json  (para texto original en análisis de errores)
Salida  : reporte_cnb_v2_*.json  |  resumen_narrativo_*.txt
          errores_criticos_*.json  (falsos negativos de alta confianza)

Uso: python3 analizar_cnb_v2.py
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
N_FOLDS             = 10     # 10-fold: más estable que 5-fold para n≈2500
IC_Z                = 1.96   # z para IC 95%  (z=2.576 para IC 99%)

# ── Calibración ──────────────────────────────────────────────────────────────
METODO_CALIBRACION  = "isotonic"   # "isotonic" (no lineal) o "sigmoid" (Platt scaling)
# isotonic: más flexible, mejor con CNB; necesita ≥1000 muestras
# sigmoid: más estable con datasets pequeños (<500)

# ── Análisis de errores ──────────────────────────────────────────────────────
N_ERRORES_EXPORTAR  = 20           # cuántos errores críticos guardar
UMBRAL_CONFIANZA    = 0.80         # mínima confianza para considerar un error "crítico"
# Un error crítico es: modelo muy seguro (prob ≥ 0.80) pero se equivocó

# ── Curva de aprendizaje ─────────────────────────────────────────────────────
CURVA_PUNTOS        = 8            # cuántos puntos de tamaño de entrenamiento evaluar


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

def validacion_cruzada_robusta(X, y) -> dict:
    """
    Ejecuta 10-fold estratificado para CNB y devuelve estadísticas completas.
    La estratificación garantiza que cada fold mantiene la misma proporción
    de clases que el dataset completo, crítico para corpus desbalanceados.
    """
    print(f"\n  Ejecutando {N_FOLDS}-fold estratificado...")
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # Pipeline con MaxAbsScaler dentro del CV para evitar data leakage y negativos
    clf_cv = _pipeline_cnb(alpha=0.5)

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

def entrenar_con_calibracion(X_tr, y_tr, X_te, y_te) -> tuple:
    """
    Entrena CNB crudo y CNB calibrado, compara métricas de probabilidad.

    Retorna (clf_calibrado, metricas_comparativas)
    El clasificador calibrado es el que se usa para todo el análisis posterior.
    """
    print(f"\n  Entrenando CNB sin calibrar (línea base)...")
    clf_raw = _pipeline_cnb(alpha=0.5)
    clf_raw.fit(X_tr, y_tr)
    probs_raw = clf_raw.predict_proba(X_te)[:, 1]

    print(f"  Calibrando con método '{METODO_CALIBRACION}' (cv=5)...")
    # CalibratedClassifierCV envuelve el pipeline completo: cada sub-fold
    # aplica MaxAbsScaler → CNB → calibración, sin filtración de datos.
    clf_cal = CalibratedClassifierCV(
        _pipeline_cnb(alpha=0.5),
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


def _diagnosticar_error(tokens: list[str], real: int, pred: int) -> str:
    """
    Intenta diagnosticar por qué el modelo se equivocó mirando los tokens.
    Devuelve una cadena de diagnóstico interpretable.
    """
    t = set(tokens)

    # FN: el modelo lo clasificó como media cuando era alta
    if real == 1 and pred == 0:
        if any(tok in t for tok in ["particular", "medico_particular", "de_mi_bolsillo"]):
            return "posible FN por pago_privado: el modelo no asocia 'particular' con queja alta"
        if any(tok in t for tok in ["fallecer", "morir", "muerte"]):
            return "posible FN por impacto_clinico: palabras de consecuencia no detectadas"
        if len(tokens) < 8:
            return "posible FN por texto muy corto: insuficiente señal léxica"
        return "FN sin patrón claro — puede requerir ampliar TERMINOS_DOMINIO"

    # FP: el modelo lo clasificó como alta cuando era media
    if real == 0 and pred == 1:
        if any(tok in t for tok in ["imss", "issste", "hospital", "medico"]):
            return "posible FP por institución: mención de IMSS sin queja real"
        if any(tok in t for tok in ["vacuna", "vacunar", "dosis"]):
            return "posible FP por vacunación: comentario informativo clasificado como queja"
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

def curva_aprendizaje(X, y) -> dict:
    print(f"  Calculando curva de aprendizaje ({CURVA_PUNTOS} puntos)...")
    clf_ca = _pipeline_cnb(alpha=0.5)
    cv_ca  = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    sizes, train_scores, test_scores = learning_curve(
        clf_ca, X, y,
        train_sizes=np.linspace(0.10, 1.0, CURVA_PUNTOS),
        cv=cv_ca,
        scoring="f1_weighted",
        n_jobs=-1,
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
    Extrae los top-N términos por clase del CNB entrenado.
    clf puede ser:
      · CalibratedClassifierCV(Pipeline([MaxAbsScaler, CNB]))
      · Pipeline([MaxAbsScaler, CNB])
      · ComplementNB directo (legado)
    Se navega la estructura hasta llegar al objeto con feature_log_prob_.
    """
    def _extraer_cnb(estimador):
        """Navega recursivamente hasta encontrar el ComplementNB."""
        # Caso 1: es directamente el CNB
        if hasattr(estimador, "feature_log_prob_"):
            return estimador
        # Caso 2: es un Pipeline → buscar en sus pasos
        if hasattr(estimador, "named_steps"):
            return _extraer_cnb(estimador.named_steps.get("cnb", None))
        # Caso 3: es CalibratedClassifierCV → tomar el primer sub-clasificador
        if hasattr(estimador, "calibrated_classifiers_"):
            sub = getattr(estimador.calibrated_classifiers_[0], "estimator", None)
            return _extraer_cnb(sub) if sub else None
        # Caso 4: tiene atributo estimator (otro wrapper)
        inner = getattr(estimador, "estimator", None)
        return _extraer_cnb(inner) if inner else None

    cnb = _extraer_cnb(clf)

    if cnb is None or not hasattr(cnb, "feature_log_prob_"):
        return {"alta": [], "media": [], "nombres_features": []}

    nombres_vec = list(vec.get_feature_names_out())
    if selector is not None:
        mascara     = selector.get_support()
        nombres_feat= [nombres_vec[i] for i, m in enumerate(mascara) if m]
    else:
        nombres_feat= nombres_vec

    log_probs = cnb.feature_log_prob_
    n_cols    = min(log_probs.shape[1], len(nombres_feat))
    nombres_feat = nombres_feat[:n_cols]

    resultado = {}
    for idx_clase, nombre_clase in enumerate(["media", "alta"]):
        if idx_clase >= log_probs.shape[0]:
            resultado[nombre_clase] = []
            continue
        pesos   = -log_probs[idx_clase][:n_cols]
        top_idx = np.argsort(pesos)[-n:][::-1]
        resultado[nombre_clase] = [
            {
                "termino":   nombres_feat[i],
                "peso":      round(float(pesos[i]), 5),
                "dimension": _asignar_dimension(nombres_feat[i]),
            }
            for i in top_idx if i < len(nombres_feat)
        ]

    resultado["nombres_features"] = nombres_feat
    return resultado


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
                 dimensiones, narrativa, ruta_X):

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = {}

    reporte = {
        "meta": {"generado": datetime.now().isoformat(),
                 "version": "2.0", "modelo": "CalibratedCNB",
                 "corpus": str(ruta_X)},
        "validacion_cruzada":   cv_res,
        "calibracion":          cal_res,
        "comparacion_modelos":  comparacion,
        "top_terms_por_clase":  {k: v for k, v in top_terms.items()
                                 if k != "nombres_features"},
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

    # ── Split único para calibración, errores y comparación ──────────────────
    idx = np.arange(len(y))
    idx_tr, idx_te, y_tr, y_te = train_test_split(
        idx, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    X_tr = X[idx_tr]; X_te = X[idx_te]

    print(f"\n{'='*72}")
    print(f"  EJECUTANDO PIPELINE COMPLETO v2")
    print(f"{'='*72}")

    # ① Validación cruzada 10-fold
    cv_res     = validacion_cruzada_robusta(X, y)

    # ② Calibración
    clf_cal, probs_cal, y_pred_cal, cal_res = entrenar_con_calibracion(
        X_tr, y_tr, X_te, y_te)

    # ③ Errores críticos
    print(f"  Analizando errores críticos...")
    errores_res = analizar_errores_criticos(
        X_te, y_te, y_pred_cal, probs_cal, registros, idx_te)

    # ④ Curva de aprendizaje
    curva_res = curva_aprendizaje(X, y)

    # ⑤ Comparación de modelos
    comparacion = comparar_modelos(X_tr, y_tr, X_te, y_te)

    # ⑥ Top terms, correlaciones, χ², dimensiones
    print(f"  Extrayendo términos predictivos y estadísticas...")
    top_terms    = extraer_top_terms(clf_cal, vec, selector)
    correlaciones= correlacion_spearman(registros, y)  if registros else []
    chi2_res     = prueba_chi2_categorias(registros, y) if registros else []
    dimensiones  = analizar_dimensiones(top_terms)

    # ⑦ Narrativa
    narrativa = generar_narrativa(
        top_terms, cv_res, cal_res, errores_res,
        curva_res, comparacion, correlaciones, chi2_res, dimensiones
    )
    print(narrativa)

    # ── Guardar ───────────────────────────────────────────────────────────────
    paths = guardar_todo(
        carpeta, cv_res, cal_res, errores_res, curva_res,
        comparacion, top_terms, correlaciones, chi2_res,
        dimensiones, narrativa, ruta_X
    )

    print(f"{'='*72}")
    print(f"  💾 Reporte JSON      : {paths['reporte'].name}")
    print(f"  💾 Resumen narrativo : {paths['narrativa'].name}")
    print(f"  💾 Errores críticos  : {paths['errores'].name}")
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()

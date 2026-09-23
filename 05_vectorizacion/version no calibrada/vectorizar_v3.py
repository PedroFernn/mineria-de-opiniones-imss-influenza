#!/usr/bin/env python3
"""
vectorizar_v2.py — Paso 2 del pipeline de modelado (CRISP-DM Fase 4)
Tema: Ineficiencia del sector salud MX frente a la influenza


Estrategias disponibles:
  A) TF-IDF estándar       — clásico, interpretable, rápido
  B) TF-IDF ponderado      — multiplica filas por peso_reddit
  C) CountVectorizer        — frecuencias crudas, línea base
  D) BERT sentence embed.  — 768 dims contextuales (requiere GPU/CPU lento)
  E) BERT + TF-IDF híbrido — concatena ambos (máxima cobertura)

Entrada : dataset_normalizado_*.json  (salida de normalizar_nlp_v2.py)
Salida  : X_features.npz  |  y_labels.npy  |  vectorizador.pkl  |  reporte

Dependencias base    : scikit-learn numpy scipy
Dependencias BERT    : sentence-transformers torch  (solo estrategias D/E)
  pip install scikit-learn numpy scipy sentence-transformers torch

Uso: python3 vectorizar_v2.py
"""

import json
import sys
import subprocess
import pickle
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GENERAL
# ══════════════════════════════════════════════════════════════════════════════

# ── TF-IDF / Count ────────────────────────────────────────────────────────────
MAX_FEATURES   = 5_000
NGRAM_RANGE    = (1, 2)
MIN_DF         = 2
SUBLINEAR_TF   = True
# MAX_DF se calcula dinámicamente (mejora ④B); este valor es el fallback si falla
MAX_DF_FALLBACK = 0.85

# ── SelectKBest χ² ────────────────────────────────────────────────────────────
APLICAR_SELECCION = True   # activar/desactivar SelectKBest
K_FEATURES        = 3_000  # cuántas features conservar tras χ² (None = todas)

# ── BERT ──────────────────────────────────────────────────────────────────────
MODELO_BERT   = "hiiamsid/sentence_similarity_spanish_es"
# Alternativas (de mejor a más ligero):
#   "PlanTL-GOB-ES/roberta-base-bne"  ← RoBERTa entrenado con datos es-MX
#   "dccuchile/bert-base-spanish-wwm-cased"  ← BETO original
#   "paraphrase-multilingual-MiniLM-L12-v2"  ← más rápido, 384 dims
BERT_BATCH    = 32    # comentarios por lote (ajustar según RAM/VRAM)

# ── Features numéricas del dataset enriquecido ───────────────────────────────
FEATURES_NUMERICAS = [
    "polaridad_score",
    "ifb_score",
    "impacto_escala",
    "num_palabras",
    "peso_reddit",
    # score_queja_salud se añade dinámicamente en el código (mejora ②)
]

# ── Etiqueta objetivo ─────────────────────────────────────────────────────────
COLUMNA_ETIQUETA = "relevancia"
MAPA_ETIQUETA    = {"alta": 1, "media": 0}


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA ② — LEXICON DE QUEJAS DE SALUD MX
#
#  Estructura: { "término": peso_negativo }
#  El score_queja_salud = suma de pesos de términos presentes en el texto.
#  Se añade como feature numérica ANTES del StandardScaler para que el modelo
#  tenga una "pista directa" de qué textos hablan de ineficiencia del sistema.
#
#  Por qué funciona mejor que TF-IDF solo:
#  TF-IDF puede darle bajo peso a "viacrucis" si es poco frecuente en el corpus,
#  aunque sea altamente informativo. El lexicon garantiza que siempre tenga peso.
# ══════════════════════════════════════════════════════════════════════════════

LEXICON_QUEJAS_SALUD: dict[str, float] = {
    # ── Burocracia y agotamiento ──────────────────────────────────────────────
    "viacrucis":             3.0,
    "vía_crucis":            3.0,
    "odisea":                2.5,
    "vuelta_y_vuelta":       3.0,
    "no_hay_sistema":        2.8,
    "sistema_caído":         2.5,
    "vuelva_manana":         2.8,
    "vuelva mañana":         2.8,
    "fuera_de_servicio":     2.0,
    "trámite":               1.5,
    "tramite":               1.5,
    "papeleo":               1.5,
    "burocracia":            2.0,
    # ── Desabasto ────────────────────────────────────────────────────────────
    "desabasto":             2.5,
    "sin_medicamento":       2.8,
    "no_hay_medicamento":    2.8,
    "no_hay_vacuna":         2.8,
    "sin_vacuna":            2.5,
    "agotado":               2.0,
    "escasez":               2.2,
    "falta_de_medicamento":  2.8,
    # ── Negligencia / maltrato ────────────────────────────────────────────────
    "negligencia":           2.8,
    "negligente":            2.8,
    "mala_atencion":         2.5,
    "no_me_atendieron":      2.5,
    "maltrato":              2.5,
    "inútil":                2.5,
    "incompetente":          2.5,
    # ── Pago privado (falla sistémica) ────────────────────────────────────────
    "particular":            2.0,
    "medico_particular":     2.2,
    "de_mi_bolsillo":        2.5,
    # ── Impacto clínico ───────────────────────────────────────────────────────
    "fallecer":              3.0,
    "morir":                 3.0,
    "muerte":                3.0,
    "empeorar":              2.5,
    "neumonía":              2.5,
    "neumonia":              2.5,
    "intubado":              2.8,
    "uci":                   2.5,
    "recaer":                2.2,
    # ── Corrupción ────────────────────────────────────────────────────────────
    "moche":                 2.8,
    "mordida":               2.8,
    "soborno":               2.8,
    "corrupción":            2.5,
    "corrupcion":            2.5,
    "palancas":              2.2,
    # ── Jerga MX de indignación ───────────────────────────────────────────────
    "desmadre":              2.5,
    "caos":                  2.0,
    "pésimo":                2.5,
    "pesimo":                2.5,
    "terrible":              2.0,
    "horrible":              2.0,
    "desastroso":            2.5,
    # ── Instituciones + problema (bigramas clave) ─────────────────────────────
    "urgencias_llenas":      2.5,
    "no_hay_cupo":           2.5,
    "sala_de_espera":        1.5,
    "no_hay":                1.8,
}


def calcular_score_queja(texto: str) -> float:
    """
    Suma los pesos del lexicon para cada término presente en el texto.
    Opera sobre texto_normalizado (ya lematizado y con n-gramas como tokens).
    Retorna float ≥ 0.0.
    """
    t = texto.lower()
    return round(sum(peso for termino, peso in LEXICON_QUEJAS_SALUD.items()
                     if termino in t), 3)


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA ④A — TOKEN PATTERN PARA SIGLAS
#
#  Problema original: r"[a-záéíóúüñ_]{2,}" convierte todo a minúsculas y
#  excluye tokens con mayúsculas o números (IMSS, H1N1, CDMX).
#
#  Solución:
#    1. Pre-normalizar siglas en el texto ANTES de pasar al vectorizador:
#       reemplazar siglas conocidas por tokens genéricos en minúsculas
#       (ej. IMSS → entidad_salud_imss) para que el patrón las capture.
#    2. Ampliar el patrón para aceptar letras mayúsculas y números en siglas.
#
#  Siglas normalizadas → token estandarizado: el modelo aprende un concepto
#  único ("entidad_salud_imss") en lugar de variantes ("IMSS", "imss", "Imss").
# ══════════════════════════════════════════════════════════════════════════════

SIGLAS_NORMALIZACION: dict[str, str] = {
    r"\bIMSS\b":        "entidad_salud_imss",
    r"\bISSSTe?\b":     "entidad_salud_issste",
    r"\bINSABI\b":      "entidad_salud_insabi",
    r"\bSSA\b":         "entidad_salud_ssa",
    r"\bCDMX\b":        "lugar_cdmx",
    r"\bH1N1\b":        "virus_h1n1",
    r"\bH3N2\b":        "virus_h3n2",
    r"\bUCI\b":         "unidad_cuidados_intensivos",
    r"\bUMF\b":         "unidad_medicina_familiar",
    r"\bOMS\b":         "organizacion_mundial_salud",
    r"\bSSEP\b":        "secretaria_salud_estatal",
}

_SIGLAS_COMPILADAS = [
    (re.compile(patron, re.IGNORECASE), reemplazo)
    for patron, reemplazo in SIGLAS_NORMALIZACION.items()
]

# Token pattern ampliado: acepta letras, guión bajo y dígitos (para h1n1 etc.)
TOKEN_PATTERN = r"[a-záéíóúüñ][a-záéíóúüñ0-9_]{1,}"


def normalizar_siglas(texto: str) -> str:
    """Reemplaza siglas conocidas por sus tokens normalizados en minúsculas."""
    for patron, reemplazo in _SIGLAS_COMPILADAS:
        texto = patron.sub(reemplazo, texto)
    return texto


def preparar_corpus_texto(corpus_raw: list[str]) -> list[str]:
    """
    Aplica normalización de siglas a todo el corpus antes de vectorizar.
    Retorna lista nueva; el original se conserva para BERT (que trabaja
    mejor con texto natural que con siglas reemplazadas).
    """
    return [normalizar_siglas(t) for t in corpus_raw]


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA ④B — MAX_DF DINÁMICO
#
#  Problema: MAX_DF=0.90 fijo puede conservar términos muy frecuentes
#  en corpus de dominio específico (ej. "salud" aparece en 95% de posts).
#
#  Solución: Analizar la distribución de frecuencias del corpus ANTES de
#  vectorizar y calcular el umbral óptimo buscando la "rodilla" de la curva,
#  es decir, el punto donde la frecuencia deja de aportar discriminación.
#
#  Metodología:
#    - Se cuenta en cuántos documentos aparece cada término del corpus.
#    - Se calcula el percentil 85 de esas frecuencias relativas.
#    - El umbral resultante es el max_df dinámico.
#    - Se aplica un techo de 0.95 y un piso de 0.60 para evitar extremos.
# ══════════════════════════════════════════════════════════════════════════════

def calcular_max_df_dinamico(corpus: list[str], percentil: int = 85) -> float:
    """
    Calcula un max_df adaptado al corpus actual.

    Parámetros:
        corpus     — lista de textos normalizados
        percentil  — percentil de la distribución de frecuencias (default 85)

    Retorna float entre 0.60 y 0.95.
    """
    n_docs      = len(corpus)
    conteo_docs: Counter = Counter()

    for doc in corpus:
        # Contar en cuántos documentos únicos aparece cada término
        tokens_unicos = set(doc.lower().split())
        conteo_docs.update(tokens_unicos)

    if not conteo_docs:
        return MAX_DF_FALLBACK

    # Frecuencias relativas (fracción de documentos que contienen cada término)
    frecuencias_relativas = [cnt / n_docs for cnt in conteo_docs.values()]
    umbral_raw = float(np.percentile(frecuencias_relativas, percentil))

    # Aplicar techo y piso
    umbral = max(0.60, min(0.95, umbral_raw))

    print(f"  max_df dinámico    : {umbral:.3f}  "
          f"(percentil {percentil} de {len(conteo_docs):,} términos únicos)")
    return round(umbral, 3)


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA ③ — SelectKBest con χ²
#
#  χ² mide la dependencia estadística entre cada feature (término) y la
#  etiqueta de clase. Features con χ² bajo son ruido estadístico y se eliminan.
#
#  Por qué después de vectorizar y no antes:
#  SelectKBest necesita la matriz X ya construida para calcular las estadísticas.
#  Se aplica SOLO sobre las features de texto (no las numéricas) para no eliminar
#  señales del lexicon o del IFB que son deterministas por construcción.
#
#  Nota importante: χ² requiere X ≥ 0. TF-IDF y CountVectorizer lo garantizan.
#  Si se usan BERT embeddings (que pueden ser negativos), usar f_classif en su lugar.
# ══════════════════════════════════════════════════════════════════════════════

def aplicar_seleccion_features(X_texto, y: list[int], k: int | None = None):
    """
    Selecciona las K features más correlacionadas con y usando χ².
    Retorna (X_seleccionado, selector) para poder transformar datos nuevos.

    Si k es None o mayor que el número de features, retorna X sin cambios.
    """
    from sklearn.feature_selection import SelectKBest, chi2

    n_features_total = X_texto.shape[1]
    k_efectivo = k or n_features_total

    if k_efectivo >= n_features_total:
        print(f"  SelectKBest        : omitido (k={k_efectivo} ≥ {n_features_total} features)")
        return X_texto, None

    selector = SelectKBest(chi2, k=k_efectivo)
    X_sel    = selector.fit_transform(X_texto, y)

    n_eliminadas = n_features_total - k_efectivo
    print(f"  SelectKBest χ²     : {n_features_total} → {k_efectivo} features "
          f"({n_eliminadas} eliminadas)")
    return X_sel, selector


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA ① — BERT SENTENCE EMBEDDINGS
#
#  Por qué BERT es mejor que TF-IDF para negación y sarcasmo:
#    TF-IDF: "No hay medicinas" y "Medicinas hay, no faltan" comparten tokens
#            → vectores similares → el modelo se confunde.
#    BERT:   Cada token atiende al contexto completo → representaciones distintas.
#
#  Modelo elegido: paraphrase-multilingual-MiniLM-L12-v2
#    · 384 dimensiones (vs 768 de BETO) — más rápido, menor memoria
#    · Entrenado en 50+ idiomas incluyendo es-MX
#    · Optimizado para similitud semántica de oraciones (ideal para clustering)
#
#  Para mayor precisión en español MX puedes cambiar MODELO_BERT a:
#    "dccuchile/bert-base-spanish-wwm-cased"  (BETO, 768 dims, más lento)
#    "PlanTL-GOB-ES/roberta-base-bne"         (RoBERTa español, estado del arte)
#
#  Importante: BERT trabaja mejor con el texto ORIGINAL (comentario limpio),
#  NO con el texto_normalizado (ya sin stopwords). Las stopwords dan contexto
#  gramatical que BERT necesita para entender la negación.
# ══════════════════════════════════════════════════════════════════════════════

def verificar_sentence_transformers() -> bool:
    """Verifica si sentence-transformers está disponible. No instala automáticamente
    porque puede tardar varios minutos (descarga ~500 MB con torch)."""
    try:
        import sentence_transformers   # noqa: F401
        return True
    except ImportError:
        return False


def generar_embeddings_bert(textos_originales: list[str]) -> np.ndarray:
    """
    Genera embeddings BERT para una lista de textos.

    Usa 'comentario' (texto limpio CON stopwords) porque BERT necesita
    contexto gramatical completo para capturar negaciones y sarcasmo.

    Retorna numpy array de shape (n_docs, n_dims).
    """
    from sentence_transformers import SentenceTransformer

    print(f"\n  Cargando modelo BERT: {MODELO_BERT}")
    print(f"  (primera ejecución descarga ~400 MB — solo una vez)")
    modelo = SentenceTransformer(MODELO_BERT)

    print(f"  Generando embeddings para {len(textos_originales)} textos "
          f"(lotes de {BERT_BATCH})...")

    embeddings = modelo.encode(
        textos_originales,
        batch_size      = BERT_BATCH,
        show_progress_bar = True,
        convert_to_numpy  = True,
        normalize_embeddings = True,   # L2-norm → cosine similarity = dot product
    )
    print(f"  Embeddings BERT    : shape {embeddings.shape}")
    return embeddings.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  INSTALACIÓN Y CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

DEPENDENCIAS_BASE = {
    "sklearn": "scikit-learn",
    "numpy":   "numpy",
    "scipy":   "scipy",
}
DEPENDENCIAS_BERT = {
    "sentence_transformers": "sentence-transformers",
    "torch": "torch",
}


def verificar_e_instalar(libs: dict[str, str], preguntar: bool = True) -> bool:
    faltantes = [pip for mod, pip in libs.items()
                 if not _importable(mod)]
    if not faltantes:
        return True
    print(f"\n  ⚠️  Librerías no encontradas: {', '.join(faltantes)}")
    if preguntar:
        resp = input("  ¿Instalarlas ahora? (s/n): ").strip().lower()
        if resp != "s":
            return False
    subprocess.run(
        [sys.executable, "-m", "pip", "install"] + faltantes + ["-q"],
        check=True
    )
    return True


def _importable(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def pedir_archivo() -> Path:
    carpeta = Path(__file__).parent
    jsons   = sorted(carpeta.glob("*.json"))
    if not jsons:
        raise FileNotFoundError("No hay archivos .json en la carpeta.")

    print(f"\n{'='*70}")
    print("  VECTORIZACIÓN v2 — PIPELINE DE MODELADO  📐🤖")
    print(f"{'='*70}\n")
    print("  Archivos .json disponibles:")
    for i, p in enumerate(jsons):
        print(f"    [{i}] {p.name}  ({p.stat().st_size/1024:.0f} KB)")

    while True:
        idx = input("\n  Número del archivo a vectorizar: ").strip()
        try:
            return jsons[int(idx)]
        except (ValueError, IndexError):
            print(f"  ❌ Elige entre 0 y {len(jsons)-1}.")


def elegir_estrategia() -> str:
    bert_ok = verificar_sentence_transformers()
    print("\n  Estrategia de vectorización:")
    print("    [A] TF-IDF estándar          — rápido, interpretable")
    print("    [B] TF-IDF ponderado Reddit  — features × peso_reddit")
    print("    [C] CountVectorizer          — frecuencias crudas (línea base)")
    print(f"    [D] BERT embeddings          — contextual, captura negación "
          f"{'✅' if bert_ok else '⚠️  requiere: pip install sentence-transformers torch'}")
    print(f"    [E] BERT + TF-IDF híbrido    — máxima cobertura "
          f"{'✅' if bert_ok else '⚠️  requiere BERT'}")

    while True:
        op = input("\n  Opción [A/B/C/D/E]: ").strip().upper()
        if op in ("A", "B", "C"):
            return op
        if op in ("D", "E"):
            if not bert_ok:
                print("  Instalando dependencias BERT...")
                if not verificar_e_instalar(DEPENDENCIAS_BERT):
                    print("  ❌ No se pudieron instalar. Elige A, B o C.")
                    continue
            return op
        print("  ❌ Escribe A, B, C, D o E.")


def cargar_registros(ruta: Path) -> tuple[list[dict], dict]:
    with open(ruta, "r", encoding="utf-8") as f:
        data = json.load(f)
    registros     = data.get("datos", data) if isinstance(data, dict) else data
    meta_anterior = data.get("meta_normalizacion", {}) if isinstance(data, dict) else {}
    return registros, meta_anterior


def extraer_corpus(registros: list[dict]) -> tuple[
    list[str],   # texto_normalizado (para TF-IDF)
    list[str],   # comentario limpio (para BERT)
    list[int],   # etiquetas y
    list[float], # pesos reddit
    list[dict],  # features numéricas raw
]:
    corpus_norm:  list[str]   = []
    corpus_orig:  list[str]   = []
    etiquetas:    list[int]   = []
    pesos:        list[float] = []
    feat_num_raw: list[dict]  = []
    omitidos = 0

    for reg in registros:
        texto_norm = reg.get("texto_normalizado", "").strip()
        texto_orig = reg.get("comentario", reg.get("comentario_raw", "")).strip()
        label      = MAPA_ETIQUETA.get(reg.get(COLUMNA_ETIQUETA, ""))

        if not texto_norm or label is None:
            omitidos += 1
            continue

        corpus_norm.append(texto_norm)
        corpus_orig.append(texto_orig or texto_norm)
        etiquetas.append(label)
        pesos.append(float(reg.get("peso_reddit", 1.0)))

        # Features numéricas base + score_queja_salud (mejora ②)
        nums = {col: float(reg.get(col) or 0.0) for col in FEATURES_NUMERICAS}
        nums["score_queja_salud"] = calcular_score_queja(texto_norm)
        feat_num_raw.append(nums)

    if omitidos:
        print(f"  ⚠️  {omitidos} registros omitidos (sin texto o etiqueta).")

    return corpus_norm, corpus_orig, etiquetas, pesos, feat_num_raw


# ══════════════════════════════════════════════════════════════════════════════
#  VECTORIZACIÓN DE TEXTO (TF-IDF / Count)
# ══════════════════════════════════════════════════════════════════════════════

def construir_vectorizador(estrategia: str, max_df: float):
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

    params = dict(
        max_features  = MAX_FEATURES,
        ngram_range   = NGRAM_RANGE,
        min_df        = MIN_DF,
        max_df        = max_df,
        token_pattern = TOKEN_PATTERN,   # ④A: acepta siglas y guión bajo
        analyzer      = "word",
    )

    if estrategia in ("A", "B"):
        return TfidfVectorizer(**params, sublinear_tf=SUBLINEAR_TF,
                               use_idf=True, smooth_idf=True, norm="l2")
    return CountVectorizer(**params)


def vectorizar_texto(corpus: list[str], pesos: list[float],
                     estrategia: str, max_df: float):
    from scipy.sparse import diags

    # ④A: normalizar siglas antes de pasar al vectorizador
    corpus_prep = preparar_corpus_texto(corpus)

    vec    = construir_vectorizador(estrategia, max_df)
    X_txt  = vec.fit_transform(corpus_prep)
    print(f"  Matriz de texto    : {X_txt.shape[0]} docs × {X_txt.shape[1]} features")

    if estrategia == "B":
        X_txt = diags(pesos).dot(X_txt)
        print(f"  Ponderación Reddit : ✅ aplicada")

    return vec, X_txt


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURES NUMÉRICAS
# ══════════════════════════════════════════════════════════════════════════════

def escalar_features_numericas(feat_num_raw: list[dict]):
    from sklearn.preprocessing import StandardScaler

    cols = FEATURES_NUMERICAS + ["score_queja_salud"]   # ② incluida
    mat  = np.array([[fila.get(c, 0.0) for c in cols]
                     for fila in feat_num_raw], dtype=np.float32)

    scaler = StandardScaler()
    X_num  = scaler.fit_transform(mat)
    print(f"  Features numéricas : {X_num.shape[1]} columnas "
          f"(incluye score_queja_salud ✅)")
    return X_num, scaler, cols


def combinar_matrices(partes: list, nombres: list[str]) -> tuple:
    """
    Concatena horizontalmente matrices sparse y/o densas.
    Convierte todo a sparse CSR para uniformidad.
    """
    from scipy.sparse import hstack, csr_matrix, issparse

    partes_sparse = [p if issparse(p) else csr_matrix(p) for p in partes]
    X_final = hstack(partes_sparse, format="csr")
    print(f"  Matriz combinada   : {X_final.shape[0]} docs × {X_final.shape[1]} features")
    print(f"  Bloques            : {' + '.join(nombres)}")
    return X_final


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS INTERPRETABLE
# ══════════════════════════════════════════════════════════════════════════════

def top_features_por_clase(vec, X_texto, y: list[int], n: int = 15) -> dict:
    nombres = vec.get_feature_names_out()
    y_arr   = np.array(y)
    result  = {}

    for clase, etq in [(1, "alta"), (0, "media")]:
        mask  = y_arr == clase
        if not mask.any():
            continue
        media    = np.asarray(X_texto[mask].mean(axis=0)).flatten()
        top_idx  = media.argsort()[-n:][::-1]
        result[etq] = [
            {"termino": nombres[i], "peso_medio": round(float(media[i]), 5)}
            for i in top_idx
        ]
    return result


def distribucion_score_queja(feat_num_raw: list[dict], etiquetas: list[int]) -> dict:
    """Estadísticas comparativas del score_queja_salud por clase."""
    scores_alta  = [r["score_queja_salud"] for r, y in
                    zip(feat_num_raw, etiquetas) if y == 1]
    scores_media = [r["score_queja_salud"] for r, y in
                    zip(feat_num_raw, etiquetas) if y == 0]

    def stats(lst):
        if not lst:
            return {}
        arr = np.array(lst)
        return {"media": round(float(arr.mean()), 3),
                "mediana": round(float(np.median(arr)), 3),
                "max": round(float(arr.max()), 3),
                "pct_cero": round((arr == 0).mean() * 100, 1)}

    return {"alta": stats(scores_alta), "media": stats(scores_media)}


# ══════════════════════════════════════════════════════════════════════════════
#  GUARDADO
# ══════════════════════════════════════════════════════════════════════════════

def guardar_artefactos(X_final, y, vec, scaler, selector,
                       cols_num, top_feat, dist_queja,
                       meta_ant, estrategia, ruta_entrada, carpeta):
    from scipy.sparse import save_npz

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    ruta_X  = carpeta / f"X_features_v2_{ts}.npz"
    ruta_y  = carpeta / f"y_labels_v2_{ts}.npy"
    ruta_p  = carpeta / f"vectorizador_v2_{ts}.pkl"
    ruta_r  = carpeta / f"reporte_vectorizacion_v2_{ts}.json"

    save_npz(str(ruta_X), X_final)
    np.save(str(ruta_y), np.array(y, dtype=np.int8))

    with open(ruta_p, "wb") as f:
        pickle.dump({
            "vectorizador":        vec,
            "scaler":              scaler,
            "selector":            selector,
            "estrategia":          estrategia,
            "features_numericas":  cols_num,
            "mapa_etiqueta":       MAPA_ETIQUETA,
            "token_pattern":       TOKEN_PATTERN,
            "siglas_normalizacion": SIGLAS_NORMALIZACION,
        }, f)

    reporte = {
        "meta_vectorizacion": {
            "generado":           datetime.now().isoformat(),
            "version_script":     "2.0",
            "archivo_fuente":     str(ruta_entrada),
            "pipeline_anterior":  meta_ant.get("version_script", "desconocido"),
            "estrategia":         estrategia,
            "dimensiones_X":      list(X_final.shape),
            "clases_y":           {etq: int((np.array(y) == v).sum())
                                   for etq, v in MAPA_ETIQUETA.items()},
            "mejoras_activas": {
                "bert_embeddings":       estrategia in ("D", "E"),
                "lexicon_quejas_salud":  True,
                "select_k_best_chi2":    selector is not None,
                "siglas_normalizadas":   True,
                "max_df_dinamico":       True,
            },
            "config": {
                "max_features":    MAX_FEATURES,
                "ngram_range":     list(NGRAM_RANGE),
                "min_df":          MIN_DF,
                "sublinear_tf":    SUBLINEAR_TF,
                "k_features_chi2": K_FEATURES,
                "bert_modelo":     MODELO_BERT if estrategia in ("D","E") else None,
                "features_num":    cols_num,
            },
            "artefactos": {
                "X_sparse":    str(ruta_X),
                "y_labels":    str(ruta_y),
                "vectorizador": str(ruta_p),
            },
        },
        "top_features_por_clase":     top_feat,
        "score_queja_salud_por_clase": dist_queja,
    }

    with open(ruta_r, "w", encoding="utf-8") as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)

    return ruta_X, ruta_y, ruta_p, ruta_r


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    verificar_e_instalar(DEPENDENCIAS_BASE)

    ruta_entrada = pedir_archivo()
    estrategia   = elegir_estrategia()
    carpeta      = ruta_entrada.parent

    # ── Carga ─────────────────────────────────────────────────────────────────
    print(f"\n  Cargando: {ruta_entrada.name} ...")
    registros, meta_anterior = cargar_registros(ruta_entrada)

    corpus_norm, corpus_orig, etiquetas, pesos, feat_num_raw = extraer_corpus(registros)
    n = len(corpus_norm)
    dist = {etq: etiquetas.count(v) for etq, v in MAPA_ETIQUETA.items()}

    print(f"  Documentos válidos : {n}")
    print(f"  Distribución y     : {dist}")

    if n < 10:
        print("  ❌ Corpus demasiado pequeño.")
        sys.exit(1)

    print(f"\n{'='*70}")
    estrategia_nombre = {
        "A": "TF-IDF estándar",
        "B": "TF-IDF ponderado Reddit",
        "C": "CountVectorizer",
        "D": "BERT embeddings",
        "E": "BERT + TF-IDF híbrido",
    }
    print(f"  CONSTRUYENDO MATRIZ X — {estrategia_nombre[estrategia]}")
    print(f"{'='*70}")

    # ── ④B max_df dinámico ────────────────────────────────────────────────────
    max_df = calcular_max_df_dinamico(corpus_norm)

    # ── Vectorización de texto ────────────────────────────────────────────────
    vec       = None
    X_texto   = None
    X_bert    = None
    selector  = None

    if estrategia in ("A", "B", "C", "E"):
        vec, X_texto = vectorizar_texto(corpus_norm, pesos, estrategia, max_df)

        # ── ③ SelectKBest χ² ──────────────────────────────────────────────────
        if APLICAR_SELECCION and estrategia != "C":
            X_texto, selector = aplicar_seleccion_features(
                X_texto, etiquetas, K_FEATURES
            )

    if estrategia in ("D", "E"):
        # BERT usa corpus_orig (texto con stopwords para contexto gramatical)
        X_bert = generar_embeddings_bert(corpus_orig)

    # ── ② Features numéricas (incluye score_queja_salud) ─────────────────────
    X_num, scaler, cols_num = escalar_features_numericas(feat_num_raw)

    # ── Combinar bloques ──────────────────────────────────────────────────────
    bloques: list = []
    nombres: list[str] = []

    if X_texto is not None:
        bloques.append(X_texto)
        nombres.append(f"TF-IDF({X_texto.shape[1]})")

    if X_bert is not None:
        bloques.append(X_bert)
        nombres.append(f"BERT({X_bert.shape[1]})")

    bloques.append(X_num)
    nombres.append(f"NUM({X_num.shape[1]})")

    from scipy.sparse import csr_matrix, issparse
    partes_sparse = [b if issparse(b) else csr_matrix(b) for b in bloques]
    from scipy.sparse import hstack
    X_final = hstack(partes_sparse, format="csr")
    print(f"  Matriz combinada   : {X_final.shape[0]} × {X_final.shape[1]}")
    print(f"  Bloques            : {' + '.join(nombres)}")

    # ── Análisis interpretable ────────────────────────────────────────────────
    top_feat   = top_features_por_clase(vec, X_texto, etiquetas) if vec else {}
    dist_queja = distribucion_score_queja(feat_num_raw, etiquetas)

    # ── Guardado ──────────────────────────────────────────────────────────────
    ruta_X, ruta_y, ruta_p, ruta_r = guardar_artefactos(
        X_final, etiquetas, vec, scaler, selector,
        cols_num, top_feat, dist_queja,
        meta_anterior, estrategia, ruta_entrada, carpeta
    )

    # ── Reporte consola ───────────────────────────────────────────────────────
    n_docs, n_feat = X_final.shape
    sparsity = round(1 - X_final.nnz / (n_docs * n_feat), 4)

    print(f"\n{'='*70}")
    print("  REPORTE DE VECTORIZACIÓN v2")
    print(f"{'='*70}")
    print(f"  Estrategia                     : {estrategia_nombre[estrategia]}")
    print(f"  Dimensiones X (docs × features): {n_docs} × {n_feat}")
    print(f"  Sparsity                       : {sparsity*100:.1f}%")
    print(f"  Distribución de clases:")
    for etq, cnt in dist.items():
        pct   = round(cnt / n_docs * 100, 1)
        barra = "█" * int(pct / 3)
        print(f"    {etq:<10} {cnt:>5} ({pct:>5}%)  {barra}")

    print(f"\n  📊 score_queja_salud por clase (mejora ②):")
    for clase, st in dist_queja.items():
        if st:
            print(f"    {clase:<10} media={st['media']:<6}  "
                  f"mediana={st['mediana']:<6}  "
                  f"sin_queja={st['pct_cero']}%")

    if top_feat:
        print(f"\n  📊 Top 8 términos — Relevancia ALTA:")
        for item in top_feat.get("alta", [])[:8]:
            barra = "█" * int(item["peso_medio"] * 400)
            print(f"    {item['termino']:<30} {item['peso_medio']:.5f}  {barra}")
        print(f"\n  📊 Top 8 términos — Relevancia MEDIA:")
        for item in top_feat.get("media", [])[:8]:
            barra = "█" * int(item["peso_medio"] * 400)
            print(f"    {item['termino']:<30} {item['peso_medio']:.5f}  {barra}")

    print(f"\n  Mejoras activas:")
    print(f"    ① BERT contextual        : {'✅' if estrategia in ('D','E') else '— (usa A/B/C para activar)'}")
    print(f"    ② Lexicon quejas salud   : ✅ ({len(LEXICON_QUEJAS_SALUD)} términos ponderados)")
    print(f"    ③ SelectKBest χ²        : {'✅' if selector else '— (k >= features disponibles)'}")
    print(f"    ④A Siglas normalizadas   : ✅ ({len(SIGLAS_NORMALIZACION)} patrones)")
    print(f"    ④B max_df dinámico      : ✅ ({max_df})")

    print(f"\n  💾 Artefactos:")
    print(f"    {ruta_X.name}")
    print(f"    {ruta_y.name}")
    print(f"    {ruta_p.name}")
    print(f"    {ruta_r.name}")
    print(f"{'='*70}")

    print(f"""
  CÓMO CARGAR Y USAR:
  ──────────────────────────────────────────────────────────────────────
  from scipy.sparse import load_npz
  import numpy as np, pickle

  X = load_npz("{ruta_X.name}")
  y = np.load("{ruta_y.name}")

  with open("{ruta_p.name}", "rb") as f:
      art = pickle.load(f)

  # División estratificada
  from sklearn.model_selection import train_test_split
  X_tr, X_te, y_tr, y_te = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y)

  # Clasificadores recomendados según estrategia:
  #   A/B → LogisticRegression, LinearSVC, SGDClassifier
  #   C   → MultinomialNB, ComplementNB
  #   D/E → MLPClassifier, SVC(kernel='rbf'), XGBoost
  from sklearn.linear_model import LogisticRegression
  clf = LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced')
  clf.fit(X_tr, y_tr)
  print(f"Accuracy: {{clf.score(X_te, y_te):.3f}}")

  # Para vectorizar texto nuevo (inferencia):
  texto_nuevo = "el imss no tenía vacuna y tuve que ir al particular"
  X_nuevo = art["vectorizador"].transform([texto_nuevo])
  # + scaler.transform para features numéricas → hstack → clf.predict
  ──────────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()

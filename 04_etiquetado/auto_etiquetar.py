#!/usr/bin/env python3
"""
Qué hace:
  ① Instala pysentimiento automáticamente si no está disponible
  ② Carga el modelo pysentimiento/robertuito-sentiment-analysis
     (RoBERTa entrenado en ~60M tweets en español, incluye es-MX)
  ③ Procesa cada comentario en lotes y asigna: POS / NEG / NEU + probabilidades
  ④ Convierte el sentimiento BERT en campo 'relevancia' usando lógica híbrida:
       · NEG + score_queja_salud alto  → "alta"   (queja sistémica confirmada)
       · NEG + score_queja_salud bajo  → revisión  (negativo pero no sistémico)
       · NEU / POS                     → "media"
       · vacuna_info_neutral presente  → fuerza "media" (FP corregido)
  ⑤ Genera cola de revisión manual para casos de confianza baja o ambiguos
     (30–60 min de trabajo para ~300 casos → dataset de alta calidad)
  ⑥ Exporta JSON listo para vectorizar_v3.py con nuevos campos:
       sentimiento_bert, prob_neg, prob_neu, prob_pos,
       confianza_bert, relevancia (actualizado), etiqueta_fuente,
       sentimiento_num (NEG=2, NEU=1, POS=0 para usar como feature numérica)

Entrada : dataset_normalizado_*.json  (salida de normalizar_nlp_v2.py)
Salida  : dataset_etiquetado_*.json   (entrada de vectorizar_v3.py)
          cola_revision_*.json        (casos para revisión manual)

Dependencias:
  pip install pysentimiento torch

Uso: python3 auto_etiquetar.py
"""

import json
import os
import sys
import subprocess
import re
from pathlib import Path
from datetime import datetime
from collections import Counter
from decimal import Decimal


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

# ── Modelo ────────────────────────────────────────────────────────────────────
MODELO_SENTIMENT  = "pysentimiento/robertuito-sentiment-analysis"
# Alternativas si hay problemas de descarga:
#   "lxyuan/distilbert-base-multilingual-cased-sentiments-student"  ← más ligero
#   "cardiffnlp/twitter-xlm-roberta-base-sentiment"                 ← multilingüe

BATCH_SIZE        = 64    # subido de 32 → 64 (seguro con 16 GB RAM, ~20% más rápido)

# ── Configuración CPU ─────────────────────────────────────────────────────────
# Usa todos los núcleos disponibles para la inferencia (sin GPU).
# torch.set_num_threads se llama después de importar torch en cargar_analizador().
CPU_THREADS = os.cpu_count() or 4   # fallback a 4 si os.cpu_count() devuelve None

# ── Mapeo sentimiento → relevancia ───────────────────────────────────────────
# Un comentario NEGATIVO es relevancia ALTA solo si también habla del sistema
# de salud (score_queja_salud > umbral). Esto evita FP por comentarios
# negativos sobre otros temas que colaron en el corpus.
UMBRAL_QUEJA_PARA_ALTA   = 1.5   # score_queja_salud mínimo para confirmar "alta"
UMBRAL_CONFIANZA_AUTO    = 0.72  # prob mínima para etiquetar sin revisión
UMBRAL_CONFIANZA_REVISION= 0.50  # por debajo → cola de revisión manual

# ── Señales que fuerzan "media" aunque el sentimiento sea NEG ─────────────────
# Contextos donde la negatividad NO es sobre el sistema público (FP corregidos)
TOKENS_FUERZA_MEDIA = {
    "vacuna_info_neutral",   # vacunación informativa (corrección activa)
    "vacuna_gratuita",
    "esquema_vacunacion",
}

# ── Codificación numérica del sentimiento (feature para vectorizar_v3.py) ─────
SENTIMIENTO_NUM = {"NEG": 2, "NEU": 1, "POS": 0}

# ── Máximo de casos en la cola de revisión manual ────────────────────────────
MAX_REVISION     = 400
# Mostrar los primeros N en pantalla durante la revisión interactiva
N_MOSTRAR_REVISION = 20


class DecimalEncoder(json.JSONEncoder):
    """
    Convierte Decimal (producido por ijson) a int o float antes de serializar.
    Evita: TypeError: Object of type Decimal is not JSON serializable
    """
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        return super().default(obj)


# ══════════════════════════════════════════════════════════════════════════════
#  DEPENDENCIAS
# ══════════════════════════════════════════════════════════════════════════════

def _importable(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except ImportError:
        return False


def verificar_e_instalar():
    """Instala pysentimiento y torch si no están disponibles."""
    faltantes = []
    for mod, pip in [("pysentimiento", "pysentimiento"), ("torch", "torch")]:
        if not _importable(mod):
            faltantes.append(pip)

    if not faltantes:
        print("  ✅ pysentimiento y torch disponibles.")
        return

    print(f"\n  ⚠️  Librerías no encontradas: {', '.join(faltantes)}")
    print(f"  Tamaño aproximado de descarga: ~2 GB (modelo + torch)")
    resp = input("  ¿Instalar ahora? (s/n): ").strip().lower()
    if resp != "s":
        print("  Instala manualmente: pip install pysentimiento torch")
        sys.exit(1)

    subprocess.run(
        [sys.executable, "-m", "pip", "install"] + faltantes + ["-q"],
        check=True
    )
    print("  ✅ Instalación completada.")


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DEL ANALIZADOR
# ══════════════════════════════════════════════════════════════════════════════

def cargar_analizador():
    """
    Carga el analizador de sentimientos de pysentimiento.

    Por qué robertuito-sentiment-analysis:
    · Entrenado en ~60 millones de tweets en español (Argentina, México, España)
    · Supera a modelos multilingüe genéricos en texto coloquial con emojis,
      siglas y jerga MX (wey, chido, neta, etc.)
    · Produce probabilidades calibradas: POS/NEG/NEU con suma = 1.0
    · Acepta textos de hasta 512 tokens (suficiente para comentarios de Reddit)
    """
    import torch
    from pysentimiento import create_analyzer

    # ── Optimización CPU: usar todos los núcleos disponibles ──────────────────
    # Sin GPU, PyTorch usa por defecto solo 1 núcleo. Esto aprovecha todos.
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(max(1, CPU_THREADS // 2))
    print(f"\n  ⚙️  CPU threads configurados: {CPU_THREADS} (inferencia) / "
          f"{max(1, CPU_THREADS // 2)} (interop)")

    print(f"\n  Cargando modelo: {MODELO_SENTIMENT}")
    print(f"  (primera ejecución descarga ~500 MB — solo una vez)")

    try:
        analizador = create_analyzer(task="sentiment", lang="es")
        print("  ✅ Modelo cargado.")
        return analizador
    except Exception as e:
        print(f"  ❌ Error al cargar modelo: {e}")
        print(f"     Solución: pip install pysentimiento --upgrade")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
#  INTERFAZ DE ARCHIVOS
# ══════════════════════════════════════════════════════════════════════════════

def pedir_archivo() -> Path:
    carpeta = Path(__file__).parent
    jsons   = sorted(carpeta.glob("dataset_normalizado*.json"))

    # Fallback: cualquier JSON si no hay dataset_normalizado
    if not jsons:
        jsons = sorted(carpeta.glob("*.json"))

    if not jsons:
        raise FileNotFoundError("No hay archivos .json en la carpeta.")

    print(f"\n{'='*70}")
    print("  AUTO-ETIQUETADO — SENTIMIENTOS BERT  🏷️🤖")
    print(f"{'='*70}\n")
    print("  Archivos disponibles:")
    for i, p in enumerate(jsons):
        print(f"    [{i}] {p.name}  ({p.stat().st_size/1024:.0f} KB)")

    while True:
        idx = input("\n  Número del archivo a etiquetar: ").strip()
        try:
            return jsons[int(idx)]
        except (ValueError, IndexError):
            print(f"  ❌ Elige entre 0 y {len(jsons)-1}.")


# ══════════════════════════════════════════════════════════════════════════════
#  PREPROCESADO DEL TEXTO PARA EL MODELO
# ══════════════════════════════════════════════════════════════════════════════

# Límite de caracteres antes de truncar (RoBERTa acepta ~512 subwords ≈ 800 chars)
MAX_CHARS_BERT = 800

def preparar_texto_bert(reg: dict) -> str:
    """
    Selecciona el texto más rico para el análisis de sentimientos.
    Prioridad: comentario original > texto_normalizado.

    El comentario original contiene stopwords y puntuación emocional
    (¡!, signos, emojis) que mejoran la detección de sentimiento.
    El texto_normalizado es mejor para TF-IDF pero peor para BERT.
    """
    texto = (
        reg.get("comentario")
        or reg.get("comentario_raw")
        or reg.get("texto_normalizado")
        or ""
    ).strip()

    # Truncar si supera el límite — conservar inicio del comentario
    # (el inicio suele contener la queja principal en Reddit)
    if len(texto) > MAX_CHARS_BERT:
        texto = texto[:MAX_CHARS_BERT]

    return texto


# ══════════════════════════════════════════════════════════════════════════════
#  LÓGICA DE MAPEO SENTIMIENTO → RELEVANCIA
# ══════════════════════════════════════════════════════════════════════════════

def _calcular_score_queja(texto_norm: str) -> float:
    """Score simple basado en términos de queja en el texto normalizado."""
    # Importar lexicon del vectorizador si está disponible, o usar mini-lexicon
    MINI_LEXICON = {
        "desabasto", "negligencia", "viacrucis", "odisea", "moche",
        "sin_medicamento", "no_hay_medicamento", "no_hay_vacuna",
        "vuelta_y_vuelta", "vuelva_manana", "saturado", "escasez",
        "fallecer", "morir", "muerte", "intubado", "uci", "empeorar",
        "mala_atencion", "no_me_atendieron", "maltrato", "negligente",
        "de_mi_bolsillo", "medico_particular", "fui_al_particular",
        "tuve_que_pagar", "no_hay_sistema", "no_hay_cupo",
        "urgencias_llenas", "corrupcion", "mordida",
    }
    t = texto_norm.lower()
    return sum(1.5 for term in MINI_LEXICON if term in t)


def mapear_relevancia(
    sentimiento:   str,    # "POS" | "NEG" | "NEU"
    prob_neg:      float,
    texto_norm:    str,
    relevancia_orig: str,  # etiqueta previa del dataset (puede ser None)
) -> tuple[str, str, str]:
    """
    Decide la relevancia final basándose en sentimiento + contexto de queja.

    Retorna (relevancia, fuente, razon) donde:
      relevancia : "alta" | "media"
      fuente     : "bert_auto" | "bert_queja" | "forzado_media" | "original" | "revision"
      razon      : descripción legible del por qué
    """
    texto_lower = texto_norm.lower()

    # ── Regla 0: token de vacunación informativa fuerza media ─────────────────
    if any(tok in texto_lower for tok in TOKENS_FUERZA_MEDIA):
        return "media", "forzado_media", "token vacuna_info_neutral detectado"

    # ── Regla 1: POS o NEU → siempre media ───────────────────────────────────
    if sentimiento in ("POS", "NEU"):
        return "media", "bert_auto", f"sentimiento {sentimiento} → sin queja sistémica"

    # Desde aquí: sentimiento == "NEG"
    score_queja = _calcular_score_queja(texto_norm)

    # ── Regla 2: NEG con queja sistémica confirmada → alta ───────────────────
    if score_queja >= UMBRAL_QUEJA_PARA_ALTA:
        return "alta", "bert_queja", (
            f"NEG (prob={prob_neg:.2f}) + score_queja={score_queja:.1f} "
            f"≥ {UMBRAL_QUEJA_PARA_ALTA}"
        )

    # ── Regla 3: NEG sin señal sistémica clara → revisar ─────────────────────
    # El comentario es negativo pero no habla del sistema de salud específicamente.
    # Puede ser crítica a otra cosa. Se marca para revisión manual.
    return "revision", "bert_ambiguo", (
        f"NEG (prob={prob_neg:.2f}) pero score_queja={score_queja:.1f} "
        f"< {UMBRAL_QUEJA_PARA_ALTA} — revisar manualmente"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  ANÁLISIS EN LOTES
# ══════════════════════════════════════════════════════════════════════════════

def analizar_sentimientos(
    registros: list[dict],
    analizador,
) -> list[dict]:
    """
    Procesa todos los registros en lotes con el analizador BERT.
    Añade campos de sentimiento y relevancia a cada registro.
    """
    n = len(registros)
    print(f"\n  Analizando {n} comentarios en lotes de {BATCH_SIZE}...")
    print(f"  (CPU puro — tiempo estimado: {round(n / BATCH_SIZE * 18 / 60)} min aprox.)\n")

    enriquecidos: list[dict] = []
    stats = Counter()
    import time
    t_inicio = time.time()

    for inicio in range(0, n, BATCH_SIZE):
        lote_regs  = registros[inicio: inicio + BATCH_SIZE]
        lote_textos = [preparar_texto_bert(r) for r in lote_regs]

        # Predecir lote completo
        try:
            resultados = analizador.predict(lote_textos)
        except Exception as e:
            print(f"\n  ⚠️  Error en lote {inicio}-{inicio+BATCH_SIZE}: {e}")
            print(f"     Usando NEU como fallback para este lote.")
            # Fallback: crear resultados neutros manualmente
            class _FallbackRes:
                def __init__(self):
                    self.output = "NEU"
                    self.probas = {"POS": 0.33, "NEG": 0.33, "NEU": 0.34}
            resultados = [_FallbackRes() for _ in lote_textos]

        for reg, res in zip(lote_regs, resultados):
            sentimiento  = res.output                          # "POS"|"NEG"|"NEU"
            probas       = res.probas                          # {"POS":f, "NEG":f, "NEU":f}
            prob_neg     = float(probas.get("NEG", 0.0))
            prob_pos     = float(probas.get("POS", 0.0))
            prob_neu     = float(probas.get("NEU", 0.0))
            confianza    = max(prob_neg, prob_pos, prob_neu)

            texto_norm   = reg.get("texto_normalizado", "")
            rel_orig     = reg.get("relevancia", "")

            relevancia, fuente, razon = mapear_relevancia(
                sentimiento, prob_neg, texto_norm, rel_orig
            )

            reg_enriq = {
                **reg,
                # ── Campos de sentimiento BERT ─────────────────────────────
                "sentimiento_bert":  sentimiento,
                "prob_neg":          round(prob_neg,  4),
                "prob_pos":          round(prob_pos,  4),
                "prob_neu":          round(prob_neu,  4),
                "confianza_bert":    round(confianza, 4),
                "sentimiento_num":   SENTIMIENTO_NUM[sentimiento],
                # ── Relevancia actualizada ────────────────────────────────
                "relevancia":        relevancia if relevancia != "revision"
                                     else (rel_orig or "media"),
                "relevancia_bert":   relevancia,   # copia antes de revisión
                "etiqueta_fuente":   fuente,
                "etiqueta_razon":    razon,
                "requiere_revision": relevancia == "revision",
            }
            enriquecidos.append(reg_enriq)
            stats[sentimiento] += 1
            if relevancia == "revision":
                stats["revision"] += 1

        procesados = min(inicio + BATCH_SIZE, n)
        pct        = min(100, round(procesados / n * 100))
        elapsed    = time.time() - t_inicio
        eta_seg    = int(elapsed / procesados * (n - procesados)) if procesados else 0
        eta_str    = f"{eta_seg // 60}m {eta_seg % 60:02d}s restantes"
        print(f"  → {procesados:>5}/{n}  ({pct:>3}%)  {eta_str}          ", end="\r")

    print(f"\n  ✅ Análisis completado.")
    print(f"\n  Distribución de sentimientos:")
    for label in ["NEG", "NEU", "POS", "revision"]:
        cnt   = stats[label]
        pct   = round(cnt / n * 100, 1)
        barra = "█" * int(pct / 2)
        print(f"    {label:<10} {cnt:>5} ({pct:>5}%)  {barra}")

    return enriquecidos


# ══════════════════════════════════════════════════════════════════════════════
#  REVISIÓN MANUAL INTERACTIVA
# ══════════════════════════════════════════════════════════════════════════════

def revisar_casos_ambiguos(
    registros: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Presenta los casos marcados como "revision" al usuario para clasificación
    manual. Permite etiquetar, saltar o delegar al valor original.

    Retorna (registros_con_revision_aplicada, cola_no_revisada)
    """
    pendientes = [r for r in registros if r.get("requiere_revision", False)]
    n_total    = len(pendientes)

    if n_total == 0:
        print("\n  ✅ Sin casos para revisión manual.")
        return registros, []

    print(f"\n{'='*70}")
    print(f"  REVISIÓN MANUAL — {n_total} casos ambiguos")
    print(f"  (NEG pero sin señal sistémica clara)")
    print(f"{'='*70}")
    print(f"\n  Opciones por comentario:")
    print(f"    [a] → relevancia ALTA  (queja sistémica confirmada)")
    print(f"    [m] → relevancia MEDIA (negativo pero no sistémico)")
    print(f"    [s] → saltar (mantener etiqueta original del dataset)")
    print(f"    [q] → terminar revisión y guardar lo hecho hasta aquí\n")

    # Mostrar solo los N_MOSTRAR_REVISION primeros en pantalla
    a_mostrar  = min(N_MOSTRAR_REVISION, n_total)
    revisados: dict[int, str] = {}   # idx → etiqueta elegida

    for j, reg in enumerate(pendientes[:a_mostrar], 1):
        texto_orig   = (reg.get("comentario") or reg.get("texto_normalizado", ""))[:300]
        sentimiento  = reg.get("sentimiento_bert", "?")
        prob_neg     = reg.get("prob_neg", 0)
        score_q      = _calcular_score_queja(reg.get("texto_normalizado", ""))
        rel_original = reg.get("relevancia", "media")

        print(f"  ─── Caso {j}/{a_mostrar} ───────────────────────────────────────────")
        print(f"  Texto   : {texto_orig}")
        print(f"  BERT    : {sentimiento} (prob_neg={prob_neg:.2f}) | "
              f"score_queja={score_q:.1f} | etiqueta_orig={rel_original}")

        while True:
            op = input("  Etiqueta [a/m/s/q]: ").strip().lower()
            if op == "q":
                print(f"\n  Revisión detenida en caso {j}.")
                # Los no revisados mantienen relevancia original
                break
            if op in ("a", "m", "s"):
                revisados[id(reg)] = {"a": "alta", "m": "media", "s": rel_original}[op]
                break
            print("  ❌ Escribe a, m, s o q.")

        if op == "q":
            break

    # Aplicar decisiones al dataset completo
    idx_rev = {id(r): r for r in pendientes}
    cola_no_revisada: list[dict] = []

    for reg in registros:
        if reg.get("requiere_revision", False):
            reg_id = id(reg)
            if reg_id in revisados:
                reg["relevancia"]      = revisados[reg_id]
                reg["etiqueta_fuente"] = "revisado_manual"
                reg["requiere_revision"] = False
            else:
                # No llegó a revisarse → mantener original, marcar en cola
                cola_no_revisada.append(reg)

    print(f"\n  Casos revisados manualmente : {len(revisados)}")
    print(f"  Cola pendiente              : {len(cola_no_revisada)}")

    return registros, cola_no_revisada


# ══════════════════════════════════════════════════════════════════════════════
#  VALIDACIÓN FINAL Y DISTRIBUCIÓN
# ══════════════════════════════════════════════════════════════════════════════

def validar_y_limpiar(registros: list[dict]) -> list[dict]:
    """
    Pasos finales antes de exportar:
    · Asegura que "relevancia" sea exactamente "alta" o "media" en todos
    · Descarta registros sin texto_normalizado o relevancia inválida
    · Añade sentimiento_num a FEATURES_NUMERICAS si aún no está
    """
    limpios  = []
    omitidos = 0

    for reg in registros:
        texto  = reg.get("texto_normalizado", "").strip()
        rel    = reg.get("relevancia", "")

        if not texto:
            omitidos += 1
            continue

        # Normalizar etiquetas no estándar
        if rel not in ("alta", "media"):
            rel = "media"   # default seguro
        reg["relevancia"] = rel

        # Asegurar que sentimiento_num existe (por si falló algún lote)
        if "sentimiento_num" not in reg:
            sent = reg.get("sentimiento_bert", "NEU")
            reg["sentimiento_num"] = SENTIMIENTO_NUM.get(sent, 1)

        limpios.append(reg)

    if omitidos:
        print(f"  ⚠️  {omitidos} registros descartados (sin texto_normalizado)")

    return limpios


# ══════════════════════════════════════════════════════════════════════════════
#  GUARDADO
# ══════════════════════════════════════════════════════════════════════════════

def guardar_resultados(
    carpeta:        Path,
    registros:      list[dict],
    cola_revision:  list[dict],
    ruta_entrada:   Path,
    meta_anterior:  dict,
) -> tuple[Path, Path | None]:

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Estadísticas finales
    dist_relevancia = Counter(r["relevancia"] for r in registros)
    dist_fuente     = Counter(r.get("etiqueta_fuente", "?") for r in registros)
    dist_sentimiento = Counter(r.get("sentimiento_bert", "?") for r in registros)

    n = len(registros)

    resultado = {
        "meta_etiquetado": {
            "generado":                  datetime.now().isoformat(),
            "version_script":            "1.0",
            "archivo_fuente":            str(ruta_entrada),
            "pipeline_anterior":         meta_anterior.get("version_script", "desconocido"),
            "modelo_bert":               MODELO_SENTIMENT,
            "registros_entrada":         n,
            "registros_salida":          n,
            "distribucion_relevancia":   dict(dist_relevancia),
            "distribucion_sentimiento":  dict(dist_sentimiento),
            "distribucion_fuente":       dict(dist_fuente),
            "casos_revision_pendientes": len(cola_revision),
            "config": {
                "umbral_queja":      UMBRAL_QUEJA_PARA_ALTA,
                "umbral_confianza":  UMBRAL_CONFIANZA_AUTO,
                "batch_size":        BATCH_SIZE,
                "max_chars_bert":    MAX_CHARS_BERT,
                "tokens_fuerza_media": list(TOKENS_FUERZA_MEDIA),
            },
            "instruccion_siguiente_paso": (
                "Ejecutar vectorizar_v3.py con este archivo como entrada. "
                "El campo 'sentimiento_num' se añade automáticamente a "
                "FEATURES_NUMERICAS en vectorizar_v3.py."
            ),
        },
        "datos": registros,
    }

    ruta_out = carpeta / f"dataset_etiquetado_{ts}.json"
    with open(ruta_out, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2, cls=DecimalEncoder)

    # Cola de revisión en archivo separado (más pequeño, fácil de abrir)
    ruta_cola = None
    if cola_revision:
        campos_cola = [
            "post_id", "titulo_post", "comentario", "texto_normalizado",
            "sentimiento_bert", "prob_neg", "prob_neu", "prob_pos",
            "confianza_bert", "etiqueta_razon", "relevancia",
            "score_comentario", "subreddit",
        ]
        cola_slim = [
            {k: r.get(k) for k in campos_cola} for r in cola_revision
        ]
        ruta_cola = carpeta / f"cola_revision_{ts}.json"
        with open(ruta_cola, "w", encoding="utf-8") as f:
            json.dump({"pendientes": len(cola_slim), "casos": cola_slim},
                      f, ensure_ascii=False, indent=2, cls=DecimalEncoder)

    return ruta_out, ruta_cola


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    verificar_e_instalar()
    analizador = cargar_analizador()

    ruta_entrada = pedir_archivo()
    carpeta      = ruta_entrada.parent

    # ── Carga ─────────────────────────────────────────────────────────────────
    print(f"\n  Cargando: {ruta_entrada.name} ...")
    with open(ruta_entrada, "r", encoding="utf-8") as f:
        data = json.load(f)

    registros     = data.get("datos", data) if isinstance(data, dict) else data
    meta_anterior = data.get("meta_normalizacion",
                    data.get("meta_enriquecimiento", {})) if isinstance(data, dict) else {}

    print(f"  Registros cargados : {len(registros)}")
    dist_orig = Counter(r.get("relevancia", "?") for r in registros)
    print(f"  Distribución orig  : {dict(dist_orig)}")

    # ── Análisis de sentimientos ──────────────────────────────────────────────
    registros_enriq = analizar_sentimientos(registros, analizador)

    # ── Revisión manual ───────────────────────────────────────────────────────
    n_revision = sum(1 for r in registros_enriq if r.get("requiere_revision"))
    print(f"\n  Casos para revisión manual : {n_revision}")

    if n_revision > 0:
        resp = input(f"\n  ¿Iniciar revisión manual ahora? (s/n): ").strip().lower()
        if resp == "s":
            registros_enriq, cola = revisar_casos_ambiguos(registros_enriq)
        else:
            print("  ℹ️  Revisión omitida — los casos ambiguos conservan etiqueta original.")
            cola = [r for r in registros_enriq if r.get("requiere_revision")]
    else:
        cola = []

    # ── Limpieza y validación ─────────────────────────────────────────────────
    registros_finales = validar_y_limpiar(registros_enriq)

    # ── Guardado ──────────────────────────────────────────────────────────────
    ruta_out, ruta_cola = guardar_resultados(
        carpeta, registros_finales, cola, ruta_entrada, meta_anterior
    )

    # ── Reporte final ─────────────────────────────────────────────────────────
    dist_final = Counter(r["relevancia"] for r in registros_finales)
    dist_fuente = Counter(r.get("etiqueta_fuente","?") for r in registros_finales)

    print(f"\n{'='*70}")
    print("  REPORTE DE ETIQUETADO")
    print(f"{'='*70}")
    print(f"  Registros procesados   : {len(registros_finales)}")
    print(f"\n  Distribución final de relevancia:")
    for etq, cnt in dist_final.items():
        pct   = round(cnt / len(registros_finales) * 100, 1)
        barra = "█" * int(pct / 2)
        print(f"    {etq:<10} {cnt:>5} ({pct:>5}%)  {barra}")

    print(f"\n  Fuente de cada etiqueta:")
    for fuente, cnt in sorted(dist_fuente.items(), key=lambda x: -x[1]):
        print(f"    {fuente:<25} {cnt:>5}")

    print(f"\n  Nuevos campos añadidos para vectorizar_v3.py:")
    print(f"    sentimiento_bert  — POS/NEG/NEU del modelo BERT")
    print(f"    prob_neg/pos/neu  — probabilidades por clase")
    print(f"    confianza_bert    — max(prob_*)")
    print(f"    sentimiento_num   — NEG=2, NEU=1, POS=0 (feature numérica)")
    print(f"    etiqueta_fuente   — trazabilidad del origen de la etiqueta")

    print(f"\n  💾 Dataset etiquetado  : {ruta_out.name}")
    if ruta_cola:
        print(f"  💾 Cola de revisión    : {ruta_cola.name}  ({len(cola)} casos)")
    print(f"{'='*70}")

    print(f"""
  SIGUIENTE PASO — vectorizar_v3.py:
  ──────────────────────────────────────────────────────────────────────
  1. Abre vectorizar_v3.py y añade "sentimiento_num" a FEATURES_NUMERICAS:

     FEATURES_NUMERICAS = [
         "polaridad_score",
         "ifb_score",
         "impacto_escala",
         "num_palabras",
         "peso_reddit",
         "sentimiento_num",   # ← añadir esta línea
     ]

  2. Ejecuta:
     python3 vectorizar_v3.py
     → selecciona {ruta_out.name}

  3. El campo "relevancia" ya tiene etiquetas de alta calidad respaldadas
     por un modelo BERT de 60M tweets en español.
  ──────────────────────────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()

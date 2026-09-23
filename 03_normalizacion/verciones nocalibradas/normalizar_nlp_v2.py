#!/usr/bin/env python3
"""
normalizar_nlp.py v2 — Paso 1 del pipeline de modelado (CRISP-DM Fase 3 → 4)
Tema: Ineficiencia del sector salud MX frente a la influenza

Mejoras v2:
  ① Stopwords refinadas: negaciones y conectores causales SIEMPRE protegidos
     ("no", "ni", "tampoco", "sin", "nunca", "porque", "debido", "por"...)
  ② Detección de N-gramas: bigramas y trigramas clave del dominio MX
     ("seguro social", "no hay sistema", "centro de salud", "vuelva mañana"...)
  ③ Lematización con preservación de sentimiento: tabla de excepciones para
     adjetivos extremos y verbos de alto impacto clínico que NO se deben
     neutralizar ("pésimo" no → "malo", "falleció" no → "fallecer" genérico)
  ④ NER — Extracción de Entidades Nombradas: detecta instituciones (ORG)
     y lugares (LOC) y genera campo "entidades_ner" por registro
  ⑤ Vectorización avanzada: peso por score de Reddit + nota sobre embeddings
  ⑥ Robustez: try/except en carga del modelo + nlp.pipe() en lotes (batches)

Heredadas de v1:
  ⑦ Tokenización, lematización, eliminación de stopwords con spaCy
  ⑧ Conservación de términos de dominio aunque sean stopwords
  ⑨ Estadísticas de vocabulario y top términos
  ⑩ JSON listo para TF-IDF / Bag of Words / embeddings

Entrada : dataset_enriquecido_*.json  (salida de enriquecer_dataset.py)
Salida  : dataset_normalizado_*.json

Uso: python3 normalizar_nlp.py
"""

import json
import re
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from collections import Counter


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GENERAL
# ══════════════════════════════════════════════════════════════════════════════

MIN_LEN_TOKEN   = 2    # filtra letras sueltas ("a", "y", "o")
BATCH_SIZE      = 64   # registros por lote en nlp.pipe() (mejora 6)
MODELO_SPACY    = "es_core_news_sm"


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA 1 — STOPWORDS REFINADAS
#  Negaciones y conectores causales NUNCA se eliminan.
#  Razón: "no hay medicina" ≠ "hay medicina"; "empeoró PORQUE no atendieron"
#  pierde la causalidad si se borra "porque".
# ══════════════════════════════════════════════════════════════════════════════

# Protegidos de cualquier filtro, incluyendo el de spaCy
NEGACIONES_PROTEGIDAS = {
    "no", "ni", "tampoco", "sin", "nunca", "jamás", "jamas",
    "nada", "nadie", "ningún", "ningun", "ninguno", "ninguna",
}

CONECTORES_CAUSALES = {
    # Causa → consecuencia: vitales para detectar "ineficiencia → daño"
    "porque", "pues", "ya que", "dado que", "puesto que",
    "debido", "por", "a causa", "gracias a",   # "gracias a su negligencia"
    "entonces", "así que", "asi que", "por eso", "por lo tanto",
    "por eso", "en consecuencia", "resultado",
    # Contraste / concesión (también importan para sentimiento)
    "pero", "aunque", "sin embargo", "a pesar",
}

# Unión de todos los términos protegidos del filtro stopword
SIEMPRE_CONSERVAR = NEGACIONES_PROTEGIDAS | CONECTORES_CAUSALES

# Términos del dominio que NO se eliminan aunque spaCy los marque stopword
TERMINOS_DOMINIO = {
    # Instituciones
    "imss", "issste", "insabi", "ssa", "imss-bienestar",
    # Enfermedad
    "influenza", "gripe", "gripa", "h1n1", "h3n2", "virus", "viral",
    # Problema estructural
    "desabasto", "escasez", "moche", "mordida", "fila", "cola",
    "saturado", "saturar", "recorte", "corrupción", "corrupcion",
    "negligencia", "negligente", "viacrucis", "odisea",
    # Tratamiento
    "vacuna", "vacunar", "oseltamivir", "tamiflu", "medicamento",
    "medicina", "tratamiento", "dosis", "antiviral", "paracetamol",
    # Impacto clínico
    "neumonía", "neumonia", "intubado", "uci", "falleció", "fallecio",
    "murió", "murio", "muerte", "empeoró", "empeoro",
    # Pago privado
    "particular", "privado", "privada",
} | SIEMPRE_CONSERVAR   # las negaciones y causales también son dominio

# Stopwords adicionales específicas de Reddit MX (no aportan tema ni sentimiento)
STOPWORDS_EXTRA = {
    "reddit", "post", "comentario", "hilo", "thread",
    "edit", "update", "fuente", "link", "bot",
    "wey", "güey", "guey", "nomás", "nomas",
    "osea", "tipo", "igual",
    "también", "tambien", "además", "ademas",
    "aquí", "aqui", "allá", "alla", "acá", "aca",
    "hacer", "decir", "saber", "poder", "querer", "tener",
    "ser", "estar", "ir", "venir", "ver", "dar",
    # Nota: "bueno/buena" y "entonces" se retiran de STOPWORDS_EXTRA porque
    # "bueno" puede ser irónico y "entonces" es conector causal → CONSERVAR
}


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA 2 — N-GRAMAS DEL DOMINIO (bigramas y trigramas predefinidos)
#
#  Por qué una lista predefinida en vez de extracción automática:
#  Los n-gramas automáticos (PMI, frecuencia) necesitan corpus grande y pueden
#  capturar ruido. Aquí definimos los n-gramas que SABEMOS que son conceptos
#  unitarios en salud pública MX, garantizando su preservación semántica.
#
#  Implementación: ANTES de tokenizar con spaCy, reemplazamos el bigrama/
#  trigrama por un token con guión bajo ("seguro_social") para que spaCy
#  lo trate como una unidad léxica única y no lematice sus partes por separado.
# ══════════════════════════════════════════════════════════════════════════════

NGRAMAS_DOMINIO: list[tuple[str, str]] = [
    # ── Trigramas (se aplican primero para no partir frases largas) ───────────
    ("no hay sistema",          "no_hay_sistema"),
    ("no hay medicamento",      "no_hay_medicamento"),
    ("no hay medicina",         "no_hay_medicina"),
    ("no hay vacuna",           "no_hay_vacuna"),
    ("no hay camas",            "no_hay_camas"),
    ("falta de medicamento",    "falta_de_medicamento"),
    ("falta de medicina",       "falta_de_medicina"),
    ("vuelva usted mañana",     "vuelva_manana"),
    ("no me atendieron",        "no_me_atendieron"),
    ("no me dieron",            "no_me_dieron"),
    ("de mi bolsillo",          "de_mi_bolsillo"),
    ("médico particular",       "medico_particular"),
    ("medico particular",       "medico_particular"),
    ("sistema de salud",        "sistema_de_salud"),
    ("sector salud",            "sector_salud"),
    ("secretaría de salud",     "secretaria_de_salud"),
    ("secretaria de salud",     "secretaria_de_salud"),
    ("vuelta y vuelta",         "vuelta_y_vuelta"),
    ("mala atención",           "mala_atencion"),
    ("mala atencion",           "mala_atencion"),
    ("no hay cupo",             "no_hay_cupo"),
    ("sala de espera",          "sala_de_espera"),
    # ── Bigramas ─────────────────────────────────────────────────────────────
    ("seguro social",           "seguro_social"),
    ("centro de salud",         "centro_de_salud"),   # trigrama compuesto frecuente
    ("imss bienestar",          "imss_bienestar"),
    ("vuelva mañana",           "vuelva_manana"),
    ("sin medicamento",         "sin_medicamento"),
    ("sin vacuna",              "sin_vacuna"),
    ("sin medicina",            "sin_medicina"),
    ("seguro popular",          "seguro_popular"),
    ("salud pública",           "salud_publica"),
    ("salud publica",           "salud_publica"),
    ("clínica familiar",        "clinica_familiar"),
    ("clinica familiar",        "clinica_familiar"),
    ("urgencias llenas",        "urgencias_llenas"),
    ("no hay",                  "no_hay"),             # bigrama residual general
]

# Compilar patrones de n-gramas una sola vez (ordenados: trigramas primero)
# re.escape garantiza que los paréntesis, puntos, etc. no rompan el patrón
_NGRAMAS_COMPILADOS: list[tuple[re.Pattern, str]] = [
    (re.compile(re.escape(original), re.IGNORECASE), reemplazo)
    for original, reemplazo in NGRAMAS_DOMINIO
]


def aplicar_ngramas(texto: str) -> str:
    """
    Reemplaza n-gramas del dominio por tokens con guión bajo ANTES de spaCy.
    El guión bajo hace que spaCy los tokenice como una sola unidad léxica.
    Ej: "no hay medicamento" → "no_hay_medicamento"
    """
    for patron, reemplazo in _NGRAMAS_COMPILADOS:
        texto = patron.sub(reemplazo, texto)
    return texto


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA 3 — LEMATIZACIÓN CON PRESERVACIÓN DE SENTIMIENTO
#
#  spaCy a veces neutraliza términos de alto impacto emocional o clínico.
#  Esta tabla de excepciones garantiza que el lema final sea el correcto
#  para el análisis, no el que spaCy devuelve.
#
#  Criterio de inclusión:
#  · Adjetivos extremos cuyo lema spaCy sería un sinónimo más suave
#  · Verbos de impacto clínico cuya raíz genérica pierde gravedad
#  · N-gramas con guión bajo que no deben lematizarse internamente
# ══════════════════════════════════════════════════════════════════════════════

EXCEPCIONES_LEMA: dict[str, str] = {
    # ── Adjetivos de negatividad extrema ─────────────────────────────────────
    # spaCy puede mapear estos a raíces neutras; los anclamos
    "pésimo":       "pésimo",       # NO → "malo"
    "pesimo":       "pésimo",
    "pésima":       "pésimo",
    "pesima":       "pésimo",
    "terrible":     "terrible",     # NO → "mal"
    "horrible":     "horrible",
    "desastroso":   "desastroso",
    "desastrosa":   "desastroso",
    "negligente":   "negligente",   # NO → "descuidado" ni lema genérico
    "incompetente": "incompetente",
    "inútil":       "inútil",
    "inutiles":     "inútil",
    "inutil":       "inútil",
    "inútiles":     "inútil",
    # ── Verbos de impacto clínico — mantener la forma de pasado / resultado ──
    # La lematización a infinitivo pierde la temporalidad del hecho
    "falleció":     "fallecer",     # OK: infinitivo es aceptable aquí
    "fallecio":     "fallecer",
    "murió":        "morir",
    "murio":        "morir",
    "empeoró":      "empeorar",
    "empeoro":      "empeorar",
    "se complicó":  "complicar",
    "se complico":  "complicar",
    "recayó":       "recaer",
    "recayo":       "recaer",
    "internaron":   "internar",
    "hospitalizaron": "hospitalizar",
    # ── N-gramas con guión bajo — conservar tal cual ──────────────────────────
    "no_hay_sistema":       "no_hay_sistema",
    "no_hay_medicamento":   "no_hay_medicamento",
    "no_hay_vacuna":        "no_hay_vacuna",
    "seguro_social":        "seguro_social",
    "centro_de_salud":      "centro_de_salud",
    "vuelva_manana":        "vuelva_manana",
    "vuelta_y_vuelta":      "vuelta_y_vuelta",
    "mala_atencion":        "mala_atencion",
    "no_me_atendieron":     "no_me_atendieron",
    "sin_medicamento":      "sin_medicamento",
    "sin_vacuna":           "sin_vacuna",
    "salud_publica":        "salud_publica",
    "sector_salud":         "sector_salud",
    "no_hay":               "no_hay",
    "de_mi_bolsillo":       "de_mi_bolsillo",
    "medico_particular":    "medico_particular",
    "urgencias_llenas":     "urgencias_llenas",
    "sala_de_espera":       "sala_de_espera",
    "no_me_dieron":         "no_me_dieron",
    "no_hay_cupo":          "no_hay_cupo",
    "falta_de_medicamento": "falta_de_medicamento",
    "imss_bienestar":       "imss_bienestar",
    "secretaria_de_salud":  "secretaria_de_salud",
    "sistema_de_salud":     "sistema_de_salud",
}


def resolver_lema(token, texto_token: str) -> str:
    """
    Devuelve el lema correcto aplicando la tabla de excepciones.
    Prioridad: excepción manual > lema de spaCy > texto limpio del token.
    """
    # 1. Buscar en excepciones por texto original (antes de lematizar)
    if texto_token in EXCEPCIONES_LEMA:
        return EXCEPCIONES_LEMA[texto_token]

    lema_spacy = token.lemma_.lower().strip()

    # 2. Buscar en excepciones por lema de spaCy
    if lema_spacy in EXCEPCIONES_LEMA:
        return EXCEPCIONES_LEMA[lema_spacy]

    # 3. Lema de spaCy si es válido
    if lema_spacy and len(lema_spacy) >= MIN_LEN_TOKEN:
        return lema_spacy

    # 4. Fallback al texto limpio
    return texto_token


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA 4 — NER: EXTRACCIÓN DE ENTIDADES NOMBRADAS
#
#  Usando el pipeline NER de es_core_news_sm para identificar:
#  · ORG → instituciones de salud (IMSS, ISSSTE, Hospital General…)
#  · LOC → lugares donde ocurrió la queja (CDMX, Monterrey…)
#
#  Estas entidades se guardan en campo "entidades_ner" por registro,
#  permitiendo filtrar y comparar quejas por institución o región.
# ══════════════════════════════════════════════════════════════════════════════

# Instituciones de referencia para normalizar entidades detectadas por NER
# (la nomenclatura varía: "el IMSS", "El Seguro", "IMSS Bienestar"…)
INSTITUCION_CANONICA: dict[str, str] = {
    "imss":             "IMSS",
    "issste":           "ISSSTE",
    "insabi":           "INSABI",
    "ssa":              "SSA",
    "salubridad":       "SSA",
    "imss-bienestar":   "IMSS_Bienestar",
    "imss bienestar":   "IMSS_Bienestar",
    "seguro social":    "IMSS",
    "el seguro":        "IMSS",
    "la raza":          "Hospital_La_Raza",
    "siglo xxi":        "Hospital_SigloXXI",
    "20 de noviembre":  "Hospital_20Nov",
    "hospital general": "Hospital_General",
    "farmacia del ahorro": "Farmacia_del_Ahorro",
    "farmacias similares": "Farmacias_Similares",
    "cruz roja":        "Cruz_Roja",
    "secretaría de salud": "SSA",
    "secretaria de salud": "SSA",
}


def extraer_entidades_ner(doc) -> dict:
    """
    Extrae entidades nombradas del doc de spaCy ya procesado.

    Retorna:
      {
        "instituciones": ["IMSS", "ISSSTE", ...],   ← ORG normalizadas
        "lugares":       ["CDMX", "Monterrey", ...], ← LOC/GPE
        "raw_ner":       [{"texto": "...", "tipo": "ORG"}, ...]
      }
    """
    instituciones_set: set[str] = set()
    lugares_set:       set[str] = set()
    raw: list[dict]             = []

    for ent in doc.ents:
        texto_ent = ent.text.strip()
        texto_lower = texto_ent.lower()

        raw.append({"texto": texto_ent, "tipo": ent.label_})

        if ent.label_ in ("ORG", "PRODUCT"):
            # Intentar normalizar con tabla canónica
            canonico = INSTITUCION_CANONICA.get(texto_lower)
            if canonico:
                instituciones_set.add(canonico)
            elif any(kw in texto_lower for kw in
                     ["hospital", "clínica", "clinica", "centro de salud",
                      "imss", "issste", "ssa", "insabi", "salud", "farmacia"]):
                instituciones_set.add(texto_ent)

        elif ent.label_ in ("LOC", "GPE"):
            lugares_set.add(texto_ent)

    # Búsqueda manual de instituciones por si NER las perdió
    # (es_core_news_sm tiene recall limitado en textos coloquiales)
    texto_doc_lower = doc.text.lower()
    for termino, canonico in INSTITUCION_CANONICA.items():
        if termino in texto_doc_lower:
            instituciones_set.add(canonico)

    return {
        "instituciones": sorted(instituciones_set),
        "lugares":        sorted(lugares_set),
        "raw_ner":        raw,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA 5 — VECTORIZACIÓN AVANZADA: PESO POR SCORE DE REDDIT
#
#  El campo "peso_reddit" por registro se calcula combinando:
#    · score_comentario: upvotes del comentario (validación comunitaria)
#    · score_post: popularidad del hilo padre
#    · profundidad: comentarios en niveles 0-1 tienen más visibilidad
#
#  Uso posterior:
#    Al construir la matriz TF-IDF, multiplicar tf_idf_matrix por
#    np.array([reg["peso_reddit"] for reg in datos]) antes de entrenar
#    → los términos en comentarios muy votados tendrán más influencia.
#
#  Para Word Embeddings (FastText / Word2Vec):
#    El campo "texto_normalizado" ya contiene n-gramas como tokens únicos
#    ("seguro_social", "no_hay_sistema"). Esto mejora la calidad de los
#    vectores porque el modelo aprende que "no_hay_sistema" es un concepto
#    distinto de "hay" y "sistema" por separado.
#
#    Modelo recomendado para el corpus (orden de preferencia):
#      1. fasttext-sbwc (FastText preentrenado en español, 1M vectores)
#         → pip install fasttext  +  descarga sbwc.bin desde INGEOTEC
#      2. es_core_news_md / es_core_news_lg de spaCy (vectores GloVe)
#         → python -m spacy download es_core_news_lg
#      3. sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
#         → pip install sentence-transformers  (mejor para similitud semántica)
# ══════════════════════════════════════════════════════════════════════════════

def calcular_peso_reddit(reg: dict) -> float:
    """
    Calcula el peso de ponderación para vectorización.
    Escala: ~0.1 (comentario invisible) → ~5.0 (comentario muy votado en hilo popular)

    Componentes:
      · score_comentario normalizado a escala log (evita que outliers dominen)
      · score_post con peso reducido (el hilo importa menos que el comentario)
      · penalización por profundidad >2 (comentarios anidados tienen menos visibilidad)
    """
    import math

    score_c = max(reg.get("score_comentario", 0), 0)  # negativos → 0
    score_p = max(reg.get("score_post", 0), 0)
    prof    = reg.get("profundidad", 0)

    # Log(1+x) para manejar la distribución asimétrica de upvotes de Reddit
    peso = (math.log1p(score_c) * 0.7 +
            math.log1p(score_p) * 0.3)

    # Penalización por profundidad alta (menor visibilidad)
    if prof >= 3:
        peso *= 0.7
    elif prof == 2:
        peso *= 0.85

    # Score mínimo de 0.1 para no anular comentarios nuevos/sin votos
    return round(max(peso, 0.1), 4)


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA 6 — ROBUSTEZ: CARGA SEGURA DE SPACY + nlp.pipe() EN LOTES
# ══════════════════════════════════════════════════════════════════════════════

def instalar_spacy():
    """Instala spaCy y el modelo de español si no están disponibles."""
    print("\n  Verificando dependencias NLP...")
    try:
        import spacy
        print("  ✅ spaCy disponible.")
    except ImportError:
        print("  Instalando spaCy...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "spacy", "-q"], check=True
        )
        print("  ✅ spaCy instalado.")

    # Modelo con try/except específico y descarga con fallback manual
    try:
        import spacy as _spacy
        _spacy.load(MODELO_SPACY)
        print(f"  ✅ Modelo {MODELO_SPACY} disponible.")
    except OSError:
        print(f"  Descargando modelo {MODELO_SPACY} (~12 MB)...")
        resultado = subprocess.run(
            [sys.executable, "-m", "spacy", "download", MODELO_SPACY, "-q"],
            capture_output=True, text=True
        )
        if resultado.returncode != 0:
            print(f"  ⚠️  Descarga automática falló. Intenta manualmente:")
            print(f"      python -m spacy download {MODELO_SPACY}")
            print(f"      (puede requerir permisos de administrador)")
            sys.exit(1)
        print(f"  ✅ Modelo descargado.")


def cargar_spacy():
    """
    Carga spaCy con manejo explícito de errores de permiso y corrupción.
    Activa NER además de tokenizer/morphologizer/lemmatizer para la mejora 4.
    """
    import spacy

    try:
        nlp = spacy.load(MODELO_SPACY)
    except OSError as e:
        print(f"\n  ❌ No se pudo cargar el modelo {MODELO_SPACY}: {e}")
        print(f"     Solución: python -m spacy download {MODELO_SPACY}")
        sys.exit(1)
    except Exception as e:
        print(f"\n  ❌ Error inesperado al cargar spaCy: {e}")
        sys.exit(1)

    # Activar: tokenizer (siempre activo), lemmatizer, ner
    # morphologizer ayuda a lemmatizer; lo incluimos si está disponible
    pipes_disponibles = [p for p in ["morphologizer", "lemmatizer", "ner"]
                         if p in nlp.pipe_names]
    nlp.select_pipes(enable=pipes_disponibles)

    print(f"  ℹ️  Pipes activos: tokenizer + {', '.join(pipes_disponibles)}")
    return nlp


# ══════════════════════════════════════════════════════════════════════════════
#  TOKENIZACIÓN / NORMALIZACIÓN (integra todas las mejoras anteriores)
# ══════════════════════════════════════════════════════════════════════════════

def limpiar_token_texto(token_text: str) -> str:
    return token_text.strip("¿?¡!.,;:\"'()[]{}…«»—–-").lower()


def preprocesar_texto(texto: str) -> str:
    """
    Preparación previa a spaCy:
      1. Convierte notaciones :emoji: → texto plano
      2. Aplica n-gramas del dominio (texto → texto_con_ngramas)
    """
    if not isinstance(texto, str):
        return ""
    # :cara_enojada: → cara enojada
    texto = re.sub(
        r":([a-záéíóúüñ_]+):",
        lambda m: m.group(1).replace("_", " "),
        texto
    )
    # N-gramas: "seguro social" → "seguro_social"
    texto = aplicar_ngramas(texto)
    return texto


def procesar_batch(
    textos_y_regs: list[tuple[str, dict]],
    nlp,
    stopwords_spacy: set
) -> list[dict]:
    """
    Procesa un lote de (texto_preprocesado, registro_original) usando nlp.pipe().
    nlp.pipe() es 3-5x más rápido que llamar nlp(texto) en un bucle porque
    aprovecha el batch processing interno de spaCy.

    Retorna lista de resultados NLP listos para añadir al registro.
    """
    textos   = [t for t, _ in textos_y_regs]
    registros = [r for _, r in textos_y_regs]

    resultados = []
    # as_tuples=False porque ya tenemos el registro por separado
    for doc, reg in zip(nlp.pipe(textos, batch_size=BATCH_SIZE), registros):
        tokens_raw    = 0
        tokens_finales: list[str] = []

        for token in doc:
            tokens_raw += 1
            texto_token = limpiar_token_texto(token.text)

            if not texto_token:
                continue
            if token.is_punct or token.is_space:
                continue
            if token.like_num:
                continue
            if len(texto_token) < MIN_LEN_TOKEN:
                continue

            # [MEJORA 3] Resolver lema con tabla de excepciones
            lema = resolver_lema(token, texto_token)
            if not lema or len(lema) < MIN_LEN_TOKEN:
                lema = texto_token

            # [MEJORA 1] Términos siempre conservados (negaciones, causales, dominio)
            if lema in TERMINOS_DOMINIO or texto_token in TERMINOS_DOMINIO:
                tokens_finales.append(lema)
                continue

            # Eliminar stopwords (spaCy + extra) — NO aplica a SIEMPRE_CONSERVAR
            if (token.is_stop or lema in stopwords_spacy or lema in STOPWORDS_EXTRA) \
                    and lema not in SIEMPRE_CONSERVAR \
                    and texto_token not in SIEMPRE_CONSERVAR:
                continue

            # Solo letras o tokens con guión bajo (n-gramas)
            if not re.match(r"^[a-záéíóúüñ_]+$", lema):
                continue

            tokens_finales.append(lema)

        # [MEJORA 4] NER
        ner = extraer_entidades_ner(doc)

        # [MEJORA 5] Peso de Reddit
        peso_reddit = calcular_peso_reddit(reg)

        resultados.append({
            "tokens":            tokens_finales,
            "texto_normalizado": " ".join(tokens_finales),
            "tokens_raw":        tokens_raw,
            "tokens_final":      len(tokens_finales),
            "entidades_ner":     ner,           # [MEJORA 4]
            "peso_reddit":       peso_reddit,   # [MEJORA 5]
        })

    return resultados


# ══════════════════════════════════════════════════════════════════════════════
#  INTERFAZ DE ARCHIVO
# ══════════════════════════════════════════════════════════════════════════════

def pedir_archivos() -> tuple[Path, Path]:
    carpeta = Path(__file__).parent
    jsons   = sorted(carpeta.glob("*.json"))
    if not jsons:
        raise FileNotFoundError("No se encontraron archivos .json en la carpeta.")

    print(f"\n{'='*68}")
    print("  NORMALIZACIÓN NLP v2 — PIPELINE DE MODELADO  🏥🔬")
    print(f"{'='*68}\n")
    print("  Archivos .json disponibles:")
    for i, p in enumerate(jsons):
        print(f"    [{i}] {p.name}  ({p.stat().st_size/1024:.1f} KB)")

    while True:
        idx = input("\n  Número del archivo a normalizar: ").strip()
        try:
            entrada = jsons[int(idx)]
            break
        except (ValueError, IndexError):
            print(f"  ❌ Elige entre 0 y {len(jsons)-1}.")

    ts             = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_default = f"dataset_normalizado_{ts}.json"
    print(f"\n  Nombre sugerido: {nombre_default}")
    resp   = input("  Ruta/nombre de salida (Enter = sugerido): ").strip()
    salida = Path(resp) if resp else carpeta / nombre_default
    if salida.suffix != ".json":
        salida = salida.with_suffix(".json")
    if not salida.is_absolute() and len(salida.parts) == 1:
        salida = carpeta / salida
    return entrada, salida


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    instalar_spacy()
    nlp = cargar_spacy()
    stopwords_spacy = nlp.Defaults.stop_words

    entrada, salida = pedir_archivos()

    print(f"\n  Cargando: {entrada.name} ...")
    with open(entrada, "r", encoding="utf-8") as f:
        data = json.load(f)

    registros     = data.get("datos", data) if isinstance(data, dict) else data
    meta_anterior = data.get("meta_enriquecimiento", {}) if isinstance(data, dict) else {}
    total_entrada = len(registros)

    print(f"\n{'='*68}")
    print(f"  Procesando {total_entrada} registros — lotes de {BATCH_SIZE} (nlp.pipe)")
    print(f"  Mejoras activas: negaciones ✅ | n-gramas ✅ | NER ✅ | pesos ✅")
    print(f"{'='*68}")

    normalizados: list[dict] = []
    vocab_global: dict[str, int] = {}
    ngrama_hits:  dict[str, int] = {}   # estadística de n-gramas detectados
    ner_inst_global: Counter      = Counter()
    stats = {
        "tokens_raw_total":   0,
        "tokens_final_total": 0,
        "vaciados":           0,
        "con_ngramas":        0,
        "con_ner_inst":       0,
    }

    # ── Procesamiento por lotes con nlp.pipe() ────────────────────────────────
    batch_actual: list[tuple[str, dict]] = []

    def flush_batch(batch: list[tuple[str, dict]]):
        """Procesa el lote acumulado y agrega resultados a normalizados."""
        if not batch:
            return
        resultados_nlp = procesar_batch(batch, nlp, stopwords_spacy)
        for (_, reg), nlp_res in zip(batch, resultados_nlp):
            if not nlp_res["tokens"]:
                stats["vaciados"] += 1
                return

            stats["tokens_raw_total"]   += nlp_res["tokens_raw"]
            stats["tokens_final_total"] += nlp_res["tokens_final"]

            # Acumular vocabulario
            for tok in nlp_res["tokens"]:
                vocab_global[tok] = vocab_global.get(tok, 0) + 1

            # Estadísticas de n-gramas usados
            for tok in nlp_res["tokens"]:
                if "_" in tok:
                    stats["con_ngramas"] += 1
                    ngrama_hits[tok] = ngrama_hits.get(tok, 0) + 1

            # Estadísticas NER
            if nlp_res["entidades_ner"]["instituciones"]:
                stats["con_ner_inst"] += 1
                for inst in nlp_res["entidades_ner"]["instituciones"]:
                    ner_inst_global[inst] += 1

            normalizados.append({
                **reg,
                "tokens":            nlp_res["tokens"],
                "texto_normalizado": nlp_res["texto_normalizado"],
                "tokens_raw":        nlp_res["tokens_raw"],
                "tokens_final":      nlp_res["tokens_final"],
                "ratio_reduccion":   round(
                    1 - nlp_res["tokens_final"] / max(nlp_res["tokens_raw"], 1), 3
                ),
                "entidades_ner":     nlp_res["entidades_ner"],   # [MEJORA 4]
                "peso_reddit":       nlp_res["peso_reddit"],     # [MEJORA 5]
            })

    for i, reg in enumerate(registros, 1):
        texto_prep = preprocesar_texto(reg.get("comentario", ""))
        batch_actual.append((texto_prep, reg))

        if len(batch_actual) >= BATCH_SIZE:
            flush_batch(batch_actual)
            batch_actual = []
            pct = round(i / total_entrada * 100)
            print(f"  → {i:>5}/{total_entrada}  ({pct}%)...", end="\r")

    flush_batch(batch_actual)   # último lote parcial
    print(f"  → {total_entrada}/{total_entrada} (100%) ✅          ")

    # ── Estadísticas finales ──────────────────────────────────────────────────
    top_terminos   = sorted(vocab_global.items(), key=lambda x: -x[1])[:30]
    top_ngramas    = sorted(ngrama_hits.items(),  key=lambda x: -x[1])[:15]
    top_inst_ner   = ner_inst_global.most_common(10)
    vocab_size     = len(vocab_global)
    total_final    = len(normalizados)
    ratio_prom     = round(
        1 - stats["tokens_final_total"] / max(stats["tokens_raw_total"], 1), 3
    )

    resultado_json = {
        "meta_normalizacion": {
            "generado":                  datetime.now().isoformat(),
            "version_script":            "2.0",
            "archivo_fuente":            str(entrada),
            "pipeline_anterior":         meta_anterior.get("version_script", "desconocido"),
            "modelo_spacy":              MODELO_SPACY,
            "batch_size":                BATCH_SIZE,
            "registros_entrada":         total_entrada,
            "registros_salida":          total_final,
            "registros_vaciados":        stats["vaciados"],
            "tokens_raw_total":          stats["tokens_raw_total"],
            "tokens_final_total":        stats["tokens_final_total"],
            "ratio_reduccion_promedio":  ratio_prom,
            "vocabulario_unico":         vocab_size,
            "registros_con_ngramas":     stats["con_ngramas"],
            "registros_con_ner_inst":    stats["con_ner_inst"],
            "top_30_terminos":           [{"termino": t, "frecuencia": f}
                                          for t, f in top_terminos],
            "top_ngramas_detectados":    [{"ngrama": n, "frecuencia": f}
                                          for n, f in top_ngramas],
            "instituciones_ner_top10":   [{"institucion": i, "menciones": c}
                                          for i, c in top_inst_ner],
            "config": {
                "min_len_token":          MIN_LEN_TOKEN,
                "terminos_dominio":       len(TERMINOS_DOMINIO),
                "negaciones_protegidas":  len(NEGACIONES_PROTEGIDAS),
                "conectores_causales":    len(CONECTORES_CAUSALES),
                "ngramas_definidos":      len(NGRAMAS_DOMINIO),
                "excepciones_lema":       len(EXCEPCIONES_LEMA),
                "stopwords_extra":        len(STOPWORDS_EXTRA),
            },
        },
        "datos": normalizados,
    }

    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(resultado_json, f, ensure_ascii=False, indent=2)

    # ── Reporte ───────────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("  REPORTE DE NORMALIZACIÓN NLP v2")
    print(f"{'='*68}")
    print(f"  Registros procesados           : {total_entrada:>5}")
    print(f"  ├─ Vaciados sin tokens         : -{stats['vaciados']:>4}")
    print(f"  └─ ✅ Registros listos          : {total_final:>5}")
    print(f"\n  Tokens totales (brutos)        : {stats['tokens_raw_total']:>7}")
    pct_cons = round((1 - ratio_prom) * 100, 1)
    print(f"  Tokens tras filtrado           : {stats['tokens_final_total']:>7}  ({pct_cons}% conservado)")
    print(f"  Vocabulario único              : {vocab_size:>7} términos distintos")
    print(f"  Registros con n-gramas activos : {stats['con_ngramas']:>7}")
    print(f"  Registros con entidad NER      : {stats['con_ner_inst']:>7}")

    print(f"\n  🔗 Top 10 n-gramas detectados:")
    for ng, frec in top_ngramas[:10]:
        barra = "█" * min(frec // max(total_final // 30, 1), 22)
        print(f"    {ng:<30} {frec:>4}  {barra}")

    print(f"\n  🏥 Instituciones más mencionadas (NER):")
    for inst, cnt in top_inst_ner:
        barra = "█" * min(cnt // max(total_final // 30, 1), 22)
        print(f"    {inst:<25} {cnt:>4}  {barra}")

    print(f"\n  📊 Top 15 términos del corpus:")
    for termino, frec in top_terminos[:15]:
        barra = "█" * min(frec // max(total_final // 30, 1), 22)
        print(f"    {termino:<25} {frec:>4}  {barra}")

    print(f"\n  ① Negaciones protegidas        : ✅ ({len(NEGACIONES_PROTEGIDAS)} términos)")
    print(f"  ② Conectores causales          : ✅ ({len(CONECTORES_CAUSALES)} términos)")
    print(f"  ③ N-gramas del dominio         : ✅ ({len(NGRAMAS_DOMINIO)} patrones)")
    print(f"  ④ Excepciones de lematización  : ✅ ({len(EXCEPCIONES_LEMA)} entradas)")
    print(f"  ⑤ NER instituciones/lugares    : ✅ (ORG + LOC/GPE)")
    print(f"  ⑥ Pesos Reddit por registro    : ✅ (log-score + profundidad)")
    print(f"  ⑦ Procesamiento nlp.pipe()     : ✅ (lotes de {BATCH_SIZE})")
    print(f"\n  💾 Guardado en: {salida}")
    print(f"{'='*68}")
    print(f"""
  SIGUIENTE PASO → auto_etiquetar.py
  ─────────────────────────────────────────────────────────────────
  Usará pysentimiento (modelo BERT entrenado en tweets MX)
  para asignar etiquetas POS/NEG/NEU a tus {total_final} registros.

  Para vectorización avanzada con embeddings, ver comentario en
  la sección MEJORA 5 de este script (FastText / sentence-transformers).
  El campo "peso_reddit" ya está listo para ponderar la matriz TF-IDF.
""")


if __name__ == "__main__":
    main()

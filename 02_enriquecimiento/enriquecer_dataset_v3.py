#!/usr/bin/env python3
"""
enriquecer_dataset.py v2 — Segunda pasada sobre el JSON producido por limpiar_dataset.py
Tema: Impacto de la ineficiencia del sector salud público MX frente a la influenza

Mejoras v2:
  ① Diccionario de sentimientos LOCALIZADO — pesos de negatividad para regionalismos MX
     (viacrucis, vuelta y vuelta, no hay sistema → negatividad alta garantizada)
  ② Índice de Fricción Burocrática (IFB) — métrica compuesta de "esfuerzo ciudadano"
     Combina: tiempo_espera + pago_privado + dimensión burocrática del sistema
  ③ Intensidad amplificada — ¡!, :cara_enojada:, letras repetidas multiplican polaridad
  ④ Nivel de Impacto en 3 dimensiones: ADMINISTRATIVO / ECONÓMICO / CLÍNICO
  ─────────────────────────────────────────────────────────────────────────────
  Heredadas de v1:
  ⑤ Limpieza profunda residual (artefactos "->" y líneas de ruido)
  ⑥ Re-clasificación de relevancia con vocabulario expandido
  ⑦ Etiqueta de contexto temático (múltiple)
  ⑧ Detección de ironía / sarcasmo
  ⑨ Conversión UTC → fecha legible (ISO + año/mes para series de tiempo)
  ⑩ Métricas de texto: num_palabras, num_caracteres, densidad_keywords

Uso: python3 enriquecer_dataset.py
"""

import json
import re
import sys
from pathlib import Path
from datetime import datetime, timezone

try:
    import emoji as emoji_lib
    EMOJI_DISPONIBLE = True
except ImportError:
    EMOJI_DISPONIBLE = False

MIN_CHARS_POST_ENRIQUECIMIENTO = 25


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA 1 — DICCIONARIO LOCALIZADO CON PESOS DE NEGATIVIDAD
# ══════════════════════════════════════════════════════════════════════════════
#
#  Estructura: { "término": peso_negativo }
#  Escala: 1.0 = negativo leve | 2.0 = negativo moderado | 3.0 = muy negativo
#
#  Por qué un diccionario propio además de modelos generales:
#  · "viacrucis" es neutro en un modelo entrenado con noticias genéricas,
#    pero en Reddit MX de salud = agotamiento total ante el sistema.
#  · "vuelta y vuelta" no aparece en corpus estándar de sentimientos.
#  · "particular" puede ser positivo en otro contexto, pero aquí = falla sistémica.
#  · Los términos de jerga coloquial MX ("desmadre", "caos") tienen alta carga
#    negativa que los modelos BERT/BETO suelen subestimar.

LEXICON_NEGATIVO_LOCALIZADO: dict[str, float] = {
    # ── Burocracia / agotamiento (negatividad ALTA) ─────────────────────────
    "viacrucis":           3.0,   # narrativa de sufrimiento extremo ante el sistema
    "odisea":              2.5,   # similar al anterior, más literario
    "vuelta y vuelta":     3.0,   # frustración burocrática repetida; sin equivalente EN
    "no hay sistema":      2.8,   # queja clásica: sistema digital caído en instituciones
    "sistema caído":       2.5,
    "sistema caido":       2.5,
    "vuelva mañana":       2.8,   # "vuelva mañana" = rechazo institucional sistemático
    "vuelva manana":       2.8,
    "a veces funciona":    1.8,   # irónico; contexto siempre negativo en salud MX
    "fuera de servicio":   2.0,
    # ── Pago privado como señal de falla sistémica ───────────────────────────
    "particular":          2.0,   # "tuve que ir al particular" = sistema falló
    "de mi bolsillo":      2.5,
    "pagar de mi":         2.5,
    "médico particular":   2.0,   # "el IMSS no tenía, fui al médico particular"
    "medico particular":   2.0,
    # ── Desabasto / suministro ────────────────────────────────────────────────
    "desabasto":           2.5,
    "sin medicamento":     2.8,
    "no hay vacuna":       2.8,
    "no hay medicamento":  2.8,
    "agotado":             2.0,
    "se acabaron":         2.0,
    "anaquel vacío":       2.2,
    "anaquel vacio":       2.2,
    "no surtieron":        2.5,
    # ── Consecuencias clínicas (negatividad MÁS ALTA) ────────────────────────
    "murió":               3.0,
    "murio":               3.0,
    "se nos murió":        3.0,
    "falleció":            3.0,
    "fallecio":            3.0,
    "muerte":              3.0,
    "intubado":            2.8,
    "uci":                 2.5,
    "neumonía":            2.5,
    "neumonia":            2.5,
    "empeoró":             2.5,
    "empeoro":             2.5,
    "se complicó":         2.5,
    "se complico":         2.5,
    "recayó":              2.2,
    "recayo":              2.2,
    # ── Negligencia / maltrato ────────────────────────────────────────────────
    "negligencia":         2.8,
    "negligente":          2.8,
    "mala atención":       2.5,
    "mala atencion":       2.5,
    "no me atendieron":    2.5,
    "me mandaron a casa":  2.5,
    "no diagnosticaron":   2.5,
    "maltrato":            2.5,
    "grosero":             2.0,
    "inútiles":            2.5,
    "inutiles":            2.5,
    # ── Jerga coloquial de indignación MX ────────────────────────────────────
    "desmadre":            2.5,   # caos total; intensamente negativo en contexto de salud
    "cochinero":           2.0,
    "caos":                2.0,
    "chingadera":          2.5,
    "nos dejan solos":     3.0,
    "abandonados":         2.8,
    "nos fallan":          2.8,
    # ── Corrupción ───────────────────────────────────────────────────────────
    "moche":               2.8,
    "mordida":             2.8,
    "soborno":             2.8,
    "corrupción":          2.5,
    "corrupcion":          2.5,
    "palancas":            2.2,
    "robo":                2.5,
    "robaron":             2.5,
    # ── Términos ambivalentes con negatividad alta EN ESTE CONTEXTO ──────────
    "recorte":             2.0,   # neutro en economía, negativo aquí
    "austeridad":          2.0,   # idem
    "privatización":       2.2,
    "privatizacion":       2.2,
    "escasez":             2.2,
    "insuficiente":        2.0,
    "deficiente":          2.0,
    "pésimo":              2.5,
    "pesimo":              2.5,
}

# Términos que actúan como AMPLIFICADORES del lexicon (multiplicadores)
# Se suman si aparecen en el mismo texto
AMPLIFICADORES_CONTEXTUALES = {
    "muy":         1.1,
    "tan":         1.1,
    "demasiado":   1.2,
    "absolutamente": 1.2,
    "totalmente":  1.2,
    "jamás":       1.3,
    "jamas":       1.3,
    "nunca":       1.2,
    "siempre":     1.1,   # "siempre es un desastre"
    "todo":        1.1,
}


# ══════════════════════════════════════════════════════════════════════════════
#  VOCABULARIO EXPANDIDO (heredado de v1, con adiciones)
# ══════════════════════════════════════════════════════════════════════════════

INDICADORES_PROBLEMA_EXT = [
    "desabasto", "escasez", "no hay", "sin medicamento", "agotado",
    "fila", "cola", "horas", "espera", "saturad",
    "moche", "soborno", "corrupción", "corrupcion",
    "recorte", "austeridad", "privatización", "privatizacion",
    "no funciona", "pésimo", "pesimo", "malo", "deficiente",
    "tardaron", "tardó", "tardo", "rechaz", "negaron",
    "murió", "murio", "falleció", "fallecio", "muerte",
    "negligencia", "negligente", "incapaz", "incompetente",
    "no atienden", "no tienen", "no contaban", "insuficiente",
    "captcha", "error 404", "no sirve", "no carga", "sistema caído",
    "sistema caido", "fuera de servicio", "no disponible", "sin sistema",
    "trámite", "tramite", "burocracia", "burocrático", "burocratico",
    "requisito", "formulario", "ventanilla", "turno", "cita",
    "no me dieron cita", "cancelaron", "rechazaron", "denegaron",
    "robo", "robaron", "desaparecieron", "nunca llegaron", "nunca existieron",
    "se los robaron", "desvío", "desvio", "malversación", "malversacion",
    "sin presupuesto", "falta de", "carencia", "abandono",
    "desmadre", "madrazo", "chingadera", "caos", "cochinero",
    "a la chingada", "no sirven", "inútiles", "inutiles",
    "nos fallan", "nos dejan", "solos", "abandonados",
    "se nos murió", "se nos murio", "se murió", "se murio",
    # v2
    "viacrucis", "odisea", "vuelta y vuelta", "vuelva mañana", "vuelva manana",
    "particular", "de mi bolsillo", "médico particular", "medico particular",
    "empeoró", "empeoro", "se complicó", "se complico", "recayó", "recayo",
    "intubado", "uci", "neumonía", "neumonia",
]

SUSTANTIVOS_SALUD_EXT = [
    "imss", "issste", "insabi", "hospital", "clínica", "clinica",
    "médico", "medico", "doctor", "consulta", "urgencias",
    "medicamento", "medicina", "vacuna", "tratamiento",
    "seguro social", "centro de salud", "salubridad",
    "sistema de salud", "sector salud", "secretaría de salud",
    "secretaria de salud", "seguro popular", "área de salud",
    "area de salud", "insumos", "farmacia", "farmacéutico",
    "umf", "clínica familiar", "la raza", "siglo xxi",
]

CONTEXTOS: dict[str, list[str]] = {
    "desabasto_medicamentos": [
        "desabasto", "escasez", "no hay medicamento", "sin medicamento",
        "no surtieron", "agotado", "no tienen", "falta de medicina",
        "robo de medicamento", "medicamentos oncológicos",
    ],
    "burocracia_sistema": [
        "captcha", "error 404", "trámite", "tramite", "formulario",
        "sistema", "plataforma", "ventanilla", "requisito", "burocracia",
        "proceso", "reporte", "vuelva mañana", "vuelva manana",
        "vuelta y vuelta", "papeleo",
    ],
    "vacunacion": [
        "vacuna", "vacunar", "vacunación", "vacunacion", "inmunización",
        "inmunizacion", "antiviral", "refuerzo", "dosis",
    ],
    "atencion_medica": [
        "consulta", "urgencias", "emergencias", "cita", "médico", "medico",
        "doctor", "enfermera", "hospitalizado", "internad",
        "sala de espera", "triage", "atención", "atencion",
    ],
    "influenza_enfermedad": [
        "influenza", "gripe", "gripa", "h1n1", "h3n2",
        "fiebre", "calentura", "tos", "síntoma", "sintoma", "contagio",
    ],
    "costos_financiamiento": [
        "presupuesto", "recorte", "austeridad", "privatización", "privatizacion",
        "pago", "cobro", "caro", "costoso", "seguro popular",
        "deducible", "copago", "sin dinero", "de mi bolsillo", "particular",
    ],
    "prevencion_higiene": [
        "cubrebocas", "mascarilla", "gel antibacterial", "lavarse las manos",
        "higiene", "prevención", "prevencion", "cuarentena", "aislamiento",
    ],
    "corrupcion": [
        "moche", "soborno", "corrupción", "corrupcion", "robo", "robaron",
        "desaparecieron", "desvío", "desvio", "malversación", "malversacion",
        "se los robaron", "palancas",
    ],
}

PATRONES_SARCASMO = [
    r"qué raro",  r"que raro", r"sorpresa", r"increíble", r"increible",
    r"vaya sorpresa", r"no me digas", r"¿en serio\?", r"en serio\?",
    r"captcha\s*\d", r"error 404",
    r"de pendejos", r"como si", r"claro que sí", r"claro que si",
    r"muy parecido al danés", r"muy parecido al danes",
    r"funciona perfecto", r"excelente servicio", r"gran trabajo",
    r"\.\.\.", r"jajaja", r"jeje", r"lol",
]
PATRON_SARCASMO_COMPILADO = re.compile("|".join(PATRONES_SARCASMO), re.IGNORECASE)


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA 2 — ÍNDICE DE FRICCIÓN BUROCRÁTICA (IFB)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Métrica compuesta que cuantifica el "esfuerzo" que el ciudadano
#  tuvo que hacer para navegar el sistema de salud.
#  Escala: 0.0 (sin fricción detectada) → 10.0 (fricción máxima)
#
#  Componentes:
#    A) Tiempo y burocracia:   palabras de espera, citas lejanas, trámites
#    B) Pago privado:          tuvo que salir del sistema público
#    C) Rechazo/negación:      le negaron atención o medicamento
#    D) Múltiples intentos:    indicadores de que fue más de una vez
#    E) Impacto clínico:       la fricción tuvo consecuencias en la salud

IFB_COMPONENTES: dict[str, dict] = {
    "tiempo_espera": {
        "peso_base": 1.5,
        "terminos": [
            "fila", "cola", "horas de espera", "espera", "tardaron",
            "cita", "meses", "semanas", "turno", "sala de espera",
            "todo el día", "desde temprano", "madrugada",
        ],
    },
    "burocracia_proceso": {
        "peso_base": 1.5,
        "terminos": [
            "trámite", "tramite", "papeleo", "requisito", "formulario",
            "ventanilla", "vuelva mañana", "vuelva manana",
            "vuelta y vuelta", "viacrucis", "odisea", "sistema caído",
            "sistema caido", "no hay sistema", "fuera de servicio",
        ],
    },
    "pago_privado": {
        "peso_base": 2.0,   # mayor peso: el sistema falló tanto que la persona pagó
        "terminos": [
            "particular", "privado", "privada", "médico particular",
            "medico particular", "de mi bolsillo", "pagar de mi",
            "gasté", "gaste", "pagué", "pague", "tuve que comprar",
            "farmacia particular", "clínica privada",
        ],
    },
    "rechazo_negacion": {
        "peso_base": 2.0,
        "terminos": [
            "no me atendieron", "me negaron", "rechazaron", "negaron",
            "no me dieron", "me mandaron a casa", "no había cupo",
            "sin cita", "no tenían", "no había", "no contaban",
        ],
    },
    "multiples_intentos": {
        "peso_base": 1.5,
        "terminos": [
            "vuelta y vuelta", "varias veces", "dos veces", "tres veces",
            "otra vez", "de nuevo", "segunda vez", "regresé", "regrese",
            "volví", "volvi", "intenté de nuevo", "intente de nuevo",
        ],
    },
    "impacto_salud_por_friccion": {
        "peso_base": 3.0,   # máximo peso: la fricción causó daño clínico
        "terminos": [
            "empeoró por la espera", "empeoro por la espera",
            "mientras esperaba", "por no atenderme", "a tiempo no",
            "tarde", "llegó tarde", "llego tarde",
            "se complicó", "se complico", "por eso se complicó",
        ],
    },
}

IFB_MAX = 10.0   # techo de la escala


def calcular_ifb(texto: str) -> dict:
    """
    Calcula el Índice de Fricción Burocrática (IFB).

    Retorna:
      {
        "ifb_score":      float  ← score total 0.0–10.0
        "ifb_nivel":      str    ← "bajo" | "medio" | "alto" | "crítico"
        "ifb_componentes": dict  ← qué componentes se activaron y con qué términos
      }

    Metodología:
      - Cada componente aporta su peso_base si AL MENOS UN término se detecta.
      - El score se acumula y se normaliza al techo IFB_MAX.
      - Un comentario que activa pago_privado + rechazo_negacion + impacto_salud
        llega casi al máximo (7.0 / 10.0 antes de normalizar).
    """
    t = texto.lower()
    score_acumulado = 0.0
    componentes_activos: dict[str, list[str]] = {}

    for nombre, cfg in IFB_COMPONENTES.items():
        terminos_encontrados = [term for term in cfg["terminos"] if term in t]
        if terminos_encontrados:
            # El peso base se gana con 1 término; términos adicionales suman 20% cada uno
            extras = (len(terminos_encontrados) - 1) * 0.2
            aporte = cfg["peso_base"] * (1 + extras)
            score_acumulado += aporte
            componentes_activos[nombre] = terminos_encontrados

    # Normalizar a 0–10
    score_normalizado = round(min(score_acumulado, IFB_MAX), 2)

    if score_normalizado == 0:
        nivel = "ninguno"
    elif score_normalizado < 2.5:
        nivel = "bajo"
    elif score_normalizado < 5.0:
        nivel = "medio"
    elif score_normalizado < 7.5:
        nivel = "alto"
    else:
        nivel = "crítico"

    return {
        "ifb_score":       score_normalizado,
        "ifb_nivel":       nivel,
        "ifb_componentes": componentes_activos,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA 3 — POLARIDAD LOCALIZADA CON INTENSIDAD AMPLIFICADA
# ══════════════════════════════════════════════════════════════════════════════
#
#  Combina el lexicon localizado con tres capas de amplificación:
#    · Signos de exclamación ¡!  → multiplica negatividad
#    · Emojis de enojo/tristeza  → suma carga emocional
#    · Letras repetidas residuales (énfasis emocional no normalizado)
#
#  El score resultante es deliberadamente simple (no reemplaza BERT/BETO)
#  pero captura la carga negativa del español coloquial mexicano en salud
#  mejor que cualquier modelo entrenado en corpus genérico.

EMOJIS_NEGATIVOS_TEXTO = [
    # Nombres que produce emoji.demojize(language="es")
    ":cara_enojada:", ":cara_muy_enojada:", ":cara_con_símbolos_en_la_boca:",
    ":cara_llorando:", ":cara_llorando_fuerte:", ":cara_disgustada:",
    ":cara_que_vomita:", ":cara_triste:", ":cara_con_cejas_fruncidas:",
    ":cara_decepcionada:", ":cara_angustiada:", ":cara_asustada:",
    # Por si vienen sin demojize (Unicode directo como texto)
    "😡", "🤬", "😤", "😢", "😭", "😞", "😟", "😠", "😰", "😱",
]

EMOJIS_NEGATIVOS_PESO = 0.4   # cada emoji negativo suma este valor al score


def calcular_polaridad_localizada(texto: str, es_sarcasmo: bool = False) -> dict:
    """
    Calcula la polaridad negativa usando el lexicon localizado y amplificadores.

    Retorna:
      {
        "polaridad_score":    float   ← 0.0 (neutro) → positivo = negativo
        "polaridad_nivel":    str     ← "neutro" | "leve" | "moderado" | "alto" | "muy_alto"
        "terminos_activos":   list    ← términos del lexicon que se activaron
        "amplificadores":     dict    ← qué amplificadores se detectaron
        "ajuste_sarcasmo":    bool    ← si se aplicó corrección de sarcasmo
      }

    Nota sobre sarcasmo:
      Cuando es_sarcasmo=True, el score se multiplica x1.3 porque el sarcasmo
      suele ocultar una negatividad mayor que las palabras literales expresan.
    """
    t        = texto.lower()
    score    = 0.0
    terminos_activos: list[str] = []
    amplificadores_activos: dict[str, float] = {}

    # ── Capa 1: Lexicon localizado ────────────────────────────────────────────
    for termino, peso in LEXICON_NEGATIVO_LOCALIZADO.items():
        if termino in t:
            score += peso
            terminos_activos.append(termino)

    # ── Capa 2: Amplificadores contextuales ──────────────────────────────────
    factor_amp = 1.0
    for amp, factor in AMPLIFICADORES_CONTEXTUALES.items():
        if amp in t:
            factor_amp = max(factor_amp, factor)   # toma el mayor (no acumula exponencialmente)
            amplificadores_activos[amp] = factor
    score *= factor_amp

    # ── Capa 3: Signos de exclamación ¡! ─────────────────────────────────────
    n_excl = texto.count("!") + texto.count("¡")
    if n_excl > 0:
        # Cada par de signos suma 10% (con techo en 50% para evitar inflación)
        factor_excl = min(1 + (n_excl * 0.1), 1.5)
        score *= factor_excl
        if factor_excl > 1.0:
            amplificadores_activos["signos_exclamacion"] = round(factor_excl, 2)

    # ── Capa 4: Emojis negativos ──────────────────────────────────────────────
    emojis_encontrados = [e for e in EMOJIS_NEGATIVOS_TEXTO if e in texto]
    if emojis_encontrados:
        score += len(emojis_encontrados) * EMOJIS_NEGATIVOS_PESO
        amplificadores_activos["emojis_negativos"] = emojis_encontrados

    # ── Capa 5: Letras repetidas residuales (énfasis no normalizado) ──────────
    # Ej: si el texto pasó por comentario_raw (sin normalizar) pueden quedar
    # "pésimoooo" o "maaalooo". Detectamos este patrón y sumamos
    if re.search(r"(.)\1{2,}", texto, re.IGNORECASE):
        score *= 1.1
        amplificadores_activos["enfasis_letras_repetidas"] = 1.1

    # ── Capa 6: Ajuste por sarcasmo ───────────────────────────────────────────
    ajuste_sarcasmo = False
    if es_sarcasmo and score > 0:
        score *= 1.3
        ajuste_sarcasmo = True

    score = round(score, 3)

    if score == 0:
        nivel = "neutro"
    elif score < 2.0:
        nivel = "leve"
    elif score < 5.0:
        nivel = "moderado"
    elif score < 9.0:
        nivel = "alto"
    else:
        nivel = "muy_alto"

    return {
        "polaridad_score":  score,
        "polaridad_nivel":  nivel,
        "terminos_activos": terminos_activos,
        "amplificadores":   amplificadores_activos,
        "ajuste_sarcasmo":  ajuste_sarcasmo,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA 4 — NIVEL DE IMPACTO EN 3 DIMENSIONES
# ══════════════════════════════════════════════════════════════════════════════
#
#  Jerarquía de gravedad creciente:
#    ADMINISTRATIVO — El ciudadano sufrió fricción pero sin consecuencias físicas/económicas
#    ECONÓMICO      — Tuvo que gastar dinero porque el sistema falló
#    CLÍNICO        — La falla del sistema afectó directamente su salud o la de un familiar

IMPACTO_ADMINISTRATIVO = [
    # Citas y acceso
    "cita", "turno", "fila", "cola", "sala de espera", "sin cupo",
    "no había cita", "me cancelaron", "no me dieron cita",
    "vuelva mañana", "vuelva manana",
    # Trámites y sistema
    "trámite", "tramite", "papeleo", "formulario", "requisito",
    "ventanilla", "vuelta y vuelta", "sistema caído", "no hay sistema",
    "viacrucis", "odisea",
    # Carnets y documentos
    "carnet", "cartilla", "credencial", "expediente", "no tienen registro",
    # Trato
    "maltrato", "grosero", "me ignoraron", "no me hicieron caso",
    "mal trato", "falta de información",
]

IMPACTO_ECONOMICO = [
    # Salida del sistema público
    "particular", "privado", "privada", "médico particular", "medico particular",
    "clínica privada", "clinica privada", "hospital privado",
    # Pago de bolsillo
    "de mi bolsillo", "pagar de mi", "gasté", "gaste", "pagué", "pague",
    "tuve que comprar", "compré el medicamento", "compre el medicamento",
    # Farmacia particular
    "farmacia particular", "compré en farmacia", "compre en farmacia",
    "botica", "me costó", "me costo", "caro", "costoso",
    # Deuda / consecuencia económica
    "me endeudé", "me endeude", "préstamo", "prestamo", "empeñé", "empene",
    "sin dinero", "no tengo para", "no alcanza",
]

IMPACTO_CLINICO = [
    # Complicaciones directas por falla del sistema
    "empeoró", "empeoro", "se complicó", "se complico",
    "mientras esperaba se", "por no atenderme", "por la demora",
    "llegó tarde el médico", "llego tarde el medico",
    "a tiempo hubiera", "si hubieran atendido",
    # Enfermedades secundarias
    "neumonía", "neumonia", "bronquitis", "sepsis", "complicación pulmonar",
    "hospitalizado", "internado", "uci", "terapia intensiva",
    "intubado", "ventilador",
    # Fallecimiento
    "murió", "murio", "falleció", "fallecio", "muerte", "fallecimiento",
    "se nos fue", "perdimos a", "ya no está",
    "no se curó", "no se curo", "recayó", "recayo",
    # Secuelas
    "secuelas", "quedó mal", "quedo mal", "daño permanente",
    "ya no se recuperó", "ya no se recupero",
]


def clasificar_nivel_impacto(texto: str) -> dict:
    """
    Clasifica el impacto de la falla del sistema en tres dimensiones.
    Un comentario puede tener impacto en más de una dimensión.

    Retorna:
      {
        "dimensiones_activas":  list[str]   ← ["administrativo", "economico", "clinico"]
        "nivel_maximo":         str         ← la dimensión de mayor gravedad
        "terminos_por_dim":     dict        ← qué términos activaron cada dimensión
        "escala_gravedad":      int         ← 0 = ninguno, 1 = admin, 2 = econ, 3 = clínico
      }

    La escala_gravedad permite ordenar el corpus de menos a más grave
    para análisis comparativos y visualizaciones de escalada de crisis.
    """
    t = texto.lower()
    dimensiones: dict[str, list[str]] = {}

    terminos_admin = [term for term in IMPACTO_ADMINISTRATIVO if term in t]
    terminos_econ  = [term for term in IMPACTO_ECONOMICO      if term in t]
    terminos_clin  = [term for term in IMPACTO_CLINICO        if term in t]

    if terminos_admin:
        dimensiones["administrativo"] = terminos_admin
    if terminos_econ:
        dimensiones["economico"]      = terminos_econ
    if terminos_clin:
        dimensiones["clinico"]        = terminos_clin

    # Nivel máximo: clínico > económico > administrativo > ninguno
    if "clinico" in dimensiones:
        nivel_max   = "clinico"
        escala      = 3
    elif "economico" in dimensiones:
        nivel_max   = "economico"
        escala      = 2
    elif "administrativo" in dimensiones:
        nivel_max   = "administrativo"
        escala      = 1
    else:
        nivel_max   = "ninguno"
        escala      = 0

    return {
        "dimensiones_activas": list(dimensiones.keys()),
        "nivel_maximo":        nivel_max,
        "terminos_por_dim":    dimensiones,
        "escala_gravedad":     escala,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FUNCIONES HEREDADAS DE v1
# ══════════════════════════════════════════════════════════════════════════════

def limpiar_profundo(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    if texto.count("->") >= 3:
        nodos = [n.strip() for n in texto.split("->") if n.strip() and len(n.strip()) > 2]
        nodos_limpios = [n for n in nodos if n not in ("...", "…")][:6]
        if nodos_limpios:
            texto = f"[flujo burocrático: {' → '.join(nodos_limpios)}]" + \
                    (" → [Error 404]" if "error 404" in texto.lower() else "")
    texto = re.sub(r"^[\.\-_\*]{3,}$", "", texto, flags=re.MULTILINE)
    texto = re.sub(r"\.{3,}", "…", texto)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    return texto.strip()


def emojis_a_texto(texto: str) -> str:
    if not EMOJI_DISPONIBLE or not isinstance(texto, str):
        return texto
    return emoji_lib.demojize(texto, language="es")


def reclasificar_relevancia(texto: str) -> str:
    t = texto.lower()
    tiene_sustantivo = any(s in t for s in SUSTANTIVOS_SALUD_EXT)
    tiene_problema   = any(p in t for p in INDICADORES_PROBLEMA_EXT)
    tiene_or_basico  = tiene_sustantivo or tiene_problema or any([
        "influenza" in t, "gripe" in t, "gripa" in t, "h1n1" in t,
        "vacuna" in t, "imss" in t, "issste" in t, "hospital" in t,
        "médico" in t, "medico" in t, "desabasto" in t,
    ])
    if not tiene_or_basico:
        return "ninguna"
    if tiene_sustantivo and tiene_problema:
        return "alta"
    return "media"


def detectar_contextos(texto: str) -> list[str]:
    t = texto.lower()
    encontrados = [ctx for ctx, palabras in CONTEXTOS.items()
                   if any(p in t for p in palabras)]
    return encontrados if encontrados else ["general"]


def detectar_sarcasmo(texto: str) -> bool:
    return bool(PATRON_SARCASMO_COMPILADO.search(texto))


def utc_a_fecha(timestamp) -> dict:
    if not timestamp:
        return {"fecha_iso": None, "anio": None, "mes": None, "anio_mes": None}
    try:
        dt = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
        return {
            "fecha_iso":  dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "anio":       dt.year,
            "mes":        dt.month,
            "anio_mes":   dt.strftime("%Y-%m"),
        }
    except (ValueError, OSError):
        return {"fecha_iso": None, "anio": None, "mes": None, "anio_mes": None}


def calcular_metricas(texto: str, contextos: list[str]) -> dict:
    palabras      = texto.split()
    todas_keywords = [kw for lista in CONTEXTOS.values() for kw in lista]
    t_lower       = texto.lower()
    hits          = sum(1 for kw in todas_keywords if kw in t_lower)
    return {
        "num_palabras":      len(palabras),
        "num_caracteres":    len(texto),
        "num_contextos":     len(contextos),
        "densidad_keywords": round(hits / max(len(palabras), 1), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA / GUARDADO / INTERFAZ
# ══════════════════════════════════════════════════════════════════════════════

def pedir_archivos() -> tuple[Path, Path]:
    carpeta = Path(__file__).parent
    jsons   = sorted(carpeta.glob("*.json"))
    if not jsons:
        raise FileNotFoundError("No se encontraron archivos .json en la carpeta.")

    print(f"\n{'='*68}")
    print("  ENRIQUECIMIENTO DE DATASET v2 — POST-LIMPIEZA  🏥📊")
    print(f"{'='*68}\n")
    print("  Archivos .json disponibles:")
    for i, p in enumerate(jsons):
        print(f"    [{i}] {p.name}  ({p.stat().st_size/1024:.1f} KB)")

    while True:
        idx = input("\n  Número del archivo a enriquecer: ").strip()
        try:
            entrada = jsons[int(idx)]
            break
        except (ValueError, IndexError):
            print(f"  ❌ Número inválido. Elige entre 0 y {len(jsons)-1}.")

    ts             = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_default = f"dataset_enriquecido_{ts}.json"
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
    if not EMOJI_DISPONIBLE:
        print("\n  ⚠️  Librería 'emoji' no encontrada.")
        resp = input("  ¿Instalarla ahora? (s/n): ").strip().lower()
        if resp == "s":
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "emoji", "-q"], check=True)
            try:
                import emoji as emoji_lib
                globals()["EMOJI_DISPONIBLE"] = True
                globals()["emoji_lib"]        = emoji_lib
                print("  ✅ emoji instalado.")
            except ImportError:
                pass

    entrada, salida = pedir_archivos()
    print(f"\n  Cargando: {entrada.name} ...")
    with open(entrada, "r", encoding="utf-8") as f:
        data = json.load(f)

    registros    = data.get("datos", data) if isinstance(data, dict) else data
    meta_anterior = data.get("meta_limpieza", {}) if isinstance(data, dict) else {}

    stats = {
        "entrada":             len(registros),
        "relevancia_cambiada": 0,
        "sarcasmo_detectado":  0,
        "texto_profundo_mod":  0,
        "eliminados_vacios":   0,
        "contextos_total":     {},
        "relevancia_final":    {"alta": 0, "media": 0, "ninguna": 0},
        # v2
        "ifb_niveles":         {"ninguno": 0, "bajo": 0, "medio": 0, "alto": 0, "crítico": 0},
        "polaridad_niveles":   {"neutro": 0, "leve": 0, "moderado": 0, "alto": 0, "muy_alto": 0},
        "impacto_niveles":     {"ninguno": 0, "administrativo": 0, "economico": 0, "clinico": 0},
    }

    enriquecidos: list[dict] = []

    print(f"\n{'='*68}")
    print("  EJECUTANDO ENRIQUECIMIENTO v2...")
    print(f"  Nuevas métricas: IFB · Polaridad localizada · Nivel de impacto")
    print(f"{'='*68}")

    for reg in registros:
        comentario = reg.get("comentario", "")

        # ① Emojis → texto (segunda oportunidad)
        if EMOJI_DISPONIBLE:
            comentario = emojis_a_texto(comentario)

        # ② Limpieza profunda residual
        comentario_profundo = limpiar_profundo(comentario)
        if comentario_profundo != comentario:
            stats["texto_profundo_mod"] += 1
        comentario = comentario_profundo

        # ③ Descartar si quedó vacío
        if len(comentario.strip()) < MIN_CHARS_POST_ENRIQUECIMIENTO:
            stats["eliminados_vacios"] += 1
            continue

        # ④ Re-clasificar relevancia
        relevancia_anterior = reg.get("relevancia", "media")
        relevancia_nueva    = reclasificar_relevancia(comentario)
        if relevancia_nueva == "ninguna":
            stats["eliminados_vacios"] += 1
            continue
        if relevancia_nueva != relevancia_anterior:
            stats["relevancia_cambiada"] += 1
        stats["relevancia_final"][relevancia_nueva] += 1

        # ⑤ Contextos temáticos
        contextos = detectar_contextos(comentario)
        for c in contextos:
            stats["contextos_total"][c] = stats["contextos_total"].get(c, 0) + 1

        # ⑥ Sarcasmo
        es_sarcasmo = detectar_sarcasmo(comentario)
        if es_sarcasmo:
            stats["sarcasmo_detectado"] += 1

        # ⑦ Fechas
        fecha_comentario = utc_a_fecha(reg.get("comentario_utc"))
        fecha_post       = utc_a_fecha(reg.get("post_creado_utc"))

        # ⑧ Métricas básicas
        metricas = calcular_metricas(comentario, contextos)

        # ══ NUEVAS MÉTRICAS v2 ════════════════════════════════════════════════

        # [MEJORA 2] Índice de Fricción Burocrática
        ifb = calcular_ifb(comentario)
        stats["ifb_niveles"][ifb["ifb_nivel"]] = \
            stats["ifb_niveles"].get(ifb["ifb_nivel"], 0) + 1

        # [MEJORA 3] Polaridad localizada con intensidad
        polaridad = calcular_polaridad_localizada(comentario, es_sarcasmo=es_sarcasmo)
        stats["polaridad_niveles"][polaridad["polaridad_nivel"]] = \
            stats["polaridad_niveles"].get(polaridad["polaridad_nivel"], 0) + 1

        # [MEJORA 4] Nivel de impacto en 3 dimensiones
        impacto = clasificar_nivel_impacto(comentario)
        stats["impacto_niveles"][impacto["nivel_maximo"]] = \
            stats["impacto_niveles"].get(impacto["nivel_maximo"], 0) + 1

        # ─────────────────────────────────────────────────────────────────────
        enriquecidos.append({
            # Identificadores
            "post_id":               reg.get("post_id", ""),
            "subreddit":             reg.get("subreddit", ""),
            "url_post":              reg.get("url_post", ""),
            # Contenido
            "titulo_post":           reg.get("titulo_post", ""),
            "comentario_raw":        reg.get("comentario_raw", ""),
            "comentario":            comentario,
            # Clasificación
            "relevancia":            relevancia_nueva,
            "contextos":             contextos,
            "posible_sarcasmo":      es_sarcasmo,
            # ── MEJORA 2: IFB ────────────────────────────────────────────────
            "ifb_score":             ifb["ifb_score"],
            "ifb_nivel":             ifb["ifb_nivel"],
            "ifb_componentes":       ifb["ifb_componentes"],
            # ── MEJORA 3: Polaridad localizada ───────────────────────────────
            "polaridad_score":       polaridad["polaridad_score"],
            "polaridad_nivel":       polaridad["polaridad_nivel"],
            "polaridad_terminos":    polaridad["terminos_activos"],
            "polaridad_amplif":      polaridad["amplificadores"],
            "polaridad_sarcasmo_aj": polaridad["ajuste_sarcasmo"],
            # ── MEJORA 4: Nivel de impacto ────────────────────────────────────
            "impacto_dimensiones":   impacto["dimensiones_activas"],
            "impacto_nivel_max":     impacto["nivel_maximo"],
            "impacto_escala":        impacto["escala_gravedad"],   # 0-3 para ordenar
            "impacto_terminos":      impacto["terminos_por_dim"],
            # Autor y comunidad
            "autor":                 reg.get("autor", ""),
            "score_comentario":      reg.get("score_comentario", 0),
            "score_post":            reg.get("score_post", 0),
            "num_comentarios":       reg.get("num_comentarios", 0),
            "profundidad":           reg.get("profundidad", 0),
            # Fechas
            "comentario_utc":        reg.get("comentario_utc"),
            "post_creado_utc":       reg.get("post_creado_utc"),
            "fecha_comentario":      fecha_comentario["fecha_iso"],
            "anio_mes_comentario":   fecha_comentario["anio_mes"],
            "anio_comentario":       fecha_comentario["anio"],
            "mes_comentario":        fecha_comentario["mes"],
            "fecha_post":            fecha_post["fecha_iso"],
            # Métricas básicas
            "num_palabras":          metricas["num_palabras"],
            "num_caracteres":        metricas["num_caracteres"],
            "densidad_keywords":     metricas["densidad_keywords"],
            # Trazabilidad
            "query_origen":          reg.get("query_origen", ""),
            # Contexto de limpieza (heredado de v3 del limpiador)
            "palabras_clave_activas": reg.get("palabras_clave_activas", {}),
        })

    total_final = len(enriquecidos)

    resultado = {
        "meta_enriquecimiento": {
            "generado":              datetime.now().isoformat(),
            "version_script":        "2.0",
            "archivo_fuente":        str(entrada),
            "pipeline_anterior":     meta_anterior.get("version_script", "desconocido"),
            "emoji_convertido":      EMOJI_DISPONIBLE,
            "registros_entrada":     stats["entrada"],
            "registros_salida":      total_final,
            "eliminados_post_enriq": stats["eliminados_vacios"],
            "relevancia_cambiada":   stats["relevancia_cambiada"],
            "sarcasmo_detectado":    stats["sarcasmo_detectado"],
            "textos_limpieza_prof":  stats["texto_profundo_mod"],
            "distribucion_relevancia":  stats["relevancia_final"],
            "distribucion_contextos":   dict(sorted(
                stats["contextos_total"].items(), key=lambda x: -x[1])),
            # v2
            "distribucion_ifb":         stats["ifb_niveles"],
            "distribucion_polaridad":   stats["polaridad_niveles"],
            "distribucion_impacto":     stats["impacto_niveles"],
        },
        "datos": enriquecidos,
    }

    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    # ── Reporte ───────────────────────────────────────────────────────────────
    print(f"\n{'='*68}")
    print("  REPORTE DE ENRIQUECIMIENTO v2")
    print(f"{'='*68}")
    print(f"  Registros entrada                      : {stats['entrada']:>5}")
    print(f"  ├─ Eliminados (vacíos/irrelevantes)    : -{stats['eliminados_vacios']:>4}")
    print(f"  ├─ Relevancia re-clasificada           : {stats['relevancia_cambiada']:>5}")
    print(f"  ├─ Textos con limpieza profunda        : {stats['texto_profundo_mod']:>5}")
    print(f"  └─ Posible sarcasmo detectado          : {stats['sarcasmo_detectado']:>5}")
    print(f"  {'─'*54}")
    print(f"  ✅ Registros enriquecidos               : {total_final:>5}")
    print(f"     ├─ Relevancia ALTA                  : {stats['relevancia_final']['alta']:>5}")
    print(f"     └─ Relevancia MEDIA                 : {stats['relevancia_final']['media']:>5}")

    print(f"\n  📊 Índice de Fricción Burocrática (IFB):")
    for nivel, cnt in [("ninguno","ninguno"),("bajo","bajo"),("medio","medio"),
                        ("alto","alto"),("crítico","crítico")]:
        n = stats["ifb_niveles"].get(nivel, 0)
        barra = "█" * min(n // max(total_final // 30, 1), 25)
        print(f"    {nivel:<12} {n:>5}  {barra}")

    print(f"\n  🎭 Polaridad localizada:")
    for nivel in ["neutro", "leve", "moderado", "alto", "muy_alto"]:
        n = stats["polaridad_niveles"].get(nivel, 0)
        barra = "█" * min(n // max(total_final // 30, 1), 25)
        print(f"    {nivel:<12} {n:>5}  {barra}")

    print(f"\n  🏥 Nivel de Impacto (escala de gravedad):")
    for nivel in ["ninguno", "administrativo", "economico", "clinico"]:
        n = stats["impacto_niveles"].get(nivel, 0)
        barra = "█" * min(n // max(total_final // 30, 1), 25)
        print(f"    {nivel:<16} {n:>5}  {barra}")

    print(f"\n  Distribución por contexto temático:")
    for ctx, cnt in sorted(stats["contextos_total"].items(), key=lambda x: -x[1]):
        barra = "█" * min(cnt // max(total_final // 30, 1), 25)
        print(f"    {ctx:<30} {cnt:>5}  {barra}")

    print(f"\n  Lexicon localizado MX                  : ✅ ({len(LEXICON_NEGATIVO_LOCALIZADO)} términos ponderados)")
    print(f"  IFB — Índice Fricción Burocrática      : ✅ (6 componentes)")
    print(f"  Polaridad con ¡!, emojis, énfasis      : ✅")
    print(f"  Nivel de impacto (admin/econ/clín)     : ✅")
    print(f"  Emojis → texto                         : {'✅ activo' if EMOJI_DISPONIBLE else '⚠️  pip install emoji'}")
    print(f"  💾 Guardado en                         : {salida}")
    print(f"{'='*68}\n")


if __name__ == "__main__":
    main()

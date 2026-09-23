#!/usr/bin/env python3
"""
limpiar_dataset.py — Limpieza y depuración del dataset Reddit para análisis de sentimientos
Tema: Impacto de la ineficiencia del sector salud público MX frente a la influenza

Mejoras v3:
  ✦ [1] INDICADORES_PROBLEMA ampliados: tiempo, burocracia, impacto clínico, pago privado
  ✦ [2] Lematizador ligero: normaliza "k"→"que", "q"→"que", énfasis de letras repetidas
  ✦ [3] SUSTANTIVOS_SALUD ampliados: nivel local, farmacia, referencias IMSS coloquiales
  ✦ [4] limpiar_texto preserva signos de exclamación/interrogación y negaciones "no + verbo"
  ✦ [5] clasificar_relevancia devuelve las palabras clave que activaron el filtro (contexto)
  ✦ Dataset de salida incluye campo "palabras_clave_activas" por registro

Pipeline de limpieza:
  1.  Cargar JSON (streaming si es >50 MB)
  2.  Eliminar registros sin comentario / vacíos
  3.  Filtrar bots y cuentas eliminadas
  4.  Convertir emojis a texto
  5.  Normalización ligera (k/q → que, énfasis de letras)
  6.  Limpiar texto (HTML, URLs, Markdown) — preserva signos y negaciones
  7.  Filtrar por longitud mínima
  8.  Filtrar por score mínimo
  9.  Filtro de relevancia combinado (OR + AND) con contexto de activación
  10. Deduplicación residual por hash
  11. Guardar resultado + reporte detallado

Uso: python3 limpiar_dataset_v3.py
"""

import json
import re
import hashlib
import sys
from pathlib import Path
from datetime import datetime
from html import unescape

# ── Dependencias opcionales ───────────────────────────────────────────────────
try:
    import emoji as emoji_lib
    EMOJI_DISPONIBLE = True
except ImportError:
    EMOJI_DISPONIBLE = False

try:
    import ijson
    IJSON_DISPONIBLE = True
except ImportError:
    IJSON_DISPONIBLE = False

UMBRAL_STREAMING_MB = 50


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

MIN_CHARS            = 30
MIN_SCORE_COMENTARIO = -5

AUTORES_DESCARTADOS = {
    "[deleted]", "AutoModerator", "automoderator",
    "BotDefense", "Spam_Detector_Bot", "RemindMeBot",
    "sneakpeekbot", "WikiSummarizerBot",
}

# ── Filtro OR: pasa si contiene AL MENOS UNA de estas palabras ───────────────
PALABRAS_OR = [
    "influenza", "gripe", "gripa", "h1n1", "h3n2",
    "fiebre", "calentura", "tos", "contagio", "contagiad",
    "enferm", "síntoma", "sintoma", "virus", "viral",
    "pandemia", "epidemia", "brote",
    "oseltamivir", "tamiflu", "paracetamol", "ibuprofeno",
    "antibiótico", "antibiotico", "medicamento", "medicina",
    "receta", "dosis", "tratamiento", "pastilla",
    "vacuna", "vacunar", "vacunación", "vacunacion",
    "inmunización", "inmunizacion", "antiviral",
    "imss", "issste", "insabi", "ssa", "salubridad",
    "sector salud", "sistema de salud", "seguro social", "seguro popular",
    "hospital", "clínica", "clinica", "centro de salud",
    "urgencias", "emergencias", "consulta", "médico", "medico",
    "doctor", "enfermera", "saturad", "saturación", "saturacion",
    "desabasto", "desabastecimiento", "escasez",
    "surtir", "no surtieron", "agotado",
    "moche", "soborno", "corrupción", "corrupcion",
    "recorte", "austeridad", "salud pública", "salud publica",
    "cubrebocas", "mascarilla", "cuarentena", "aislamiento",
    # ── Añadidos v3 ──
    "cita", "trámite", "tramite", "papeleo", "burocracia",
    "particular", "privado", "privada", "gasté", "gaste", "pagué", "pague",
    "neumonía", "neumonia", "recayó", "recayo", "empeoró", "empeoro",
    "farmacia", "botica", "anaquel",
]

# ── Filtro AND: sustantivo de salud + indicador de problema ──────────────────
# Comentarios que combinen ambos → relevancia 'alta'
# Solo OR, sin el AND → relevancia 'media'

SUSTANTIVOS_SALUD = [
    # ── Instituciones formales ──
    "imss", "issste", "insabi", "ssa", "hospital", "clínica", "clinica",
    "médico", "medico", "doctor", "consulta", "urgencias",
    "medicamento", "medicina", "vacuna", "tratamiento",
    "seguro social", "salubridad",
    # ── Nivel local (v3) ──
    "centro de salud",
    "la clínica", "clínica familiar",
    "la 1", "la t1",                 # referencias coloquiales a UMFs del IMSS
    "umf",                           # Unidad de Medicina Familiar
    "módulo de salud",
    # ── Farmacia / suministro (v3) ──
    "farmacia", "botica", "anaquel",
    # ── Apodos ciudadanos ──
    "el seguro", "la raza", "siglo xxi", "20 de noviembre",
    "sector salud",
    # ── Complementos ──
    "especialista", "laboratorio", "carnet",
]

INDICADORES_PROBLEMA = [
    # ── Desabasto / suministro ──
    "desabasto", "escasez", "no hay", "sin medicamento", "agotado",
    "no surtieron", "no tienen", "no contaban", "insuficiente",
    "vacuna agotada", "se acabaron", "anaquel vacío",
    # ── Espera / tiempo / burocracia (v3) ──
    "fila", "cola", "horas", "espera", "tardaron", "tardó", "tardo",
    "cita", "meses",                 # "me dieron cita en 3 meses"
    "vuelta y vuelta",               # frustración burocrática clásica MX
    "trámite", "tramite",
    "papeleo",
    "viacrucis", "odisea",
    "vuelva mañana",
    # ── Saturación ──
    "saturad", "lleno", "urgencias llenas", "no hay camas", "no hay cupo",
    # ── Corrupción ──
    "moche", "mordida", "soborno", "corrupción", "corrupcion",
    "palancas", "privatización", "privatizacion",
    # ── Trato / atención deficiente ──
    "no funciona", "pésimo", "pesimo", "malo", "deficiente",
    "rechaz", "negaron", "no me atendieron", "no atienden",
    "maltrato", "grosero", "incapaz", "incompetente",
    "negligencia", "negligente",
    "no me hicieron caso", "no me diagnosticaron",
    # ── Impacto en salud (v3) ──
    "empeoró", "empeoro", "complicó", "complico",
    "neumonía", "neumonia",
    "recayó", "recayo",
    "no se curó", "no se curo",
    "murió", "murio", "falleció", "fallecio", "muerte",
    "intubado", "uci", "grave",
    # ── Pago privado como señal de falla del sistema (v3) ──
    "particular",                    # "tuve que ir al particular"
    "privado", "privada",            # "médico privado porque el IMSS no tenía"
    "gasté", "gaste",
    "pagué de mi bolsillo", "pagué", "pague",
    "de mi bolsillo",
    # ── Desfinanciamiento ──
    "recorte", "austeridad",
    # ── Sistema caído ──
    "no hay sistema", "sistema caído",
]


# ══════════════════════════════════════════════════════════════════════════════
#  CATEGORÍAS SEMÁNTICAS para el contexto de relevancia (Mejora 5)
#  Cada entrada: (etiqueta_categoria, [palabras_clave_del_grupo])
# ══════════════════════════════════════════════════════════════════════════════

CATEGORIAS_CONTEXTO: list[tuple[str, list[str]]] = [
    ("desabasto",       ["desabasto", "escasez", "no hay", "sin medicamento",
                         "agotado", "no surtieron", "anaquel vacío", "vacuna agotada"]),
    ("tiempo_espera",   ["fila", "cola", "horas", "espera", "tardaron", "tardó",
                         "tardo", "cita", "meses", "vuelta y vuelta", "viacrucis",
                         "odisea", "vuelva mañana", "trámite", "tramite", "papeleo"]),
    ("saturacion",      ["saturad", "lleno", "urgencias llenas", "no hay camas",
                         "no hay cupo"]),
    ("corrupcion",      ["moche", "mordida", "soborno", "corrupción", "corrupcion",
                         "palancas", "privatización", "privatizacion"]),
    ("maltrato",        ["rechaz", "negaron", "no me atendieron", "no atienden",
                         "maltrato", "grosero", "incapaz", "incompetente",
                         "no me hicieron caso", "no me diagnosticaron"]),
    ("negligencia",     ["negligencia", "negligente", "pésimo", "pesimo", "malo",
                         "deficiente", "no funciona"]),
    ("impacto_salud",   ["empeoró", "empeoro", "complicó", "complico",
                         "neumonía", "neumonia", "recayó", "recayo",
                         "no se curó", "no se curo", "murió", "murio",
                         "falleció", "fallecio", "muerte", "intubado", "uci", "grave"]),
    ("pago_privado",    ["particular", "privado", "privada", "gasté", "gaste",
                         "pagué", "pague", "de mi bolsillo"]),
    ("recorte",         ["recorte", "austeridad", "no hay sistema", "sistema caído"]),
]


# ══════════════════════════════════════════════════════════════════════════════
#  INSTALAR DEPENDENCIAS
# ══════════════════════════════════════════════════════════════════════════════

def instalar_dependencias():
    global EMOJI_DISPONIBLE, IJSON_DISPONIBLE, emoji_lib, ijson

    faltantes = []
    if not EMOJI_DISPONIBLE:
        faltantes.append("emoji")
    if not IJSON_DISPONIBLE:
        faltantes.append("ijson")
    if not faltantes:
        return

    print(f"\n  ⚠️  Librerías opcionales no encontradas: {', '.join(faltantes)}")
    resp = input("  ¿Instalarlas ahora? (s/n): ").strip().lower()
    if resp != "s":
        print("  Continuando sin ellas (funcionalidad reducida).\n")
        return

    import subprocess
    for lib in faltantes:
        print(f"  Instalando {lib}...")
        subprocess.run([sys.executable, "-m", "pip", "install", lib, "-q"], check=True)

    if "emoji" in faltantes:
        try:
            import emoji as emoji_lib
            EMOJI_DISPONIBLE = True
            print("  ✅ emoji instalado.")
        except ImportError:
            pass
    if "ijson" in faltantes:
        try:
            import ijson
            IJSON_DISPONIBLE = True
            print("  ✅ ijson instalado.")
        except ImportError:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA 2 — LEMATIZADOR LIGERO / NORMALIZACIÓN DE VARIANTES
# ══════════════════════════════════════════════════════════════════════════════

# Patron para detectar letras repetidas 3+ veces: "pésimoooo" → "pésimo"
_RE_LETRAS_REPETIDAS = re.compile(r"(.)\1{2,}", re.UNICODE)

# Patrones de jerga digital mexicana que afectan la detección de palabras clave
_NORMALIZACIONES: list[tuple[re.Pattern, str]] = [
    # "k" o "q" solos como pronombre/conjunción → "que"
    # Ej: "k no había nada" → "que no había nada"
    (re.compile(r"\bk\b", re.IGNORECASE),          "que"),
    (re.compile(r"\bq\b", re.IGNORECASE),           "que"),
    # "xq" / "xque" / "porq" → "porque"
    (re.compile(r"\bxqu[eé]\b", re.IGNORECASE),    "porque"),
    (re.compile(r"\bxq\b", re.IGNORECASE),          "porque"),
    (re.compile(r"\bporqu\b", re.IGNORECASE),       "porque"),
    # "tb" / "tmb" → "también"
    (re.compile(r"\btmbi[eé]n\b", re.IGNORECASE),  "también"),
    (re.compile(r"\btmb\b", re.IGNORECASE),         "también"),
    (re.compile(r"\btb\b", re.IGNORECASE),          "también"),
    # "wey" / "wei" / "güey" → "wey" (estandarizado, no afecta palabras clave pero normaliza)
    (re.compile(r"\bgüey\b", re.IGNORECASE),        "wey"),
    (re.compile(r"\bguey\b", re.IGNORECASE),        "wey"),
    # "d" sola como preposición → "de"
    (re.compile(r"(?<!\w)d(?!\w)", re.IGNORECASE), "de"),
    # "pa" solo como preposición → "para"  (cuidado: no afectar "papá")
    (re.compile(r"\bpa\b(?!')", re.IGNORECASE),     "para"),
    # "ntp" → "no te preocupes"  (señal contextual negativa)
    (re.compile(r"\bntp\b", re.IGNORECASE),         "no te preocupes"),
    # "msm" / "msmo" → "mismo"
    (re.compile(r"\bmsmo?\b", re.IGNORECASE),       "mismo"),
]


def normalizar_texto(texto: str) -> str:
    """
    Lematizador ligero aplicado ANTES del filtro de relevancia.
    No altera el texto guardado en 'comentario_raw'; solo sirve para
    que el clasificador detecte variantes coloquiales como si fueran
    las palabras clave originales.

    Pasos:
      1. Colapsar letras repetidas: "pésimoooooo" → "pésimo"
         Esto recupera enfasis emocional como señal semántica válida.
      2. Sustituir abreviaturas y jerga digital por su forma canónica.
    """
    # 1. Colapsar énfasis de letras: "maaaloooo" → "malo", "siiiiii" → "si"
    texto = _RE_LETRAS_REPETIDAS.sub(r"\1\1", texto)
    # (dejamos 2 repeticiones para no romper palabras como "cc", "ll", "rr")
    # Segunda pasada para colapsar el residuo a 1 cuando no es dígrafo legítimo
    texto = re.sub(r"([^clrn])\1+", r"\1", texto)   # cc→c sería raro; ll, rr, nn son válidos

    # 2. Abreviaturas y jerga
    for patron, reemplazo in _NORMALIZACIONES:
        texto = patron.sub(reemplazo, texto)

    return texto


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA 4 — LIMPIEZA DE TEXTO (preserva puntuación y negaciones)
# ══════════════════════════════════════════════════════════════════════════════

def emojis_a_texto(texto: str) -> str:
    """
    Convierte emojis a su nombre descriptivo entre dos puntos.
    😡 → :cara_enojada:   💊 → :pastilla:
    Los emojis son señales de sentimiento — se preservan como texto.
    """
    if not EMOJI_DISPONIBLE or not isinstance(texto, str):
        return texto
    return emoji_lib.demojize(texto, language="es")


def limpiar_texto(texto: str) -> str:
    """
    Limpieza textual orientada a análisis de sentimientos.

    PRESERVA (deliberadamente):
      · Signos de exclamación ¡! — "¡Pésimo servicio!" tiene más carga que "Pésimo servicio."
      · Signos de interrogación ¿? — indican duda/queja retórica
      · Negaciones "no + [verbo/adjetivo]" — el espacio entre "no" y la palabra
        siguiente NO se elimina; no se aplica ningún regex que pudiera separar
        el "no" de su contexto ("no había camas" ≠ "había camas")
      · Comas y puntos — delimitan oraciones; útiles para segmentación futura
      · Tildes y caracteres especiales del español (ñ, á, é, í, ó, ú, ü)

    ELIMINA:
      · HTML entities  → unescape previo
      · URLs (http/https/www)
      · Menciones Reddit u/usuario y r/subreddit
      · Marcado Markdown (negritas, cursivas, headers, bloques de código, citas >)
      · Espacios/saltos de línea excesivos
    """
    if not isinstance(texto, str):
        return ""

    # Entidades HTML → caracteres reales (ej. &amp; → &)
    texto = unescape(texto)

    # URLs
    texto = re.sub(r"https?://\S+|www\.\S+", "", texto)

    # Menciones Reddit
    texto = re.sub(r"\bu/\w+", "", texto)
    texto = re.sub(r"\br/\w+", "", texto)

    # Markdown: negritas e itálicas (*** ** * _ __)
    texto = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", texto, flags=re.DOTALL)
    texto = re.sub(r"_{1,2}(.+?)_{1,2}",   r"\1", texto, flags=re.DOTALL)

    # Headers Markdown (##, ###...)
    texto = re.sub(r"^#{1,6}\s*", "", texto, flags=re.MULTILINE)

    # Bloques de código (```...``` y `...`)
    texto = re.sub(r"`{1,3}.*?`{1,3}", "", texto, flags=re.DOTALL)

    # Citas Markdown (líneas que empiezan con >)
    texto = re.sub(r"^>.*$", "", texto, flags=re.MULTILINE)

    # Espacios y saltos de línea múltiples
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    texto = re.sub(r"[ \t]{2,}", " ", texto)

    # ── NO se elimina: ¡ ! ¿ ? , . — los conservamos todos ──

    return texto.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  MEJORA 5 — CLASIFICAR RELEVANCIA CON CONTEXTO DE ACTIVACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def clasificar_relevancia(texto_limpio: str, texto_normalizado: str) -> tuple[str, dict]:
    """
    Clasifica la relevancia temática y devuelve las palabras/categorías
    que activaron el filtro.

    Retorna:
      (relevancia, contexto)

      relevancia:  'alta' | 'media' | 'ninguna'
      contexto:    {
          "sustantivos_activos":   [...],  # qué sustantivos de salud se encontraron
          "indicadores_activos":   [...],  # qué indicadores de problema se encontraron
          "categorias_activas":    [...],  # etiquetas semánticas (desabasto, maltrato…)
      }

    El campo 'contexto' estará vacío si la relevancia es 'ninguna'.
    Usar texto_normalizado para el matching asegura que variantes como
    "pesimooooo" → "pesimo" activen sus indicadores correspondientes.
    """
    t_limpio = texto_limpio.lower()
    t_norm   = texto_normalizado.lower()

    # Trabajamos con ambas versiones para no perder lo que ya estaba correcto
    # en el texto limpio pero podría haber cambiado en la normalización
    def contiene(texto: str, kw: str) -> bool:
        return kw in texto

    # ── Filtro OR (red amplia) ──
    pasa_or = any(contiene(t_limpio, kw) or contiene(t_norm, kw)
                  for kw in PALABRAS_OR)
    if not pasa_or:
        return "ninguna", {}

    # ── Filtro AND: sustantivos ──
    sustantivos_activos = [
        s for s in SUSTANTIVOS_SALUD
        if contiene(t_limpio, s) or contiene(t_norm, s)
    ]

    # ── Filtro AND: indicadores ──
    indicadores_activos = [
        p for p in INDICADORES_PROBLEMA
        if contiene(t_limpio, p) or contiene(t_norm, p)
    ]

    # ── Categorías semánticas activadas ──
    categorias_activas: list[str] = []
    for etiqueta, palabras in CATEGORIAS_CONTEXTO:
        if any(contiene(t_limpio, kw) or contiene(t_norm, kw) for kw in palabras):
            categorias_activas.append(etiqueta)

    contexto = {
        "sustantivos_activos": sustantivos_activos,
        "indicadores_activos": indicadores_activos,
        "categorias_activas":  categorias_activas,
    }

    tiene_sustantivo = bool(sustantivos_activos)
    tiene_problema   = bool(indicadores_activos)

    relevancia = "alta" if (tiene_sustantivo and tiene_problema) else "media"
    return relevancia, contexto


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def hash_texto(texto: str) -> str:
    return hashlib.md5(" ".join(texto.lower().split()).encode("utf-8")).hexdigest()


def cargar_registros(ruta: Path):
    tam_mb = ruta.stat().st_size / (1024 * 1024)

    if tam_mb > UMBRAL_STREAMING_MB and IJSON_DISPONIBLE:
        print(f"  📦 Archivo grande ({tam_mb:.1f} MB) — usando streaming (ijson).")
        return _stream_ijson(ruta)

    if tam_mb > UMBRAL_STREAMING_MB:
        print(f"  ⚠️  Archivo grande ({tam_mb:.1f} MB) — ijson no disponible, cargando en RAM.")

    with open(ruta, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        for val in data.values():
            if isinstance(val, list):
                return val
        return [data]
    return data


def _stream_ijson(ruta: Path):
    with open(ruta, "rb") as f:
        for item in ijson.items(f, "datos.item"):
            yield item


# ══════════════════════════════════════════════════════════════════════════════
#  INTERFAZ DE ARCHIVO
# ══════════════════════════════════════════════════════════════════════════════

def pedir_archivos() -> tuple[Path, Path]:
    carpeta = Path(__file__).parent
    jsons = sorted(carpeta.glob("*.json"))
    if not jsons:
        raise FileNotFoundError("No se encontraron archivos .json en la carpeta.")

    print(f"\n{'='*66}")
    print("  LIMPIEZA DE DATASET v3 — ANÁLISIS DE SENTIMIENTOS  🏥")
    print(f"{'='*66}\n")
    print("  Archivos .json disponibles:")
    for i, p in enumerate(jsons):
        print(f"    [{i}] {p.name}  ({p.stat().st_size/1024:.1f} KB)")

    idx     = input("\n  Número del archivo a limpiar: ").strip()
    entrada = jsons[int(idx)]

    ts             = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_default = f"dataset_limpio_{ts}.json"
    print(f"\n  Nombre sugerido: {nombre_default}")
    resp = input("  Ruta/nombre de salida (Enter = sugerido): ").strip()

    if not resp:
        salida = carpeta / nombre_default
    else:
        salida = Path(resp)
        if salida.suffix != ".json":
            salida = salida.with_suffix(".json")
        if not salida.is_absolute() and len(salida.parts) == 1:
            salida = carpeta / salida

    return entrada, salida


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    instalar_dependencias()
    entrada, salida = pedir_archivos()

    print(f"\n  Cargando: {entrada.name} ...")
    registros = cargar_registros(entrada)

    eliminados = {
        "sin_comentario":      0,
        "autor_bot_eliminado": 0,
        "texto_muy_corto":     0,
        "score_bajo":          0,
        "no_relevante":        0,
        "duplicado_texto":     0,
    }
    conteo_relevancia  = {"alta": 0, "media": 0}
    conteo_categorias: dict[str, int] = {etq: 0 for etq, _ in CATEGORIAS_CONTEXTO}
    total_inicial = 0
    limpios: list[dict] = []
    hashes_vistos: set[str] = set()

    print(f"\n{'='*66}")
    print("  EJECUTANDO PIPELINE DE LIMPIEZA v3...")
    print(f"{'='*66}")

    for reg in registros:
        total_inicial += 1

        # 1. Comentario existe
        comentario_raw = reg.get("comentario", "")
        if not comentario_raw or not isinstance(comentario_raw, str):
            eliminados["sin_comentario"] += 1
            continue

        # 2. Filtro autor
        autor = reg.get("autor", "")
        if not autor or autor in AUTORES_DESCARTADOS:
            eliminados["autor_bot_eliminado"] += 1
            continue

        # 3. Emojis → texto (ANTES de limpiar, para preservar señal de sentimiento)
        comentario_con_emojis = emojis_a_texto(comentario_raw)

        # 4. Limpieza textual (preserva ¡!¿? y negaciones)
        comentario_limpio = limpiar_texto(comentario_con_emojis)

        # 5. Longitud mínima
        if len(comentario_limpio) < MIN_CHARS:
            eliminados["texto_muy_corto"] += 1
            continue

        # 6. Score mínimo
        score = reg.get("score_comentario", 0)
        if isinstance(score, (int, float)) and score < MIN_SCORE_COMENTARIO:
            eliminados["score_bajo"] += 1
            continue

        # 7. Normalización ligera (solo para clasificación; NO modifica el texto guardado)
        comentario_normalizado = normalizar_texto(comentario_limpio)

        # 8. Relevancia + contexto de activación
        relevancia, contexto = clasificar_relevancia(comentario_limpio, comentario_normalizado)
        if relevancia == "ninguna":
            eliminados["no_relevante"] += 1
            continue
        conteo_relevancia[relevancia] += 1

        # Acumular conteo de categorías para el reporte
        for cat in contexto.get("categorias_activas", []):
            conteo_categorias[cat] = conteo_categorias.get(cat, 0) + 1

        # 9. Deduplicación residual
        h = hash_texto(comentario_limpio)
        if h in hashes_vistos:
            eliminados["duplicado_texto"] += 1
            continue
        hashes_vistos.add(h)

        limpios.append({
            "post_id":               reg.get("post_id", ""),
            "subreddit":             reg.get("subreddit", ""),
            "titulo_post":           limpiar_texto(emojis_a_texto(reg.get("titulo_post", ""))),
            "comentario_raw":        comentario_raw,        # original intacto
            "comentario":            comentario_limpio,      # limpio, con ¡!¿? preservados
            "comentario_norm":       comentario_normalizado, # versión normalizada (para debug/NLP)
            "relevancia":            relevancia,             # 'alta' | 'media'
            # ── MEJORA 5: Contexto de relevancia ──────────────────────────
            "palabras_clave_activas": contexto,
            # ── Metadatos del registro ─────────────────────────────────────
            "autor":                 autor,
            "score_comentario":      score,
            "score_post":            reg.get("score_post", 0),
            "num_comentarios":       reg.get("num_comentarios", 0),
            "profundidad":           reg.get("profundidad", 0),
            "comentario_utc":        reg.get("comentario_utc", None),
            "post_creado_utc":       reg.get("post_creado_utc", None),
            "query_origen":          reg.get("query_origen", ""),
            "url_post":              reg.get("url_post", ""),
        })

    total_final = len(limpios)
    pct = round(total_final / total_inicial * 100, 2) if total_inicial else 0

    # ── Guardar ───────────────────────────────────────────────────────────────
    resultado = {
        "meta_limpieza": {
            "generado":             datetime.now().isoformat(),
            "version_script":       "3.0",
            "archivo_fuente":       str(entrada),
            "emoji_convertido":     EMOJI_DISPONIBLE,
            "carga_streaming":      IJSON_DISPONIBLE,
            "registros_entrada":    total_inicial,
            "registros_salida":     total_final,
            "registros_eliminados": sum(eliminados.values()),
            "tasa_retencion_pct":   pct,
            "relevancia_alta":      conteo_relevancia["alta"],
            "relevancia_media":     conteo_relevancia["media"],
            "detalle_eliminados":   eliminados,
            "distribucion_categorias": conteo_categorias,  # ← Nuevo en v3
            "config": {
                "min_chars":             MIN_CHARS,
                "min_score_comentario":  MIN_SCORE_COMENTARIO,
                "palabras_or":           len(PALABRAS_OR),
                "sustantivos_salud":     len(SUSTANTIVOS_SALUD),
                "indicadores_problema":  len(INDICADORES_PROBLEMA),
                "categorias_contexto":   len(CATEGORIAS_CONTEXTO),
            },
        },
        "datos": limpios,
    }

    salida.parent.mkdir(parents=True, exist_ok=True)
    with open(salida, "w", encoding="utf-8") as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    # ── Reporte en consola ────────────────────────────────────────────────────
    print(f"\n{'='*66}")
    print("  REPORTE DE LIMPIEZA v3")
    print(f"{'='*66}")
    print(f"  Registros entrada                  : {total_inicial:>6}")
    print(f"  ├─ Sin comentario                  : -{eliminados['sin_comentario']:>5}")
    print(f"  ├─ Bots / cuentas borradas         : -{eliminados['autor_bot_eliminado']:>5}")
    print(f"  ├─ Texto muy corto (<{MIN_CHARS}c)         : -{eliminados['texto_muy_corto']:>5}")
    print(f"  ├─ Score bajo (<{MIN_SCORE_COMENTARIO})               : -{eliminados['score_bajo']:>5}")
    print(f"  ├─ No relevante (temática)         : -{eliminados['no_relevante']:>5}")
    print(f"  └─ Duplicados residuales           : -{eliminados['duplicado_texto']:>5}")
    print(f"  {'─'*50}")
    print(f"  ✅ Registros limpios                : {total_final:>6}  ({pct}% retenido)")
    print(f"     ├─ Relevancia ALTA  (AND)       : {conteo_relevancia['alta']:>6}  ← queja/problema explícito")
    print(f"     └─ Relevancia MEDIA (OR)        : {conteo_relevancia['media']:>6}  ← mención general de salud")
    print(f"  {'─'*50}")
    print(f"  📊 Distribución por categoría semántica:")
    for etq, cnt in sorted(conteo_categorias.items(), key=lambda x: -x[1]):
        barra = "█" * min(cnt // max(1, total_final // 40), 30)
        print(f"     {etq:<18} {cnt:>5}  {barra}")
    print(f"  {'─'*50}")
    print(f"  Lematizador ligero                 : ✅ activo (k/q→que, énfasis colapsado)")
    print(f"  Signos ¡!¿? preservados            : ✅")
    print(f"  Negaciones preservadas             : ✅")
    print(f"  Contexto de relevancia guardado    : ✅ (campo palabras_clave_activas)")
    print(f"  Emojis → texto                     : {'✅ activo' if EMOJI_DISPONIBLE else '⚠️  pip install emoji'}")
    print(f"  Streaming ijson                    : {'✅ activo' if IJSON_DISPONIBLE else '⚠️  pip install ijson'}")
    print(f"  💾 Guardado en                     : {salida}")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Pipeline de limpieza:
  1.  Cargar JSON (streaming siempre para archivos >50 MB)
  2.  Eliminar registros sin comentario / vacíos
  3.  Filtrar bots y cuentas eliminadas
  4.  Convertir emojis a texto
  5.  Normalización ligera (k/q → que, énfasis, sarcasmo, frustración)
  6.  Limpiar texto (HTML, URLs, Markdown) — preserva signos y negaciones
  7.  Filtrar por longitud mínima (≥20 c)
  8.  Filtrar por score mínimo
  9.  Filtro de relevancia combinado (OR + AND) con contexto de activación
      v6: prioridad a opiniones sobre vacuna/vacunación de influenza
  10. Deduplicación residual por hash
  11. Guardar resultado en streaming + reporte detallado

Uso: python3 limpiar_dataset_v6.py
"""

import json
import re
import hashlib
import sys
from pathlib import Path
from datetime import datetime
from html import unescape
from decimal import Decimal

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


class DecimalEncoder(json.JSONEncoder):
    """
    ijson deserializa los números como Decimal para mayor precisión.
    Este encoder los convierte a int o float antes de serializar,
    evitando el TypeError: Object of type Decimal is not JSON serializable.
    """
    def default(self, obj):
        if isinstance(obj, Decimal):
            return int(obj) if obj % 1 == 0 else float(obj)
        return super().default(obj)


# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN
# ══════════════════════════════════════════════════════════════════════════════

MIN_CHARS            = 20  # ↓ v5: captura quejas cortas pero directas (FN en textos breves)
MIN_SCORE_COMENTARIO = -5

AUTORES_DESCARTADOS = {
    "[deleted]", "AutoModerator", "automoderator",
    "BotDefense", "Spam_Detector_Bot", "RemindMeBot",
    "sneakpeekbot", "WikiSummarizerBot",
}

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
    "cita", "trámite", "tramite", "papeleo", "burocracia",
    "particular", "privado", "privada", "gasté", "gaste", "pagué", "pague",
    "neumonía", "neumonia", "recayó", "recayo", "empeoró", "empeoro",
    "farmacia", "botica", "anaquel",
]

SUSTANTIVOS_SALUD = [
    "imss", "issste", "insabi", "ssa", "hospital", "clínica", "clinica",
    "médico", "medico", "doctor", "consulta", "urgencias",
    "medicamento", "medicina", "vacuna", "tratamiento",
    "seguro social", "salubridad",
    "centro de salud",
    "la clínica", "clínica familiar",
    "la 1", "la t1",
    "umf",
    "módulo de salud",
    "farmacia", "botica", "anaquel",
    "el seguro", "la raza", "siglo xxi", "20 de noviembre",
    "sector salud",
    "especialista", "laboratorio", "carnet",
]

# ── v6: léxico específico de vacuna / vacunación de influenza ─────────────────
SUSTANTIVOS_VACUNA = [
    "vacuna", "vacunas", "vacunación", "vacunacion",
    "vacunar", "vacunarse", "vacunado", "vacunada",
    "antiflu", "antigripal", "antiviral",
    "dosis de vacuna", "refuerzo", "segunda dosis",
    "inmunización", "inmunizacion", "inmunizar",
    "módulo de vacunación", "modulo de vacunacion",
    "campaña de vacunación", "campaña de vacunacion",
    "esquema de vacunación", "esquema de vacunacion",
]

INDICADORES_VACUNA = [
    # Hesitancia / rechazo
    "no me vacuno", "no quiero vacunarme", "no me la puse", "no me la voy a poner",
    "no me vacuné", "no me vacune",
    "miedo a la vacuna", "le tengo miedo", "desconfío de la vacuna", "desconfio de la vacuna",
    "no confío en la vacuna", "no confio en la vacuna",
    "antivacuna", "anti-vacuna", "anti vacuna",
    "no creo en las vacunas", "no creo en la vacuna",
    "prefiero no vacunarme", "me niego a vacunarme",
    "no es necesaria", "no es necesario vacunarse",
    "propaganda", "negocio de las vacunas",
    # Efectos secundarios / reacciones
    "efecto secundario", "efectos secundarios",
    "reacción a la vacuna", "reaccion a la vacuna",
    "me dolió el brazo", "me cayó mal la vacuna", "me cayo mal la vacuna",
    "me dio fiebre después", "me dio fiebre despues",
    "me sentí mal después", "me senti mal despues",
    "me puse mal con la vacuna", "me enfermé de la vacuna", "me enferme de la vacuna",
    "alergia a la vacuna", "reacción alérgica", "reaccion alergica",
    # Acceso / disponibilidad
    "vacuna agotada", "no hay vacuna", "no había vacuna", "no habia vacuna",
    "no me quisieron vacunar", "no me vacunaron", "no me pusieron la vacuna",
    "no tenían vacuna", "no tenian vacuna", "se acabó la vacuna", "se acabo la vacuna",
    "no aplican la vacuna",
    # Efectividad / dudas sobre eficacia
    "no sirve la vacuna", "no funciona la vacuna", "la vacuna no funciona",
    "me vacuné y me enfermé", "me vacune y me enferme",
    "igual me dio gripa", "de todas formas me contagié", "de todas formas me contagie",
    "vacuna inútil", "vacuna inutil", "vacuna ineficaz",
    "para qué me vacuno", "para que me vacuno",
    "de qué sirve", "de que sirve vacunarse",
    # Opiniones positivas / recomendación
    "me vacuné", "ya me vacune", "ya me vacuné",
    "recomienda vacunarse", "recomiendo vacunarse", "vacúnense", "vacunense",
    "gracias a la vacuna", "la vacuna me protegió", "la vacuna me protegió",
    "importante vacunarse", "es importante vacunarse",
]

INDICADORES_PROBLEMA = [
    "desabasto", "escasez", "no hay", "sin medicamento", "agotado",
    "no surtieron", "no tienen", "no contaban", "insuficiente",
    "vacuna agotada", "se acabaron", "anaquel vacío",
    "fila", "cola", "horas", "espera", "tardaron", "tardó", "tardo",
    "cita", "meses",
    "vuelta y vuelta",
    "trámite", "tramite",
    "papeleo",
    "viacrucis", "odisea",
    "vuelva mañana",
    "saturad", "lleno", "urgencias llenas", "no hay camas", "no hay cupo",
    "moche", "mordida", "soborno", "corrupción", "corrupcion",
    "palancas", "privatización", "privatizacion",
    "no funciona", "pésimo", "pesimo", "malo", "deficiente",
    "rechaz", "negaron", "no me atendieron", "no atienden",
    "maltrato", "grosero", "incapaz", "incompetente",
    "negligencia", "negligente",
    "no me hicieron caso", "no me diagnosticaron",
    "empeoró", "empeoro", "complicó", "complico",
    "neumonía", "neumonia",
    "recayó", "recayo",
    "no se curó", "no se curo",
    "murió", "murio", "falleció", "fallecio", "muerte",
    "intubado", "uci", "grave",
    "particular",
    "privado", "privada",
    "gasté", "gaste",
    "pagué de mi bolsillo", "pagué", "pague",
    "de mi bolsillo",
    "recorte", "austeridad",
    "no hay sistema", "sistema caído",
    # ── v5: lenguaje coloquial mexicano / quejas viscerales cortas ───────────
    "valen madre", "una basura", "de la chingada", "no sirve para nada",
    "puras promesas", "nos traen a vueltas", "negligencia", "ni un paracetamol",
    "no hay medicina", "puro cuento", "pinche imss", "pinche issste",
]

CATEGORIAS_CONTEXTO: list[tuple[str, list[str]]] = [
    # ── v6: categoría prioritaria — opiniones sobre la vacuna de influenza ─────
    ("vacuna_opinion",   ["vacuna", "vacunas", "vacunación", "vacunacion", "vacunar",
                          "vacunarse", "vacunado", "vacunada", "antiflu", "antigripal",
                          "efecto secundario", "efectos secundarios",
                          "no me vacuno", "no me vacuné", "antivacuna", "anti-vacuna",
                          "vacuna agotada", "no hay vacuna", "no sirve la vacuna",
                          "me vacuné y me enfermé", "inmunización", "inmunizacion",
                          "módulo de vacunación", "campaña de vacunación",
                          "esquema de vacunación", "reacción a la vacuna",
                          "hesitancia", "refuerzo", "segunda dosis"]),
    ("desabasto",       ["desabasto", "escasez", "no hay", "sin medicamento",
                         "agotado", "no surtieron", "anaquel vacío", "vacuna agotada",
                         "no hay medicina", "ni un paracetamol", "puro cuento",
                         "puras promesas"]),
    # ── v5: insumos básicos (gasas, jeringas, etc.) ──────────────────────────
    ("insumos_basicos",  ["gasa", "gasas", "jeringa", "jeringas", "suero", "alcohol",
                          "guantes", "cubrebocas", "mascarilla", "material de curación",
                          "insumo", "insumos", "ni gasa", "ni jeringa"]),
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
#  LEMATIZADOR LIGERO
# ══════════════════════════════════════════════════════════════════════════════

_RE_LETRAS_REPETIDAS = re.compile(r"(.)\1{2,}", re.UNICODE)

_NORMALIZACIONES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bk\b", re.IGNORECASE),          "que"),
    (re.compile(r"\bq\b", re.IGNORECASE),           "que"),
    (re.compile(r"\bxqu[eé]\b", re.IGNORECASE),    "porque"),
    (re.compile(r"\bxq\b", re.IGNORECASE),          "porque"),
    (re.compile(r"\bporqu\b", re.IGNORECASE),       "porque"),
    (re.compile(r"\btmbi[eé]n\b", re.IGNORECASE),  "también"),
    (re.compile(r"\btmb\b", re.IGNORECASE),         "también"),
    (re.compile(r"\btb\b", re.IGNORECASE),          "también"),
    (re.compile(r"\bgüey\b", re.IGNORECASE),        "wey"),
    (re.compile(r"\bguey\b", re.IGNORECASE),        "wey"),
    (re.compile(r"(?<!\w)d(?!\w)", re.IGNORECASE), "de"),
    (re.compile(r"\bpa\b(?!')", re.IGNORECASE),     "para"),
    (re.compile(r"\bntp\b", re.IGNORECASE),         "no te preocupes"),
    (re.compile(r"\bmsmo?\b", re.IGNORECASE),       "mismo"),
    # ── v5: señales de sarcasmo y frustración ────────────────────────────────
    (re.compile(r"\b(jaja|jeje|jiji)+\b", re.IGNORECASE), "sarcasmo_risa"),
    (re.compile(r"\b(ptm|hdp|hdpm)\b",    re.IGNORECASE), "frustracion_extrema"),
]


def normalizar_texto(texto: str) -> str:
    texto = _RE_LETRAS_REPETIDAS.sub(r"\1\1", texto)
    texto = re.sub(r"([^clrn])\1+", r"\1", texto)
    for patron, reemplazo in _NORMALIZACIONES:
        texto = patron.sub(reemplazo, texto)
    return texto


# ══════════════════════════════════════════════════════════════════════════════
#  LIMPIEZA DE TEXTO
# ══════════════════════════════════════════════════════════════════════════════

def emojis_a_texto(texto: str) -> str:
    if not EMOJI_DISPONIBLE or not isinstance(texto, str):
        return texto
    return emoji_lib.demojize(texto, language="es")


def limpiar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    texto = unescape(texto)
    texto = re.sub(r"https?://\S+|www\.\S+", "", texto)
    texto = re.sub(r"\bu/\w+", "", texto)
    texto = re.sub(r"\br/\w+", "", texto)
    texto = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", texto, flags=re.DOTALL)
    texto = re.sub(r"_{1,2}(.+?)_{1,2}",   r"\1", texto, flags=re.DOTALL)
    texto = re.sub(r"^#{1,6}\s*", "", texto, flags=re.MULTILINE)
    texto = re.sub(r"`{1,3}.*?`{1,3}", "", texto, flags=re.DOTALL)
    texto = re.sub(r"^>.*$", "", texto, flags=re.MULTILINE)
    texto = re.sub(r"\n{3,}", "\n\n", texto)
    texto = re.sub(r"[ \t]{2,}", " ", texto)
    return texto.strip()


# ══════════════════════════════════════════════════════════════════════════════
#  CLASIFICADOR DE RELEVANCIA
# ══════════════════════════════════════════════════════════════════════════════

def clasificar_relevancia(texto_limpio: str, texto_normalizado: str) -> tuple[str, dict]:
    t_limpio = texto_limpio.lower()
    t_norm   = texto_normalizado.lower()

    def contiene(texto: str, kw: str) -> bool:
        return kw in texto

    pasa_or = any(contiene(t_limpio, kw) or contiene(t_norm, kw)
                  for kw in PALABRAS_OR)
    if not pasa_or:
        return "ninguna", {}

    sustantivos_activos = [
        s for s in SUSTANTIVOS_SALUD
        if contiene(t_limpio, s) or contiene(t_norm, s)
    ]
    indicadores_activos = [
        p for p in INDICADORES_PROBLEMA
        if contiene(t_limpio, p) or contiene(t_norm, p)
    ]

    # ── v6: camino prioritario — vacuna / vacunación de influenza ────────────
    vacuna_sustantivos_activos = [
        s for s in SUSTANTIVOS_VACUNA
        if contiene(t_limpio, s) or contiene(t_norm, s)
    ]
    vacuna_indicadores_activos = [
        p for p in INDICADORES_VACUNA
        if contiene(t_limpio, p) or contiene(t_norm, p)
    ]
    # "alta" si hay sustantivo de vacuna + cualquier indicador (vacuna o sistema)
    es_alta_vacuna = bool(vacuna_sustantivos_activos and
                          (vacuna_indicadores_activos or indicadores_activos))
    # "alta" por vía original: sustantivo de salud + indicador de problema
    es_alta_salud  = bool(sustantivos_activos and indicadores_activos)

    categorias_activas: list[str] = []
    for etiqueta, palabras in CATEGORIAS_CONTEXTO:
        if any(contiene(t_limpio, kw) or contiene(t_norm, kw) for kw in palabras):
            categorias_activas.append(etiqueta)

    contexto = {
        "sustantivos_activos":        sustantivos_activos,
        "indicadores_activos":        indicadores_activos,
        "vacuna_sustantivos_activos": vacuna_sustantivos_activos,   # v6
        "vacuna_indicadores_activos": vacuna_indicadores_activos,   # v6
        "categorias_activas":         categorias_activas,
    }

    relevancia = "alta" if (es_alta_vacuna or es_alta_salud) else "media"
    return relevancia, contexto


# ══════════════════════════════════════════════════════════════════════════════
#  CARGA DE DATOS — v4: detección automática de ruta del array
# ══════════════════════════════════════════════════════════════════════════════

def hash_texto(texto: str) -> str:
    return hashlib.md5(" ".join(texto.lower().split()).encode("utf-8")).hexdigest()


def _detectar_ruta_ijson(ruta: Path) -> str:
    """
    Lee los primeros 8 KB del JSON para inferir en qué ruta vive el array
    de registros. Devuelve el prefijo ijson correcto.

    Casos manejados:
      - Array crudo:          [{...}, ...]               → "item"
      - Wrapper "datos":      {"datos": [{...}]}          → "datos.item"
      - Wrapper "registros":  {"registros": [{...}]}      → "registros.item"
      - Wrapper genérico:     {"cualquier_clave": [...]}  → "<clave>.item"
    """
    # Leer los primeros 8 KB para detectar la estructura sin cargar todo
    with open(ruta, "rb") as f:
        muestra = f.read(8192).decode("utf-8", errors="replace").strip()

    # Array directo
    if muestra.lstrip().startswith("["):
        return "item"

    # Busca la primera clave cuyo valor sea un array
    import re as _re
    # Busca: "clave": [  (el array puede empezar inmediatamente)
    match = _re.search(r'"(\w+)"\s*:\s*\[', muestra)
    if match:
        clave = match.group(1)
        return f"{clave}.item"

    # Fallback seguro
    print("  ⚠️  No se pudo detectar la estructura del JSON. Asumiendo array crudo.")
    return "item"


def _stream_ijson(ruta: Path):
    """
    Streaming con ijson. Detecta automáticamente la ruta del array
    (FIX principal de v4 respecto a v3 que asumía siempre 'datos.item').
    """
    ruta_array = _detectar_ruta_ijson(ruta)
    print(f"  🔍 Ruta detectada para streaming: '{ruta_array}'")

    with open(ruta, "rb") as f:
        count = 0
        for item in ijson.items(f, ruta_array):
            yield item
            count += 1
            if count % 10_000 == 0:
                print(f"     ... {count:,} registros leídos", end="\r")
    print(f"     ... {count:,} registros leídos en total.       ")


def _stream_json_stdlib(ruta: Path):
    """
    Fallback cuando ijson no está disponible.
    Carga el JSON completo con la stdlib y detecta el array de registros.
    Para archivos muy grandes puede usar mucha RAM; se recomienda instalar ijson.
    """
    with open(ruta, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return iter(data)

    if isinstance(data, dict):
        # Busca la primera clave cuyo valor sea una lista
        for val in data.values():
            if isinstance(val, list):
                return iter(val)
        return iter([data])

    return iter([data])


def cargar_registros(ruta: Path):
    tam_mb = ruta.stat().st_size / (1024 * 1024)

    if tam_mb > UMBRAL_STREAMING_MB:
        if IJSON_DISPONIBLE:
            print(f"  📦 Archivo grande ({tam_mb:.1f} MB) — streaming con ijson.")
            return _stream_ijson(ruta)
        else:
            print(f"  ⚠️  Archivo grande ({tam_mb:.1f} MB) — ijson no disponible.")
            print(f"      Cargando en RAM (puede ser lento). Instala ijson para mejor rendimiento.")

    return _stream_json_stdlib(ruta)


# ══════════════════════════════════════════════════════════════════════════════
#  ESCRITURA EN STREAMING — v4: no acumula lista en RAM
# ══════════════════════════════════════════════════════════════════════════════

class EscritorStreamingJSON:
    """
    Escribe un JSON con estructura {"meta_limpieza": {...}, "datos": [...]}
    de forma incremental: cada registro se serializa y escribe al instante,
    sin necesidad de tener toda la lista en memoria.

    FIX v4: en v3 se hacía json.dump(resultado, ...) al final, lo que
    requería tener TODOS los registros limpios en RAM simultáneamente.
    """

    def __init__(self, ruta: Path):
        self.ruta = ruta
        self._f = None
        self._primer_registro = True
        self._total_escritos = 0

    def abrir(self, meta: dict):
        ruta = self.ruta
        ruta.parent.mkdir(parents=True, exist_ok=True)
        self._f = open(ruta, "w", encoding="utf-8")
        # Escribir apertura del JSON raíz y la sección meta
        self._f.write("{\n")
        self._f.write('  "meta_limpieza": ')
        json.dump(meta, self._f, ensure_ascii=False, indent=2, cls=DecimalEncoder)
        self._f.write(',\n')
        self._f.write('  "datos": [\n')
        self._primer_registro = True

    def escribir(self, registro: dict):
        if not self._f:
            raise RuntimeError("Llama a abrir() antes de escribir().")
        if not self._primer_registro:
            self._f.write(",\n")
        # indent=2 para cada registro, pero lo desplazamos 4 espacios
        lineas = json.dumps(registro, ensure_ascii=False, indent=2, cls=DecimalEncoder)
        # Indentar todas las líneas con 4 espacios para que quede dentro de "datos"
        lineas_indentadas = "\n".join("    " + l for l in lineas.splitlines())
        self._f.write(lineas_indentadas)
        self._primer_registro = False
        self._total_escritos += 1

    def cerrar(self, meta_final: dict | None = None):
        """
        meta_final: si se pasa, se actualiza la sección meta_limpieza
        con los conteos definitivos. Requiere reescribir el archivo completo
        si el archivo es pequeño, o simplemente cerrar y dejar los campos
        ya escritos (estrategia de dos pasos para archivos grandes).
        """
        if not self._f:
            return
        self._f.write("\n  ]\n}")
        self._f.close()
        self._f = None

        # Actualizar meta_limpieza con los totales definitivos (reescritura ligera)
        if meta_final:
            self._actualizar_meta(meta_final)

    def _actualizar_meta(self, meta_final: dict):
        """
        Relee el archivo, reemplaza el bloque meta_limpieza con los valores
        definitivos (conteos que solo se conocen al terminar el pipeline)
        y lo reescribe. Solo se hace UNA vez al final.
        """
        try:
            with open(self.ruta, "r", encoding="utf-8") as f:
                contenido = f.read()

            # Encontrar y reemplazar el bloque meta_limpieza
            inicio = contenido.find('"meta_limpieza":')
            if inicio == -1:
                return
            # Buscar el final del objeto meta (primer '}' seguido de ',\n  "datos"')
            # Usamos json.decoder para encontrar el límite exacto
            decoder = json.JSONDecoder()
            # Encontrar el inicio del objeto meta
            idx_obj = contenido.index("{", inicio)
            obj_meta, _ = decoder.raw_decode(contenido, idx_obj)
            obj_meta.update(meta_final)

            nueva_meta = json.dumps(obj_meta, ensure_ascii=False, indent=2, cls=DecimalEncoder)
            fin_obj = contenido.index("{", inicio)
            # Reconstruir usando búsqueda de delimitador seguro
            partes = contenido.split('"meta_limpieza": ', 1)
            resto = partes[1]
            # Saltar el objeto meta original
            _, idx_fin = decoder.raw_decode(resto)
            nuevo_contenido = (
                partes[0]
                + '"meta_limpieza": '
                + nueva_meta
                + resto[idx_fin:]
            )
            with open(self.ruta, "w", encoding="utf-8") as f:
                f.write(nuevo_contenido)
        except Exception as e:
            print(f"  ⚠️  No se pudo actualizar meta_limpieza en el archivo: {e}")
            print(f"      Los totales en el reporte de consola son los correctos.")


# ══════════════════════════════════════════════════════════════════════════════
#  INTERFAZ DE ARCHIVO
# ══════════════════════════════════════════════════════════════════════════════

def pedir_archivos() -> tuple[Path, Path]:
    carpeta = Path(__file__).parent
    jsons = sorted(carpeta.glob("*.json"))
    if not jsons:
        raise FileNotFoundError("No se encontraron archivos .json en la carpeta.")

    print(f"\n{'='*66}")
    print("  LIMPIEZA DE DATASET v6 — ANÁLISIS DE SENTIMIENTOS  🏥")
    print(f"{'='*66}\n")
    print("  Archivos .json disponibles:")
    for i, p in enumerate(jsons):
        tam = p.stat().st_size
        if tam >= 1_048_576:
            tam_str = f"{tam/1_048_576:.1f} MB"
        else:
            tam_str = f"{tam/1024:.1f} KB"
        print(f"    [{i}] {p.name}  ({tam_str})")

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
    hashes_vistos: set[str] = set()

    print(f"\n{'='*66}")
    print("  EJECUTANDO PIPELINE DE LIMPIEZA v6...")
    print(f"{'='*66}")

    # ── Abrir escritor en streaming ───────────────────────────────────────────
    # La meta se llenará con placeholders; se actualizará al cerrar.
    meta_placeholder = {
        "generado":             datetime.now().isoformat(),
        "version_script":       "6.0",
        "archivo_fuente":       str(entrada),
        "emoji_convertido":     EMOJI_DISPONIBLE,
        "carga_streaming":      IJSON_DISPONIBLE,
        "registros_entrada":    "PENDIENTE",
        "registros_salida":     "PENDIENTE",
        "registros_eliminados": "PENDIENTE",
        "tasa_retencion_pct":   "PENDIENTE",
        "relevancia_alta":      "PENDIENTE",
        "relevancia_media":     "PENDIENTE",
        "detalle_eliminados":   eliminados,
        "distribucion_categorias": conteo_categorias,
        "config": {
            "min_chars":             MIN_CHARS,
            "min_score_comentario":  MIN_SCORE_COMENTARIO,
            "palabras_or":           len(PALABRAS_OR),
            "sustantivos_salud":     len(SUSTANTIVOS_SALUD),
            "indicadores_problema":  len(INDICADORES_PROBLEMA),
            "categorias_contexto":   len(CATEGORIAS_CONTEXTO),
            # v6: léxico de vacuna
            "sustantivos_vacuna":    len(SUSTANTIVOS_VACUNA),
            "indicadores_vacuna":    len(INDICADORES_VACUNA),
        },
    }

    escritor = EscritorStreamingJSON(salida)
    escritor.abrir(meta_placeholder)

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

        # 3. Emojis → texto
        comentario_con_emojis = emojis_a_texto(comentario_raw)

        # 4. Limpieza textual
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

        # 7. Normalización ligera (solo para clasificación)
        comentario_normalizado = normalizar_texto(comentario_limpio)

        # 8. Relevancia + contexto de activación
        relevancia, contexto = clasificar_relevancia(comentario_limpio, comentario_normalizado)
        if relevancia == "ninguna":
            eliminados["no_relevante"] += 1
            continue
        conteo_relevancia[relevancia] += 1

        for cat in contexto.get("categorias_activas", []):
            conteo_categorias[cat] = conteo_categorias.get(cat, 0) + 1

        # 9. Deduplicación residual
        h = hash_texto(comentario_limpio)
        if h in hashes_vistos:
            eliminados["duplicado_texto"] += 1
            continue
        hashes_vistos.add(h)

        # 10. Escribir registro directamente al archivo (sin acumular en RAM)
        escritor.escribir({
            "post_id":               reg.get("post_id", ""),
            "subreddit":             reg.get("subreddit", ""),
            "titulo_post":           limpiar_texto(emojis_a_texto(reg.get("titulo_post", ""))),
            "comentario_raw":        comentario_raw,
            "comentario":            comentario_limpio,
            "comentario_norm":       comentario_normalizado,
            "relevancia":            relevancia,
            "palabras_clave_activas": contexto,
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

    total_final = escritor._total_escritos
    pct = round(total_final / total_inicial * 100, 2) if total_inicial else 0

    # ── Cerrar escritor y actualizar meta con totales reales ──────────────────
    meta_final = {
        "registros_entrada":    total_inicial,
        "registros_salida":     total_final,
        "registros_eliminados": sum(eliminados.values()),
        "tasa_retencion_pct":   pct,
        "relevancia_alta":      conteo_relevancia["alta"],
        "relevancia_media":     conteo_relevancia["media"],
        "detalle_eliminados":   eliminados,
        "distribucion_categorias": conteo_categorias,
    }
    escritor.cerrar(meta_final)

    # ── Reporte en consola ────────────────────────────────────────────────────
    print(f"\n{'='*66}")
    print("  REPORTE DE LIMPIEZA v6")
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
    print(f"  Prioridad vacuna/vacunación (v6)   : ✅ ({len(SUSTANTIVOS_VACUNA)} sustantivos, {len(INDICADORES_VACUNA)} indicadores)")
    print(f"  Emojis → texto                     : {'✅ activo' if EMOJI_DISPONIBLE else '⚠️  pip install emoji'}")
    print(f"  Streaming ijson (entrada)          : {'✅ activo' if IJSON_DISPONIBLE else '⚠️  pip install ijson'}")
    print(f"  Streaming escritura (salida)       : ✅ activo (sin acumulación en RAM)")
    print(f"  💾 Guardado en                     : {salida}")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()

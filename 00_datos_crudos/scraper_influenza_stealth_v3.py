import requests
import time
import random
import json
import sys
from datetime import datetime
from itertools import product as iterproduct

# ===========================================================================
# DETECCIÓN DE ENCODINGS SOPORTADOS
# ===========================================================================

def _codificaciones_soportadas() -> str:
    encodings = ["gzip", "deflate"]
    try:
        import brotli
        encodings.append("br")
    except ImportError:
        pass
    try:
        import zstandard
        encodings.append("zstd")
    except ImportError:
        pass
    return ", ".join(encodings)

ACCEPT_ENCODING_SEGURO = _codificaciones_soportadas()

# ===========================================================================
# 1. POOL DE IDENTIDADES DE NAVEGADOR
# ===========================================================================

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 11.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_3) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
]

ACCEPT_LANGUAGES = [
    "es-MX,es;q=0.9,en;q=0.8",
    "es-MX,es;q=0.8,en-US;q=0.5,en;q=0.3",
    "es,en-US;q=0.9,en;q=0.8",
    "es-419,es;q=0.9",
    "es-MX,es;q=0.9,en-US;q=0.7,en;q=0.5",
]

REFERERS = [
    "https://www.google.com/",
    "https://www.google.com.mx/",
    "https://duckduckgo.com/",
    "https://www.reddit.com/",
    "https://www.reddit.com/r/Mexico/",
    "https://www.reddit.com/r/CDMX/",
    "https://www.bing.com/",
]

def generar_headers(ua: str | None = None, referer: str | None = None) -> dict:
    ua = ua or random.choice(USER_AGENTS)
    return {
        "User-Agent":      ua,
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": random.choice(ACCEPT_LANGUAGES),
        "Accept-Encoding": ACCEPT_ENCODING_SEGURO,
        "Referer":         referer or random.choice(REFERERS),
        "DNT":             "1",
        "Connection":      "keep-alive",
        "Sec-Fetch-Dest":  "document",
        "Sec-Fetch-Mode":  "navigate",
        "Sec-Fetch-Site":  "same-origin",
        "Cache-Control":   "max-age=0",
        "TE":              "trailers",
    }


# ===========================================================================
# 2. DICCIONARIOS DE BÚSQUEDA — AMPLIADOS CON EMOJIS
# ===========================================================================

# ── Emojis como señales emocionales directas ──────────────────────────────
# Reddit MX usa emojis en posts de queja/frustración/urgencia. Incluirlos
# como términos captura posts que de otro modo quedarían fuera de las
# combinaciones de texto puro. Son parte real del lenguaje digital mexicano.

TERMINOS_ENFERMEDAD = [
    # ── Técnicos / formales ──
    "influenza", "H1N1", "H3N2", "virus influenza", "influenza AH1N1",
    "oseltamivir", "tamiflu", "zanamivir",
    "vacuna influenza", "vacuna antigripal",
    # ── Coloquialismos mexicanos ──
    "gripe", "gripa", "gripaza",
    "calentura y tos", "calentura", "fiebre y tos",
    "temporada invernal", "brote influenza", "epidemia gripe",
    "vacunación", "cartilla vacunación", "campaña vacunación",
    # ── Con emojis (captura posts emocionales / urgentes) ──
    "influenza 🤧", "gripa 😷", "fiebre 🤒",
    "vacuna 💉", "enfermo 😰", "hospital 🏥",
    "influenza AH1N1 😷", "gripa 🤧 hospital",
]

TERMINOS_INSTITUCION = [
    # ── Nombres formales ──
    "IMSS", "ISSSTE", "Secretaría de Salud", "SSA",
    "INSABI", "IMSS Bienestar", "IMSS-Bienestar",
    "Subsecretaría de Salud", "CONACYT salud",
    "Servicios de Salud CDMX", "SEDESA",
    # ── Apodos / jerga ciudadana ──
    "el seguro",          # "me fui al seguro y no había nada"
    "la raza",            # Hospital La Raza (IMSS, CDMX)
    "siglo xxi",          # Hospital Siglo XXI (IMSS, CDMX)
    "20 de noviembre",    # Hospital 20 de Noviembre (ISSSTE)
    "centro de salud",
    "clínica familiar",
    "clínica",
    "salubridad",
    "Hospital General",
    "Sector Salud",
    "Seguro Social",
    "UMF",                # Unidad de Medicina Familiar (IMSS)
    "urgencias IMSS",
    "urgencias ISSSTE",
    # ── Con emojis ──
    "IMSS 😤", "ISSSTE 😡", "Seguro Social 🏥",
    "el seguro 😤", "hospital gobierno 😒",
]

TERMINOS_PROBLEMA = [
    # ── Desabasto / infraestructura ──
    "desabasto", "sin medicamento", "sin medicinas", "no hay medicamento",
    "no hay camas", "saturado", "urgencias lleno", "saturación hospitales",
    "no me atendieron", "me mandaron a casa", "me negaron atención",
    "no hay espacio", "no hay cupo",
    # ── Gestión deficiente y corrupción ──
    "negligencia", "mala atención", "tardaron horas", "no diagnosticaron",
    "ineficiencia", "burocracia", "pésimo servicio",
    "moches", "mordida",          # corrupción informal
    "no hay sistema",             # "el sistema está caído"
    "viacrucis", "odisea",        # narrativa de agotamiento
    "personal grosero", "maltrato",
    "no me hicieron caso",
    # ── Prevención / vacunación ──
    "no hay vacuna", "no me vacunaron", "sin vacunas",
    "vacuna agotada", "fila de horas", "no alcanzo vacuna",
    "se acabaron las dosis",
    # ── Medicamentos ──
    "paracetamol",                # "solo me dieron paracetamol" = sin antivirales
    "no tienen oseltamivir",
    "me dieron suero y ya",
    "sin antibióticos",
    # ── Consecuencias clínicas ──
    "murió de influenza", "falleció", "complicaciones",
    "neumonía por influenza", "internado UCI",
    "intubado", "grave por gripa",
    # ── Con emojis (expresan indignación y desesperación) ──
    "sin medicamento 😡", "no hay vacuna 😤", "negligencia 😱",
    "murió 😢 influenza", "desabasto 😠", "mala atención 🤬",
    "no hay camas 😔", "viacrucis 😩 IMSS", "urgencias 🚨 llenas",
    "no me atendieron 😤", "fiebre 🤒 sin medicamento",
]

# Subreddits relevantes (ciudadanos mexicanos)
SUBREDDITS = [
    "Mexico", "mexico", "CDMX", "preguntaleareddit",
    "es", "aves_de_mexico",     # Comunidades hispanohablantes activas
]

RESULTADOS_POR_QUERY = 8
MAX_PROFUNDIDAD       = 5
BLOQUE_QUERIES        = 100  # Tamaño de cada bloque interactivo (antes MAX_QUERIES=15)
PAUSA_LARGA_CADA      = 20
PAUSA_LARGA_DURACION  = (45, 90)

SORT_MODES = ["relevance", "top", "new", "comments"]


# ===========================================================================
# 3. GENERACIÓN Y ALEATORIZACIÓN DE QUERIES
# ===========================================================================

def construir_queries() -> list[tuple[str, str, str]]:
    """
    Genera tripletas (subreddit, query, sort) de todas las combinaciones.
    Incluye queries de texto puro + queries con emojis combinados.
    Se mezclan al azar para patrón de acceso impredecible.
    """
    combos = []

    # ── Combinaciones de 3 términos (núcleo original) ──
    for enf, inst, prob in iterproduct(TERMINOS_ENFERMEDAD, TERMINOS_INSTITUCION, TERMINOS_PROBLEMA):
        query = f"{enf} {inst} {prob}"
        sub   = random.choice(SUBREDDITS)
        sort  = random.choice(SORT_MODES)
        combos.append((sub, query, sort))

    # ── Queries de 2 términos con emojis (cobertura de posts cortos) ──
    terminos_con_emoji = [t for t in (
        TERMINOS_ENFERMEDAD + TERMINOS_INSTITUCION + TERMINOS_PROBLEMA
    ) if any(ord(c) > 127 for c in t)]

    for t1, t2 in iterproduct(terminos_con_emoji, TERMINOS_PROBLEMA[:10]):
        query = f"{t1} {t2}"
        sub   = random.choice(SUBREDDITS)
        sort  = random.choice(SORT_MODES)
        combos.append((sub, query, sort))

    random.shuffle(combos)
    return combos


# ===========================================================================
# 4. SESIÓN HTTP CON IDENTIDAD ROTATIVA
# ===========================================================================

class SesionHumana:
    """
    Wrapper sobre requests.Session con comportamiento anti-detección:
      · Rotación de UA e headers
      · Delays gaussianos
      · Pausas largas periódicas
      · Backoff exponencial con jitter
      · Cadena de Referer coherente
    """

    def __init__(self):
        self.session = requests.Session()
        self._request_count  = 0
        self._ua_actual      = random.choice(USER_AGENTS)
        self._ultimo_referer : str | None = None

    def _rotar_identidad(self):
        if random.random() < 0.15:
            self._ua_actual = random.choice(USER_AGENTS)

    def _pausa_humana(self, minimo: float = 2.0, maximo: float = 6.0):
        mu     = (minimo + maximo) / 2
        sigma  = (maximo - minimo) / 4
        espera = max(minimo, min(maximo, random.gauss(mu, sigma)))
        if random.random() < 0.08:
            espera += random.uniform(3, 8)
        time.sleep(espera)

    def _verificar_pausa_larga(self):
        if self._request_count > 0 and self._request_count % PAUSA_LARGA_CADA == 0:
            dur = random.uniform(*PAUSA_LARGA_DURACION)
            print(f"\n  [⏸] Pausa de {dur:.0f}s (simulando comportamiento humano)...\n")
            time.sleep(dur)

    def get(self, url: str, max_reintentos: int = 4) -> requests.Response | None:
        self._rotar_identidad()
        headers = generar_headers(self._ua_actual, referer=self._ultimo_referer)

        for intento in range(max_reintentos):
            try:
                self._verificar_pausa_larga()
                resp = self.session.get(url, headers=headers, timeout=20)
                self._request_count += 1

                if resp.status_code == 200:
                    self._ultimo_referer = url.replace(".json", "")
                    return resp

                elif resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 90))
                    jitter      = random.uniform(10, 30)
                    espera      = retry_after + jitter
                    print(f"  [429] Rate limit. Pausando {espera:.0f}s...")
                    time.sleep(espera)

                elif resp.status_code in (403, 401):
                    print(f"  [{resp.status_code}] Acceso denegado. Rotando identidad...")
                    self._ua_actual = random.choice(USER_AGENTS)
                    time.sleep(random.uniform(15, 30))

                elif resp.status_code >= 500:
                    espera = (2 ** intento) * 5 + random.uniform(0, 5)
                    print(f"  [{resp.status_code}] Error servidor. Reintentando en {espera:.1f}s...")
                    time.sleep(espera)

                else:
                    print(f"  [HTTP {resp.status_code}] No recuperable: {url}")
                    return None

            except requests.exceptions.Timeout:
                espera = (2 ** intento) * 4
                print(f"  [Timeout] Intento {intento+1}. Esperando {espera}s...")
                time.sleep(espera)

            except requests.exceptions.ConnectionError as e:
                espera = (2 ** intento) * 6 + random.uniform(0, 4)
                print(f"  [ConError] {e}. Esperando {espera:.1f}s...")
                time.sleep(espera)

        print(f"  [FAIL] Abandonando tras {max_reintentos} intentos: {url[:80]}")
        return None


# ===========================================================================
# 5. EXTRACCIÓN DE CONTENIDO
# ===========================================================================

def extraer_comentarios_recursivo(children: list, prof_max: int = MAX_PROFUNDIDAD) -> list[dict]:
    """Extrae comentarios con anidamiento completo (niveles 2-3: micro-historias)."""
    resultado = []
    for child in children:
        if child.get("kind") != "t1":
            continue
        d = child.get("data", {})
        if "body" in d and d["body"] not in ("[deleted]", "[removed]", ""):
            resultado.append({
                "texto":       d["body"],
                "autor":       d.get("author", "[deleted]"),
                "score":       d.get("score", 0),
                "profundidad": d.get("depth", 0),
                "creado_utc":  d.get("created_utc"),
            })
            if prof_max > 0:
                replies = d.get("replies")
                if isinstance(replies, dict):
                    nested = replies.get("data", {}).get("children", [])
                    resultado.extend(extraer_comentarios_recursivo(nested, prof_max - 1))
    return resultado


def extraer_hilo(sesion: SesionHumana, url_post: str) -> list[dict]:
    """Descarga y parsea comentarios de un hilo individual."""
    url = url_post if url_post.endswith(".json") else url_post + ".json"
    sesion._pausa_humana(1.5, 3.5)
    resp = sesion.get(url)
    if resp is None:
        return []
    try:
        data = resp.json()
        return extraer_comentarios_recursivo(data[1]["data"]["children"])
    except (IndexError, KeyError, json.JSONDecodeError, ValueError) as e:
        print(f"  [Parse hilo] {e}")
        return []


def buscar_posts(sesion: SesionHumana, subreddit: str, query: str, limite: int,
                 sort: str = "relevance") -> list[dict]:
    """
    Busca posts con paginación (token after) para superar el límite de 25.
    Soporta emojis en la query mediante urllib quote.
    """
    posts = []
    after = None
    vistos = set()

    while len(posts) < limite:
        url = (
            f"https://www.reddit.com/r/{subreddit}/search.json"
            f"?q={requests.utils.quote(query)}&restrict_sr=1"
            f"&sort={sort}&limit=25&type=link"
        )
        if after:
            url += f"&after={after}"

        sesion._pausa_humana(2, 5)
        resp = sesion.get(url)
        if resp is None:
            break

        try:
            payload  = resp.json()
            children = payload["data"]["children"]
            if not children:
                break

            for p in children:
                d   = p["data"]
                pid = d.get("id")
                if pid and pid not in vistos:
                    vistos.add(pid)
                    posts.append({
                        "id":              pid,
                        "titulo":          d.get("title", ""),
                        "permalink":       "https://www.reddit.com" + d.get("permalink", ""),
                        "score":           d.get("score", 0),
                        "num_comentarios": d.get("num_comments", 0),
                        "creado_utc":      d.get("created_utc"),
                        "subreddit":       d.get("subreddit"),
                        "url_externa":     d.get("url", ""),
                        "sort_usado":      sort,
                    })
                if len(posts) >= limite:
                    break

            after = payload["data"].get("after")
            if not after:
                break

        except (KeyError, json.JSONDecodeError) as e:
            print(f"  [Parse búsqueda] {e}")
            break

    return posts


# ===========================================================================
# 6. PREGUNTA INTERACTIVA DE CONTINUACIÓN
# ===========================================================================

def preguntar_continuar(bloque_num: int, total_acumulado: int) -> bool:
    """
    Muestra un resumen del bloque completado y pregunta al usuario
    si desea continuar con las siguientes 100 búsquedas o finalizar.
    Devuelve True = continuar, False = parar.
    """
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print(f"║  ✅ BLOQUE {bloque_num:02d} COMPLETADO — 100 queries procesadas    ║")
    print(f"║  📊 Comentarios acumulados hasta ahora: {total_acumulado:<14}║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  ¿Deseas continuar con las siguientes 100 búsquedas? ║")
    print("║                                                      ║")
    print("║    [S] Sí, continuar con otras 100 búsquedas         ║")
    print("║    [N] No, guardar dataset y finalizar               ║")
    print("╚══════════════════════════════════════════════════════╝")

    while True:
        try:
            respuesta = input("\n  Tu elección [S/N]: ").strip().upper()
        except (EOFError, KeyboardInterrupt):
            # Si stdin está redirigido o se interrumpe, se detiene
            print("\n  [Interrupción detectada] Finalizando...")
            return False

        if respuesta in ("S", "SI", "SÍ", "Y", "YES", "1"):
            print("  → Continuando con el siguiente bloque...\n")
            return True
        elif respuesta in ("N", "NO", "0"):
            print("  → Finalizando scraping por decisión del usuario.\n")
            return False
        else:
            print("  ⚠ Respuesta no reconocida. Escribe S para continuar o N para parar.")


# ===========================================================================
# 7. GUARDAR DATASET (soporte explícito de emojis)
# ===========================================================================

def guardar_dataset(data: list[dict], bloque: int | None = None) -> str:
    """
    Guarda el dataset como JSON con soporte completo de Unicode/emojis.
    ensure_ascii=False conserva los emojis tal cual en el archivo.
    """
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    sufijo = f"_bloque{bloque:02d}" if bloque else "_final"
    nombre = f"dataset_influenza_crudo{sufijo}_{ts}.json"

    with open(nombre, "w", encoding="utf-8") as f:
        # ensure_ascii=False → emojis y caracteres especiales se guardan sin escapar
        json.dump(data, f, ensure_ascii=False, indent=4)

    return nombre


# ===========================================================================
# 8. PROCESO PRINCIPAL
# ===========================================================================

def main():
    print("=" * 62)
    print("  SCRAPER v3 — SALUD PÚBLICA MX / INFLUENZA  🏥🤧💉")
    print(f"  Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)

    # Construye el pool completo de queries y lo mezcla
    todas_las_queries = construir_queries()
    total_disponibles = len(todas_las_queries)
    print(f"  Pool total de queries generadas : {total_disponibles:,}")
    print(f"  Tamaño de cada bloque           : {BLOQUE_QUERIES}")
    print(f"  Subreddits activos              : {SUBREDDITS}")
    print(f"  Soporte emojis en queries       : ✅")
    print()

    sesion        : SesionHumana  = SesionHumana()
    data_final    : list[dict]    = []          # Acumula todos los bloques
    posts_vistos  : set[str]      = set()       # Deduplicación global
    stats = {
        "posts_procesados":  0,
        "posts_duplicados":  0,
        "comentarios":       0,
        "queries_vacias":    0,
        "bloques_completos": 0,
    }

    offset = 0   # Puntero al inicio del bloque actual dentro del pool

    while offset < total_disponibles:
        bloque_num    = stats["bloques_completos"] + 1
        bloque_actual = todas_las_queries[offset: offset + BLOQUE_QUERIES]

        if not bloque_actual:
            break

        print(f"\n{'─'*62}")
        print(f"  🔍 BLOQUE {bloque_num:02d}  |  Queries {offset+1}–{offset+len(bloque_actual)}"
              f" de {total_disponibles}")
        print(f"{'─'*62}\n")

        for i, (subreddit, query, sort) in enumerate(bloque_actual, 1):
            idx_global = offset + i
            print(f"[{idx_global:03d}/{total_disponibles}] r/{subreddit} [{sort:>10}] | '{query}'")

            posts = buscar_posts(sesion, subreddit, query, RESULTADOS_POR_QUERY, sort=sort)
            if not posts:
                stats["queries_vacias"] += 1
                print("  → Sin resultados.\n")
                continue

            print(f"  → {len(posts)} posts encontrados.")

            for post in posts:
                pid = post["id"]
                if pid in posts_vistos:
                    stats["posts_duplicados"] += 1
                    continue
                posts_vistos.add(pid)
                stats["posts_procesados"] += 1

                titulo_corto = post["titulo"][:55] + "..." if len(post["titulo"]) > 55 else post["titulo"]
                print(f"     [{pid}] '{titulo_corto}' ({post['num_comentarios']} cmts)")

                comentarios = extraer_hilo(sesion, post["permalink"])
                stats["comentarios"] += len(comentarios)

                for c in comentarios:
                    data_final.append({
                        # ── Trazabilidad de minería ──
                        "bloque":             bloque_num,
                        "query_origen":       query,
                        "sort_usado":         sort,
                        "subreddit":          post["subreddit"] or subreddit,
                        # ── Post ──
                        "post_id":            pid,
                        "titulo_post":        post["titulo"],
                        "url_post":           post["permalink"],
                        "score_post":         post["score"],
                        "num_comentarios":    post["num_comentarios"],
                        "post_creado_utc":    post["creado_utc"],
                        # ── Comentario (emojis preservados tal cual) ──
                        "comentario":         c["texto"],
                        "autor":              c["autor"],
                        "score_comentario":   c["score"],
                        "profundidad":        c["profundidad"],
                        "comentario_utc":     c["creado_utc"],
                        # ── Metadatos ──
                        "fecha_extraccion":   datetime.now().isoformat(),
                    })

        # ── Fin del bloque: guardar checkpoint ──
        stats["bloques_completos"] += 1
        offset += BLOQUE_QUERIES

        checkpoint = guardar_dataset(data_final, bloque=bloque_num)
        print(f"\n  💾 Checkpoint guardado: {checkpoint}")
        print(f"  📊 Stats acumuladas → posts: {stats['posts_procesados']} | "
              f"comentarios: {stats['comentarios']}")

        # ── ¿Quedan más queries disponibles? ──
        if offset >= total_disponibles:
            print("\n  ℹ️  Se han procesado todas las queries disponibles del pool.")
            break

        # ── Pregunta interactiva de continuación ──
        continuar = preguntar_continuar(bloque_num, stats["comentarios"])
        if not continuar:
            break

    # ── Dataset final consolidado ──
    nombre_final = guardar_dataset(data_final)

    print("\n" + "=" * 62)
    print("  🏁 SCRAPING FINALIZADO")
    print(f"  Bloques completados              : {stats['bloques_completos']}")
    print(f"  Posts únicos procesados          : {stats['posts_procesados']}")
    print(f"  Posts duplicados saltados        : {stats['posts_duplicados']}")
    print(f"  Queries sin resultados           : {stats['queries_vacias']}")
    print(f"  Comentarios recolectados         : {stats['comentarios']}")
    print(f"  Emojis preservados en dataset    : ✅ (ensure_ascii=False)")
    print(f"  Archivo final guardado           : {nombre_final}")
    print("=" * 62)


if __name__ == "__main__":
    main()

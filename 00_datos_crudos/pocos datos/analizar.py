import json
import os
import glob
from collections import Counter

def analizar_json(ruta_archivo):
    with open(ruta_archivo, "r", encoding="utf-8") as f:
        datos = json.load(f)

    total_comentarios = len(datos)
    posts_unicos = set(d["post_id"] for d in datos if "post_id" in d)
    subreddits = Counter(d.get("subreddit", "desconocido") for d in datos)
    queries = Counter(d.get("query_origen", "desconocido") for d in datos)

    print(f"\n📄 Archivo: {os.path.basename(ruta_archivo)}")
    print(f"💬 Total de comentarios : {total_comentarios}")
    print(f"📌 Posts únicos         : {len(posts_unicos)}")

    print(f"\n🌐 Comentarios por subreddit:")
    for sub, n in subreddits.most_common():
        print(f"   r/{sub}: {n}")

    print(f"\n🔍 Comentarios por query de origen:")
    for q, n in queries.most_common():
        print(f"   '{q}': {n}")

if __name__ == "__main__":
    # Busca todos los JSON en la misma carpeta que el script
    carpeta = os.path.dirname(os.path.abspath(__file__))
    archivos = [f for f in glob.glob(os.path.join(carpeta, "*.json"))
                if os.path.basename(f) != "analizar.json"]

    if not archivos:
        print("No se encontró ningún archivo .json en esta carpeta.")
    else:
        for archivo in archivos:
            analizar_json(archivo)
            print("-" * 50)
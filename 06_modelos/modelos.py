#!/usr/bin/env python3
"""
modelos.py — Definiciones de modelos de clasificación (CRISP-DM Fase 5)
Tema: Ineficiencia del sector salud MX frente a la influenza

Pipelines disponibles
─────────────────────
  _pipeline_cnb              → Complement Naive Bayes   (texto puro, rápido, interpretable)
  _pipeline_lr               → Logistic Regression      (features híbridas texto + BERT)
  _pipeline_svm              → Linear SVM calibrado     (margen máximo, robusto en TF-IDF)
  obtener_ensamble_votacion  → VotingClassifier soft    (combina CNB + LR + SVM)

Por qué estos tres clasificadores base para salud:
  · CNB  : mejor NB para texto desbalanceado; ignora evidencia negativa
            al calcular complementos → ideal cuando "media" es mayoría.
  · LR   : aprende coeficientes conjuntos sobre features correlacionadas
            (prob_neg BERT ↔ sentimiento_num); excelente con datos híbridos.
  · SVM  : maximiza el margen entre clases en espacios de alta dimensión;
            el más robusto ante vocabulario ruidoso o jerga médica no vista.
            LinearSVC se calibra con CalibratedClassifierCV (método isotónico)
            para obtener probabilidades utilizables en inferencia de tesis.

Por qué el VotingClassifier como modelo PRINCIPAL:
  · Soft voting promedia las probabilidades calibradas de los tres clasificadores.
    Cada uno aporta una perspectiva diferente sobre la misma queja ciudadana:
      - CNB  captura frecuencia de términos de dominio (desabasto, negligencia).
      - LR   pondera señales BERT y TF-IDF conjuntamente.
      - SVM  actúa como árbitro de margen máximo en casos ambiguos.
  · El error se reduce porque los errores individuales no están correlacionados:
    cuando CNB falla en una queja irónica, LR la rescata por prob_neg BERT.
  · La tasa de desacuerdo entre clasificadores ES información: si los tres
    discrepan, el comentario está en la zona gris entre "media" y "alta" →
    valioso para la discusión de tesis (casos límite de ineficiencia sistémica).

CATÁLOGO_MODELOS
─────────────────
Diccionario que mapea nombre → callable factory.
Se importa en analizar_v4.py para construir la comparación de manera
dinámica: añadir un modelo nuevo = una sola línea en este catálogo.

Uso:
    from modelos import (CATALOGO_MODELOS, obtener_ensamble_votacion,
                         _pipeline_cnb, _pipeline_lr, _pipeline_svm)
"""

import numpy as np
from sklearn.naive_bayes         import ComplementNB
from sklearn.linear_model        import LogisticRegression
from sklearn.svm                 import LinearSVC
from sklearn.calibration         import CalibratedClassifierCV
from sklearn.ensemble            import VotingClassifier
from sklearn.pipeline            import Pipeline
from sklearn.preprocessing       import MaxAbsScaler, FunctionTransformer
from scipy.sparse                import issparse


# ══════════════════════════════════════════════════════════════════════════════
#  HELPER INTERNO — clip de negativos (sparse-safe)
#
#  Contexto: la matriz X combina TF-IDF (≥ 0) con features numéricas escaladas
#  con StandardScaler (pueden ser < 0). MaxAbsScaler mapea cada columna a [-1,1]
#  pero NO elimina los negativos. ComplementNB requiere X ≥ 0 estrictamente.
#  Solución: operar directamente sobre .data de la sparse para evitar densa.
# ══════════════════════════════════════════════════════════════════════════════

def _clip_negativos(X):
    """
    Fuerza X >= 0 respetando el formato sparse si aplica.
    Solo debe usarse antes de ComplementNB — LR y SVM manejan negativos.
    """
    if issparse(X):
        X = X.copy()
        X.data = np.clip(X.data, 0.0, None)
        return X
    return np.clip(X, 0.0, None)


# ══════════════════════════════════════════════════════════════════════════════
#  MODELO 1 — Complement Naive Bayes (CNB)
#
#  Cuándo es el mejor:
#    · Corpus de texto puro (sin features BERT activas)
#    · Clases desbalanceadas — CNB usa log_prob del complemento (otras clases)
#      → más estable que MultinomialNB cuando "media" >> "alta"
#    · Requiere X ≥ 0 → se añade paso clip tras MaxAbsScaler
#
#  Hiperparámetro alpha (suavizado de Laplace):
#    · 0.5 (raíz cuadrada de Laplace) es el mejor compromiso en textos
#      cortos/medios donde algunas palabras del dominio son raras.
#    · alpha=0 → riesgo de log(0) en tokens no vistos en entrenamiento.
#    · alpha=1 → suavizado completo de Laplace (Lidstone), más conservador.
# ══════════════════════════════════════════════════════════════════════════════

def _pipeline_cnb(alpha: float = 0.5) -> Pipeline:
    """
    Pipeline sparse-compatible: MaxAbsScaler → clip(0) → ComplementNB.

    Args:
        alpha: Suavizado de Laplace-Lidstone. Default 0.5 (óptimo para texto
               con vocabulario de dominio reducido como el de salud pública MX).

    Returns:
        sklearn.pipeline.Pipeline listo para fit/predict/predict_proba.
    """
    return Pipeline([
        ("scaler", MaxAbsScaler()),
        ("clip",   FunctionTransformer(_clip_negativos, validate=False)),
        ("cnb",    ComplementNB(alpha=alpha)),
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  MODELO 2 — Logistic Regression (LR)
#
#  Cuándo es el mejor:
#    · Corpus con features híbridas texto + BERT (prob_neg, sentimiento_num…)
#    · Features correlacionadas — la función de pérdida logística aprende
#      coeficientes conjuntos; CNB asume independencia condicional (incorrecto
#      cuando prob_neg BERT y sentimiento_num miden lo mismo).
#    · Coeficientes interpretables para la tesis: cada β_j cuantifica cuánto
#      aporta 'negligencia' vs prob_neg=0.91 a la decisión final.
#
#  Opciones de solver:
#    · lbfgs  : cuasi-Newton, eficiente en memoria, soporta multiclase.
#    · saga   : estocástico, escala a millones de muestras (>50k comentarios).
#    · liblinear: rápido en problemas pequeños (<10k), no soporta warm_start.
#  Se usa lbfgs (corpus ~2500-10k comentarios, binario).
# ══════════════════════════════════════════════════════════════════════════════

def _pipeline_lr(C: float = 1.0) -> Pipeline:
    """
    Pipeline para LogisticRegression con features híbridas (texto + BERT).
    No incluye clip porque LR maneja valores negativos nativamente.

    Args:
        C: Inverso de la fuerza de regularización L2. Valores menores
           → más regularización → reduce overfitting en vocabularios grandes.
           Default 1.0 (sin regularización fuerte); reducir a 0.1 si el corpus
           tiene >5000 features activas y gap train-test > 10%.

    Returns:
        sklearn.pipeline.Pipeline listo para fit/predict/predict_proba.
    """
    return Pipeline([
        ("scaler", MaxAbsScaler()),
        ("lr",     LogisticRegression(
            C            = C,
            max_iter     = 2000,       # convergencia garantizada con lbfgs
            class_weight = "balanced", # compensa desbalance media >> alta
            solver       = "lbfgs",
        )),
    ])


# ══════════════════════════════════════════════════════════════════════════════
#  MODELO 3 — Linear SVM calibrado (SVM)
#
#  Por qué LinearSVC para texto en salud:
#    · Maximiza el margen geométrico entre "alta" y "media" en el espacio
#      TF-IDF de alta dimensión → generaliza mejor a vocabulario no visto.
#    · Más robusto que LR ante outliers léxicos (jerga, abreviaciones médicas,
#      términos regionales de salud como "clínica de primer nivel").
#    · En benchmarks de clasificación de texto médico en español,
#      LinearSVC supera a CNB y es competitivo con LR en F1 de clase minoritaria
#      (referencia: Solano-Chinchilla et al., 2023, corpus IMSS-Twitter).
#
#  Calibración con CalibratedClassifierCV:
#    · LinearSVC no produce probabilidades directamente (usa distancia al hiperplano).
#    · CalibratedClassifierCV (isotónico, cv=5) aprende una función monótona
#      que convierte scores de decisión en probabilidades calibradas.
#    · Esencial para el ensamble: el soft voting requiere predict_proba real.
#
#  Hiperparámetro C:
#    · C bajo (0.1-0.5): margen más amplio, más regularización → corpus ruidoso.
#    · C alto (1-5)    : margen más estrecho, más ajuste → corpus limpio/grande.
#    · Default 0.5: equilibrio para tweets/comentarios cortos con ruido ortográfico.
# ══════════════════════════════════════════════════════════════════════════════

def _pipeline_svm(C: float = 0.5) -> CalibratedClassifierCV:
    """
    LinearSVC calibrado isotónicamente.

    Devuelve un CalibratedClassifierCV que envuelve el Pipeline de SVM.
    El objeto resultante soporta predict_proba (necesario para AUC, calibración
    y soft voting en el VotingClassifier).

    Args:
        C: Parámetro de penalización (margen). Rango recomendado [0.1, 2.0].
           Default 0.5 adecuado para tweets médicos en español con ruido.

    Returns:
        CalibratedClassifierCV listo para fit/predict/predict_proba.

    Nota técnica:
        CalibratedClassifierCV(pipeline_svm, cv=5) usa validación cruzada interna
        de 5 folds para aprender la calibración isotónica — NO requiere un split
        adicional. Si el corpus es pequeño (<500 muestras), reducir cv=3.
    """
    pipeline_base = Pipeline([
        ("scaler", MaxAbsScaler()),
        ("svm",    LinearSVC(
            C            = C,
            max_iter     = 3000,       # más iteraciones que LR por la naturaleza del solver
            class_weight = "balanced",
            dual         = "auto",     # auto elige primal/dual según n_samples vs n_features
        )),
    ])
    return CalibratedClassifierCV(pipeline_base, method="isotonic", cv=5)


# ══════════════════════════════════════════════════════════════════════════════
#  MODELO PRINCIPAL — VotingClassifier (Ensamble de votación suave)
#
#  Arquitectura del ensamble:
#    ┌─────────────────────────────────────────────────────────────────────┐
#    │         X  ──► CNB (pipeline_cnb) ──► P(alta | X)_cnb             │
#    │         X  ──► LR  (pipeline_lr)  ──► P(alta | X)_lr     ─► avg  │
#    │         X  ──► SVM (pipeline_svm) ──► P(alta | X)_svm            │
#    │                                                         ─► ŷ      │
#    └─────────────────────────────────────────────────────────────────────┘
#
#  Soft voting vs Hard voting:
#    · Soft (voting='soft'): promedia las probabilidades de todos los
#      clasificadores y predice la clase con mayor prob. promedio.
#      → Más fino: aprovecha la magnitud de la confianza de cada modelo.
#      → Preferido cuando todos los modelos están calibrados (probabilidades
#        reales, no solo scores).
#    · Hard: voto mayoritario de clases discretas.
#      → Menos fino, puede perder casos donde 2 modelos dicen "media" con
#        baja confianza y 1 dice "alta" con certeza 0.95.
#
#  Pesos (weights):
#    [1, 2, 2] → LR y SVM tienen el doble de peso que CNB.
#    Justificación:
#      - CNB asume independencia condicional (incorrecto con features BERT).
#      - LR y SVM aprenden dependencias entre features → más informativos
#        cuando el corpus incluye señales numéricas (prob_neg, sentimiento).
#    Para corpus TEXTO PURO (sin BERT), usa weights=[2, 1, 2].
#
#  Discrepancias como evidencia de tesis:
#    Los casos donde CNB → "alta" pero LR → "media" representan comentarios
#    donde el vocabulario es intenso (alto TF-IDF de términos como "muerte",
#    "negligencia") pero el contexto semántico es ambiguo. Estas discrepancias
#    son la zona gris de la ineficiencia sistémica: comentarios que "suenan"
#    graves pero cuya estructura narrativa es moderada.
#    → Ver analizar_discrepancias() en analizar_v4.py para extracción y export.
# ══════════════════════════════════════════════════════════════════════════════

def obtener_ensamble_votacion(weights: list = None) -> VotingClassifier:
    """
    Construye el ensamble de votación suave (soft voting) combinando
    ComplementNB, LogisticRegression y LinearSVM calibrado.

    El soft voting promedia las probabilidades P(alta | x) de cada
    clasificador ponderadas por `weights`. La clase predicha es
    argmax del promedio ponderado.

    Args:
        weights: Lista de 3 pesos [w_cnb, w_lr, w_svm].
                 Default None → [1, 2, 2] (LR y SVM con doble peso que CNB).
                 Para corpus sin BERT: usa [2, 1, 2].
                 Para corpus pequeño (<1000): usa [1, 1, 1] (igual peso).

    Returns:
        VotingClassifier con voting='soft', listo para fit/predict/predict_proba.

    Nota sobre CalibratedClassifierCV anidado:
        _pipeline_svm() ya retorna un CalibratedClassifierCV.
        VotingClassifier acepta estimadores heterogéneos en su lista —
        no necesita que todos sean Pipeline. El único requisito es que
        todos soporten predict_proba con voting='soft'. ✓ (los tres lo hacen).

    Acceso a sub-estimadores post-fit (para extraer coeficientes de LR):
        ensemble.named_estimators_['lr'].named_steps['lr'].coef_

    Ejemplo de uso:
        ens = obtener_ensamble_votacion()
        ens.fit(X_tr, y_tr)
        probs = ens.predict_proba(X_te)  # shape (n_te, 2)
        y_pred = ens.predict(X_te)       # array binario
    """
    if weights is None:
        weights = [1, 2, 2]    # LR y SVM pesan más que CNB por defecto

    return VotingClassifier(
        estimators=[
            ("cnb", _pipeline_cnb()),
            ("lr",  _pipeline_lr()),
            ("svm", _pipeline_svm()),
        ],
        voting   = "soft",
        weights  = weights,
        n_jobs   = -1,         # paralelizar entrenamiento de sub-estimadores
    )


# ══════════════════════════════════════════════════════════════════════════════
#  CATÁLOGO DE MODELOS
#
#  Estructura: { "Nombre legible": callable_factory }
#  El callable_factory es una función sin argumentos que retorna un estimador
#  sklearn con interfaz fit / predict / predict_proba.
#
#  VotingEnsemble está incluido para que comparar_modelos() lo evalúe junto
#  a los modelos individuales — la diferencia de métricas entre el ensamble
#  y el mejor modelo individual cuantifica la ganancia del ensemble learning.
#
#  Para añadir un modelo nuevo al pipeline completo (analizar_v4.py):
#    1. Definir aquí su función _pipeline_nuevo(...)
#    2. Añadir una línea en CATALOGO_MODELOS:
#         "NombreNuevo": _pipeline_nuevo
#    3. Sin más cambios — comparar_modelos() iterará automáticamente.
#    4. Si quieres incluirlo en el ensamble, editar obtener_ensamble_votacion().
# ══════════════════════════════════════════════════════════════════════════════

CATALOGO_MODELOS: dict[str, callable] = {
    "ComplementNB":       _pipeline_cnb,
    "LogisticRegression": _pipeline_lr,
    "LinearSVM":          _pipeline_svm,
    "VotingEnsemble":     obtener_ensamble_votacion,
}

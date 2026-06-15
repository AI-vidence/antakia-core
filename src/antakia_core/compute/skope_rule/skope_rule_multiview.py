"""
Règles descriptives multi-view (VS + ES) — spike exploratoire.

Stratégie
---------
1. Extraire des candidats indépendamment dans chaque espace via SkopeRules
   (même configuration que ``skope_rules``).
2. Scorer chaque candidat sur la sélection utilisateur :
   - VS-only / ES-only : précision/rappel dans l'espace concerné ;
   - conjoint (VS∧ES) : intersection des masques dans les deux espaces.
3. Retourner la paire conjointe de meilleur F1, plus un petit front de Pareto
   (VS-only, ES-only, conjoint).

Il n'y a pas d'optimisation globale type imodels RuleFit : on évalue
post-hoc un produit cartésien de candidats SkopeRules par espace.
"""

from __future__ import annotations

import warnings
from typing import Literal, TypedDict

import pandas as pd
from skope_rules_antakia import SkopeRules

from antakia_core.data_handler.rules import RuleSet
from antakia_core.utils.variable import DataVariables

CandidateKind = Literal["vs_only", "es_only", "conjoint"]

_SKOPE_KWARGS = dict(
    n_estimators=5,
    max_depth_duplication=0,
    max_samples=1.0,
    max_depth=3,
)


class RuleScore(TypedDict):
    precision: float
    recall: float
    f1: float


class ParetoCandidate(TypedDict):
    kind: CandidateKind
    vs: RuleSet
    es: RuleSet
    score: RuleScore


class MultiviewResultMeta(TypedDict, total=False):
    score: RuleScore
    pareto: list[ParetoCandidate]
    vs_candidates: int
    es_candidates: int


def _build_skope_classifier(
    variables: DataVariables,
    precision: float,
    recall: float,
    random_state: int,
) -> SkopeRules:
    return SkopeRules(
        feature_names=variables.columns_list(),
        random_state=random_state,
        recall_min=recall,
        precision_min=precision,
        **_SKOPE_KWARGS,
    )


def _extract_candidates(
    sk_rules: list,
    variables: DataVariables,
    max_candidates: int,
) -> list[tuple[RuleSet, RuleScore]]:
    candidates: list[tuple[RuleSet, RuleScore]] = []
    for rule_tuple in sk_rules[:max_candidates]:
        rule_set, score = RuleSet.sk_rules_to_rule_set([rule_tuple], variables)
        if len(rule_set) > 0:
            candidates.append((rule_set, score))
    return candidates


def _score_mask(pred_mask: pd.Series, y_mask: pd.Series) -> RuleScore:
    y = y_mask.astype(bool)
    pred = pred_mask.astype(bool)
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    fn = int((~pred & y).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


def _score_rule(rule_set: RuleSet, x: pd.DataFrame, y_mask: pd.Series) -> RuleScore:
    return _score_mask(rule_set.get_matching_mask(x), y_mask)


def _score_conjoint(
    vs_rules: RuleSet,
    es_rules: RuleSet,
    x_vs: pd.DataFrame,
    x_es: pd.DataFrame,
    y_mask: pd.Series,
) -> RuleScore:
    conjoint_mask = vs_rules.get_matching_mask(x_vs) & es_rules.get_matching_mask(
        x_es
    )
    return _score_mask(conjoint_mask, y_mask)


def _dominates(a: RuleScore, b: RuleScore) -> bool:
    return (
        a["precision"] >= b["precision"]
        and a["recall"] >= b["recall"]
        and (a["precision"] > b["precision"] or a["recall"] > b["recall"])
    )


def _pareto_front(candidates: list[ParetoCandidate]) -> list[ParetoCandidate]:
    front: list[ParetoCandidate] = []
    for candidate in candidates:
        if any(_dominates(other["score"], candidate["score"]) for other in candidates):
            continue
        front.append(candidate)
    return sorted(front, key=lambda c: (-c["score"]["f1"], c["kind"]))


def _fit_candidates(
    x: pd.DataFrame,
    y_mask: pd.Series,
    variables: DataVariables,
    precision: float,
    recall: float,
    random_state: int,
    max_candidates: int,
) -> list[tuple[RuleSet, RuleScore]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        classifier = _build_skope_classifier(
            variables, precision, recall, random_state
        )
        classifier.fit(x, y_mask.astype(int))
    if not classifier.rules_:
        return []
    return _extract_candidates(classifier.rules_, variables, max_candidates)


def skope_rules_multiview(
    df_mask: pd.Series,
    x_vs: pd.DataFrame,
    x_es: pd.DataFrame,
    variables: DataVariables | None = None,
    precision: float = 0.7,
    recall: float = 0.7,
    random_state: int = 42,
    max_candidates: int = 5,
) -> tuple[RuleSet, RuleSet, MultiviewResultMeta]:
    """
    Extrait et score conjointement des règles dans VS et ES.

    Parameters
    ----------
    df_mask
        Sélection utilisateur (cible binaire).
    x_vs, x_es
        DataFrames alignés sur les mêmes lignes (espaces VS et ES).
    variables
        Variables AntakIA ; construites depuis ``x_vs`` si absentes.
    precision, recall
        Seuils SkopeRules par espace.
    max_candidates
        Nombre max de règles SkopeRules conservées par espace.

    Returns
    -------
    vs_rules, es_rules, meta
        Paire recommandée (meilleur conjoint VS∧ES par F1) et métadonnées
        incluant le front de Pareto (VS-only, ES-only, conjoint).
    """
    empty_meta: MultiviewResultMeta = {"score": {}, "pareto": []}
    if df_mask.all() or not df_mask.any():
        return RuleSet(), RuleSet(), empty_meta
    if len(x_vs) != len(x_es) or not df_mask.index.equals(x_vs.index):
        raise ValueError("x_vs, x_es et df_mask doivent être alignés.")

    if variables is None:
        variables = DataVariables.build_variables(x_vs, [])

    vs_candidates = _fit_candidates(
        x_vs, df_mask, variables, precision, recall, random_state, max_candidates
    )
    es_candidates = _fit_candidates(
        x_es,
        df_mask,
        variables,
        precision,
        recall,
        random_state + 1,
        max_candidates,
    )

    pareto_pool: list[ParetoCandidate] = []

    best_vs_only: ParetoCandidate | None = None
    for vs_rules, _ in vs_candidates:
        score = _score_rule(vs_rules, x_vs, df_mask)
        candidate: ParetoCandidate = {
            "kind": "vs_only",
            "vs": vs_rules.copy(),
            "es": RuleSet(),
            "score": score,
        }
        pareto_pool.append(candidate)
        if best_vs_only is None or score["f1"] > best_vs_only["score"]["f1"]:
            best_vs_only = candidate

    best_es_only: ParetoCandidate | None = None
    for es_rules, _ in es_candidates:
        score = _score_rule(es_rules, x_es, df_mask)
        candidate = {
            "kind": "es_only",
            "vs": RuleSet(),
            "es": es_rules.copy(),
            "score": score,
        }
        pareto_pool.append(candidate)
        if best_es_only is None or score["f1"] > best_es_only["score"]["f1"]:
            best_es_only = candidate

    best_conjoint: ParetoCandidate | None = None
    for vs_rules, _ in vs_candidates:
        for es_rules, _ in es_candidates:
            score = _score_conjoint(vs_rules, es_rules, x_vs, x_es, df_mask)
            candidate = {
                "kind": "conjoint",
                "vs": vs_rules.copy(),
                "es": es_rules.copy(),
                "score": score,
            }
            pareto_pool.append(candidate)
            if best_conjoint is None or score["f1"] > best_conjoint["score"]["f1"]:
                best_conjoint = candidate

    pareto = _pareto_front(pareto_pool)

    # Recommandation : meilleur conjoint, sinon repli VS-only puis ES-only.
    recommended = best_conjoint or best_vs_only or best_es_only
    if recommended is None:
        return RuleSet(), RuleSet(), empty_meta

    meta: MultiviewResultMeta = {
        "score": recommended["score"],
        "pareto": pareto,
        "vs_candidates": len(vs_candidates),
        "es_candidates": len(es_candidates),
    }
    return recommended["vs"].copy(), recommended["es"].copy(), meta

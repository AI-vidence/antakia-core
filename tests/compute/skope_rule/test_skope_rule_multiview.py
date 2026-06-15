from unittest import TestCase

import pandas as pd

from antakia_core.compute.skope_rule.skope_rule_multiview import (
    skope_rules_multiview,
    _score_conjoint,
    _pareto_front,
)
from antakia_core.data_handler.rules import RuleSet
from antakia_core.data_handler.rule import Rule
from antakia_core.utils.variable import Variable, DataVariables


class TestSkopeRuleMultiview(TestCase):

    def setUp(self):
        self.x_vs = pd.DataFrame(
            [
                [1, 2],
                [2, 1],
                [4, 2],
                [10, 1],
                [20, 2],
            ],
            columns=["var1", "var2"],
        )
        # ES décorrélé : la sélection est identifiable conjointement.
        self.x_es = pd.DataFrame(
            [
                [0.1, 0.9],
                [0.9, 0.1],
                [0.5, 0.5],
                [0.9, 0.1],
                [0.1, 0.9],
            ],
            columns=["var1", "var2"],
        )
        self.var1 = Variable(0, "var1", "float")
        self.var2 = Variable(0, "var2", "float")
        self.variables = DataVariables([self.var1, self.var2])

    def test_empty_mask_returns_empty_rules(self):
        mask_all = pd.Series([True] * 5)
        vs, es, meta = skope_rules_multiview(
            mask_all, self.x_vs, self.x_es, self.variables
        )
        assert len(vs) == 0
        assert len(es) == 0
        assert meta["pareto"] == []

    def test_multiview_returns_paired_rules_and_pareto(self):
        mask = pd.Series([True, True, False, False, False])
        vs, es, meta = skope_rules_multiview(
            mask,
            self.x_vs,
            self.x_es,
            self.variables,
            precision=0.5,
            recall=0.5,
        )
        assert isinstance(vs, RuleSet)
        assert isinstance(es, RuleSet)
        assert "score" in meta
        assert "pareto" in meta
        assert len(meta["pareto"]) >= 1
        kinds = {c["kind"] for c in meta["pareto"]}
        assert kinds <= {"vs_only", "es_only", "conjoint"}

    def test_impossible_selection_returns_empty(self):
        mask = pd.Series([True, True, False, True, True])
        vs, es, meta = skope_rules_multiview(
            mask, self.x_vs, self.x_es, self.variables
        )
        assert len(vs) == 0
        assert len(es) == 0
        assert meta["score"] == {}

    def test_score_conjoint_intersection(self):
        vs_rule = RuleSet([Rule(self.var1, max=3, includes_max=True)])
        es_rule = RuleSet([Rule(self.var2, min=0.8, includes_min=False)])
        mask = pd.Series([True, False, False, False, False])
        score = _score_conjoint(
            vs_rule, es_rule, self.x_vs, self.x_es, mask
        )
        assert score["precision"] == 1.0
        assert score["recall"] == 1.0

    def test_pareto_front_keeps_non_dominated(self):
        front = _pareto_front([
            {
                "kind": "vs_only",
                "vs": RuleSet(),
                "es": RuleSet(),
                "score": {"precision": 1.0, "recall": 0.5, "f1": 0.667},
            },
            {
                "kind": "es_only",
                "vs": RuleSet(),
                "es": RuleSet(),
                "score": {"precision": 0.8, "recall": 0.9, "f1": 0.847},
            },
            {
                "kind": "conjoint",
                "vs": RuleSet(),
                "es": RuleSet(),
                "score": {"precision": 0.7, "recall": 0.7, "f1": 0.7},
            },
        ])
        assert len(front) == 2
        f1_values = {c["score"]["f1"] for c in front}
        assert 0.667 in f1_values
        assert 0.847 in f1_values

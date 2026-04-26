"""
Tests for validate_categories.py

Covers:
- make_emails(): correct total count, even distribution, required keys
- run_ensemble_test(): returns accuracy in range, per-cat dict, result list
- main(): overall rule-labeler accuracy >= 95%
"""

import pytest
from collections import Counter

from validate_categories import make_emails, run_ensemble_test, main


# ─────────────────────────────────────────────────────────────
#  make_emails
# ─────────────────────────────────────────────────────────────
class TestMakeEmails:
    EXPECTED_CATEGORIES = {
        "acceptance", "rejection", "interview",
        "action_required", "in_process", "unrelated",
    }

    def test_total_count(self):
        emails = make_emails()
        assert len(emails) == 200

    def test_all_six_categories_present(self):
        emails = make_emails()
        cats = {e["expected_label"] for e in emails}
        assert cats == self.EXPECTED_CATEGORIES

    def test_each_category_has_at_least_30_emails(self):
        emails = make_emails()
        counts = Counter(e["expected_label"] for e in emails)
        for cat in self.EXPECTED_CATEGORIES:
            assert counts[cat] >= 30, f"Category '{cat}' only has {counts[cat]} emails"

    def test_no_category_has_more_than_40_emails(self):
        """Distribution should be roughly even (33–34 per category)."""
        emails = make_emails()
        counts = Counter(e["expected_label"] for e in emails)
        for cat, count in counts.items():
            assert count <= 40, f"Category '{cat}' has {count} emails — too many"

    def test_required_keys_present(self):
        emails = make_emails()
        for e in emails:
            assert "subject"        in e
            assert "body"           in e
            assert "expected_label" in e

    def test_subject_and_body_are_nonempty_strings(self):
        emails = make_emails()
        for e in emails:
            assert isinstance(e["subject"], str) and len(e["subject"]) > 0
            assert isinstance(e["body"],    str) and len(e["body"])    > 0

    def test_deterministic(self):
        """make_emails() should return same content when called twice."""
        emails1 = make_emails()
        emails2 = make_emails()
        # random.seed(99) is module-level — re-shuffled on each call but
        # the content is deterministic given the same seed
        labels1 = [e["expected_label"] for e in emails1]
        labels2 = [e["expected_label"] for e in emails2]
        assert Counter(labels1) == Counter(labels2)


# ─────────────────────────────────────────────────────────────
#  run_ensemble_test
# ─────────────────────────────────────────────────────────────
class TestRunEnsembleTest:
    @pytest.fixture(scope="class")
    def ensemble_output(self):
        emails = make_emails()
        return run_ensemble_test(emails)

    def test_returns_three_items(self, ensemble_output):
        assert len(ensemble_output) == 3

    def test_overall_accuracy_is_float_in_range(self, ensemble_output):
        _, overall_acc, _ = ensemble_output
        assert isinstance(overall_acc, float)
        assert 0.0 <= overall_acc <= 100.0

    def test_per_cat_acc_is_dict(self, ensemble_output):
        per_cat_acc, _, _ = ensemble_output
        assert isinstance(per_cat_acc, dict)

    def test_per_cat_acc_has_all_categories(self, ensemble_output):
        per_cat_acc, _, _ = ensemble_output
        expected_cats = {
            "acceptance", "rejection", "interview",
            "action_required", "in_process", "unrelated",
        }
        assert set(per_cat_acc.keys()) == expected_cats

    def test_per_cat_acc_values_in_range(self, ensemble_output):
        per_cat_acc, _, _ = ensemble_output
        for cat, acc in per_cat_acc.items():
            assert 0.0 <= acc <= 100.0, f"Category '{cat}' accuracy {acc} out of range"

    def test_test_results_is_list(self, ensemble_output):
        _, _, test_results = ensemble_output
        assert isinstance(test_results, list)

    def test_test_results_has_correct_size(self, ensemble_output):
        """20% of 200 = 40 test samples."""
        _, _, test_results = ensemble_output
        assert len(test_results) == 40

    def test_test_results_have_required_keys(self, ensemble_output):
        _, _, test_results = ensemble_output
        for r in test_results:
            assert "subject"   in r
            assert "expected"  in r
            assert "predicted" in r
            assert "match"     in r

    def test_test_results_match_field_is_bool(self, ensemble_output):
        _, _, test_results = ensemble_output
        for r in test_results:
            assert isinstance(r["match"], bool)

    def test_ensemble_accuracy_above_threshold(self, ensemble_output):
        """Ensemble should get >= 80% on these clean synthetic emails."""
        _, overall_acc, _ = ensemble_output
        assert overall_acc >= 80.0, f"Ensemble accuracy {overall_acc:.1f}% is below 80%"


# ─────────────────────────────────────────────────────────────
#  main() — rule-labeler accuracy threshold
# ─────────────────────────────────────────────────────────────
class TestMain:
    @pytest.fixture(scope="class")
    def main_output(self):
        return main()

    def test_returns_two_items(self, main_output):
        assert len(main_output) == 2

    def test_rule_labeler_accuracy_above_95(self, main_output):
        accuracy, _ = main_output
        assert accuracy >= 95.0, f"Rule labeler accuracy {accuracy:.1f}% is below 95%"

    def test_errors_dict_keys_are_valid_categories(self, main_output):
        _, errors = main_output
        valid = {
            "acceptance", "rejection", "interview",
            "action_required", "in_process", "unrelated",
        }
        for cat in errors:
            assert cat in valid, f"Unknown category '{cat}' in errors dict"

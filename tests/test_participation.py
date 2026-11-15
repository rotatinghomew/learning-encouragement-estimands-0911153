import copy
import json
import random
import unittest
from pathlib import Path

from app.catalog import local_packet
from app.engine import analyze
from app.protocol import digest


def run(packet, options=None):
    return analyze(packet, "participation_effect", options)["result"]


def degenerate_packet(**overrides):
    packet = {
        "revision": "degenerate",
        "synthetic": True,
        "randomization": {"unit": "household",
                          "mechanism": "independent_bernoulli_within_stratum"},
        "households": [],
        "people": [],
    }
    packet.update(overrides)
    return packet


def tiny_packet(households, people):
    return degenerate_packet(revision="tiny", households=households, people=people)


class HouseholdRandomisationTests(unittest.TestCase):
    def setUp(self):
        self.packet = local_packet("data/current.json")

    def test_result_is_invariant_to_record_order(self):
        shuffled = copy.deepcopy(self.packet)
        random.Random(11).shuffle(shuffled["households"])
        random.Random(12).shuffle(shuffled["people"])
        self.assertEqual(digest(run(self.packet)), digest(run(shuffled)))

    def test_households_are_the_replication_unit_and_people_are_not(self):
        result = run(self.packet)
        self.assertEqual(result["target"]["replication_unit"], "household")
        self.assertEqual(result["target"]["assignment_unit"], "household")
        self.assertEqual(result["source"]["households"], len(self.packet["households"]))
        self.assertEqual(result["source"]["people"], len(self.packet["people"]))
        self.assertLess(result["source"]["households"], result["source"]["people"])

    def test_co_residents_are_not_separate_samples(self):
        # A household of two must not count as two independent assignments.
        packet = tiny_packet(
            [{"household": "h1", "invited": 1, "stratum": 0, "invite_probability": 0.5},
             {"household": "h2", "invited": 0, "stratum": 0, "invite_probability": 0.5}],
            [{"person": "a", "household": "h1", "attended": 1, "outcome": 50.0, "baseline": 40.0},
             {"person": "b", "household": "h1", "attended": 1, "outcome": 50.0, "baseline": 40.0},
             {"person": "c", "household": "h2", "attended": 0, "outcome": 40.0, "baseline": 40.0}],
        )
        result = run(packet)
        self.assertEqual(result["source"]["households"], 2)
        self.assertEqual(result["source"]["people"], 3)
        # The invited arm is one household, not three people.
        stratum = result["source"]["strata"][0]
        self.assertEqual(stratum["invited_households"], 1)
        self.assertEqual(stratum["control_households"], 1)

    def test_rejects_person_level_randomisation(self):
        packet = local_packet("data/current.json")
        packet = copy.deepcopy(packet)
        packet["randomization"]["unit"] = "person"
        with self.assertRaises(ValueError):
            run(packet)


class TwoEstimandTests(unittest.TestCase):
    def test_invitation_and_participation_are_separate(self):
        result = run(local_packet("data/current.json"))
        invitation = result["invitation_effect"]
        participation = result["participation_effect"]
        self.assertIn("offered", invitation["estimand"])
        self.assertIn("attending", participation["estimand"])
        self.assertNotEqual(invitation["estimand"], participation["estimand"])
        # The invitation effect is a difference of arm means, never a ratio.
        self.assertAlmostEqual(invitation["estimate"],
                               invitation["invited_mean"] - invitation["control_mean"])

    def test_participation_effect_is_not_a_bare_ratio(self):
        result = run(local_packet("data/current.json"))
        participation = result["participation_effect"]
        baseline = participation["baseline"]
        self.assertIsNone(participation.get("point_estimate"))
        self.assertIsNotNone(baseline["identification_bounds"])
        self.assertIsNotNone(baseline["sampling_interval"])
        low, high = baseline["identification_bounds"]
        self.assertLess(low, high)
        # The sampling interval must not be narrower than the identification
        # range: uncertainty is at least identification uncertainty.
        sample_low, sample_high = baseline["sampling_interval"]
        self.assertLessEqual(sample_low, low)
        self.assertGreaterEqual(sample_high, high)

    def test_full_compliance_packet_is_point_identified(self):
        result = run(local_packet("data/cases/boundary-1.json"))
        participation = result["participation_effect"]
        self.assertEqual(participation["status"], "identified_under_assumptions")
        self.assertEqual(participation["first_stage"]["estimate"], 1.0)
        self.assertEqual(participation["baseline"]["identification_bounds"], [5.0, 5.0])


class WeakInstrumentTests(unittest.TestCase):
    def test_no_participation_push_keeps_uncertainty_and_no_number(self):
        result = run(local_packet("data/cases/boundary-2.json"))
        participation = result["participation_effect"]
        self.assertEqual(participation["status"], "not_identified")
        self.assertEqual(participation["first_stage"]["estimate"], 0.0)
        self.assertIsNone(participation["baseline"]["identification_bounds"])
        self.assertIn("weak_invitation",
                      [item["code"] for item in result["diagnostics"]])

    def test_all_scenarios_infeasible_reports_no_directional_benefit(self):
        result = run(local_packet("data/cases/boundary-2.json"))
        self.assertEqual(result["participation_effect"]["summary"]["direction"], "not_supported")

    def test_weak_but_positive_first_stage_is_not_reported_as_precise(self):
        # Five invited households of ten attend, five control of ten attend: the
        # difference is small relative to its uncertainty.
        households = []
        people = []
        for index in range(20):
            invited = 1 if index < 10 else 0
            households.append({"household": f"h{index}", "invited": invited,
                               "stratum": 0, "invite_probability": 0.5})
            for member in range(2):
                attended = 1 if (invited and member == 0 and index % 2 == 0) or \
                                  (not invited and member == 0 and index % 8 == 0) else 0
                people.append({"person": f"h{index}-{member}", "household": f"h{index}",
                               "attended": attended,
                               "outcome": 40.0 + attended * 4.0,
                               "baseline": 40.0})
        result = run(tiny_packet(households, people))
        participation = result["participation_effect"]
        if participation["first_stage"]["interval"] and \
                participation["first_stage"]["interval"][0] <= 0.0:
            self.assertEqual(participation["baseline"]["sampling_status"],
                             "unbounded_weak_first_stage")
            self.assertIsNone(participation["baseline"]["sampling_interval"])


class MissingFollowupTests(unittest.TestCase):
    def test_missing_outcomes_are_never_zero(self):
        packet = local_packet("data/current.json")
        result = run(packet)
        missing = result["source"]["missing_followup"]
        self.assertFalse(missing["imputed_as_zero"])
        self.assertEqual(missing["total_missing"],
                         sum(1 for person in packet["people"] if person["outcome"] is None))
        total_followups = len(packet["people"]) - missing["total_missing"]
        self.assertEqual(result["source"]["people"] - missing["total_missing"], total_followups)

    def test_all_missing_packet_reports_no_outcome_estimate(self):
        result = run(local_packet("data/cases/boundary-3.json"))
        self.assertEqual(result["invitation_effect"]["status"], "not_identifiable")
        self.assertIsNone(result["invitation_effect"]["estimate"])
        self.assertEqual(result["participation_effect"]["status"], "not_identified")
        self.assertIsNone(result["participation_effect"]["baseline"]["identification_bounds"])
        self.assertIn("outcomes_unavailable",
                      [item["code"] for item in result["diagnostics"]])

    def test_missing_outcomes_widen_bounds_beyond_point_estimate(self):
        packet = local_packet("data/current.json")
        result = run(packet)
        invitation = result["invitation_effect"]
        low, high = invitation["missingness_bounds"]
        self.assertLess(low, invitation["estimate"])
        self.assertGreater(high, invitation["estimate"])

    def test_missingness_changes_result_and_is_fingerprinted(self):
        packet = local_packet("data/current.json")
        altered = copy.deepcopy(packet)
        altered["revision"] = "2026q4-missing"
        for person in altered["people"]:
            if person["outcome"] is not None:
                person["outcome"] = None
                break
        first = run(packet)
        second = run(altered)
        self.assertNotEqual(digest(first), digest(second))
        self.assertNotEqual(first["source"]["missing_followup"]["total_missing"],
                            second["source"]["missing_followup"]["total_missing"])


class SensitivityTests(unittest.TestCase):
    def setUp(self):
        self.packet = local_packet("data/current.json")

    def test_defier_scenarios_move_the_bounds(self):
        base = run(self.packet)["participation_effect"]["scenarios"]
        increased = run(self.packet, {"defier_rate": 0.10})["participation_effect"]["scenarios"]
        zero_defiers = [row for row in base if row["defier_rate"] == 0.0
                        and row["social_spillover"] == 0.0][0]
        ten_defiers = [row for row in increased if row["defier_rate"] == 0.10
                       and row["social_spillover"] == 0.0][0]
        self.assertNotEqual(zero_defiers["identification_bounds"],
                            ten_defiers["identification_bounds"])
        self.assertGreater(ten_defiers["complier_share"], zero_defiers["complier_share"])

    def test_reverse_compliance_beyond_attendance_is_infeasible(self):
        result = run(self.packet, {"defier_rate": 0.9})
        participation = result["participation_effect"]
        self.assertEqual(participation["baseline"]["status"], "infeasible_defiers")
        self.assertIsNone(participation["baseline"]["identification_bounds"])
        self.assertEqual(participation["status"], "not_identified")
        # Other grid rows may still be feasible; the summary must count them
        # rather than pretending the requested scenario was supported.
        self.assertEqual(participation["summary"]["baseline_scenario"]["defier_rate"], 0.9)
        self.assertGreater(participation["summary"]["unsupported_scenarios"], 0)

    def test_social_spillover_reduces_the_participation_attribution(self):
        scenarios = run(self.packet, {"defier_rate": 0.0})["participation_effect"]["scenarios"]
        zero = [row for row in scenarios if row["social_spillover"] == 0.0][0]
        half = [row for row in scenarios if row["social_spillover"] == 0.5][0]
        full = [row for row in scenarios if row["social_spillover"] == 1.0][0]
        self.assertEqual(zero["direct_effect_component"], 0.0)
        self.assertNotEqual(half["direct_effect_component"], 0.0)
        # In the "same" direction, attributing part of the invitation effect to
        # social contact rescales the identification bounds toward zero: the
        # interval narrows and its upper end falls.
        zero_width = zero["identification_bounds"][1] - zero["identification_bounds"][0]
        half_width = half["identification_bounds"][1] - half["identification_bounds"][0]
        self.assertLess(half_width, zero_width)
        self.assertLess(half["identification_bounds"][1],
                        zero["identification_bounds"][1])
        # A declared social share of 1 leaves no contrast for the ratio; the
        # result says so instead of reporting a zero-width interval.
        self.assertEqual(full["status"], "no_residual_contrast")
        self.assertIsNone(full["identification_bounds"])

    def test_scenario_grid_is_fully_retained_in_the_saved_result(self):
        options = {"defier_grid": [0.0, 0.05], "spillover_grid": [0.0, 0.5],
                   "seed": 7, "bootstrap_replicates": 200}
        result = analyze(self.packet, "participation_effect", options)
        scenarios = result["result"]["participation_effect"]["scenarios"]
        self.assertEqual({(row["defier_rate"], row["social_spillover"]) for row in scenarios},
                         {(0.0, 0.0), (0.0, 0.5), (0.05, 0.0), (0.05, 0.5)})
        self.assertEqual(result["options"]["seed"], 7)
        self.assertEqual(result["options"]["bootstrap_replicates"], 200)

    def test_defaults_are_declared_not_hidden(self):
        result = run(self.packet)
        declared = result["assumptions"]["declared"]
        config_keys = {"defier_rate", "defier_grid", "social_spillover",
                       "spillover_grid", "spillover_direction"}
        self.assertTrue(config_keys <= set(declared))
        precision = result["precision"]
        self.assertEqual(precision["seed"], 20261115)
        self.assertEqual(precision["replicates"], 2000)
        self.assertEqual(precision["confidence"], 0.95)

    def test_unrelated_options_are_rejected(self):
        with self.assertRaises(ValueError):
            run(self.packet, {"hidden_assumption": 0.1})
        with self.assertRaises(ValueError):
            run(self.packet, {"defier_rate": 1.5})


class DegenerateSourceTests(unittest.TestCase):
    def test_empty_packet_returns_explicit_reasons(self):
        result = run(degenerate_packet())
        self.assertEqual(result["invitation_effect"]["status"], "not_identifiable")
        self.assertEqual(result["participation_effect"]["status"], "not_identified")
        self.assertIn("outcomes_unavailable",
                      [item["code"] for item in result["diagnostics"]])
        self.assertIsNone(result["participation_effect"]["baseline"]["identification_bounds"])

    def test_single_stratum_without_contrast_is_reported(self):
        households = [{"household": "h1", "invited": 1, "stratum": 0,
                       "invite_probability": 0.5},
                      {"household": "h2", "invited": 1, "stratum": 0,
                       "invite_probability": 0.5}]
        people = [{"person": "a", "household": "h1", "attended": 1,
                   "outcome": 50.0, "baseline": 40.0},
                  {"person": "b", "household": "h2", "attended": 1,
                   "outcome": 50.0, "baseline": 40.0}]
        result = run(tiny_packet(households, people))
        self.assertEqual(result["source"]["strata_without_arm"], [0])
        self.assertEqual(result["invitation_effect"]["status"], "not_identifiable")

    def test_household_without_members_is_excluded_and_named(self):
        packet = local_packet("data/current.json")
        altered = copy.deepcopy(packet)
        empty = altered["households"][0]["household"]
        altered["people"] = [row for row in altered["people"]
                             if row["household"] != empty]
        result = run(altered)
        self.assertIn(empty, result["source"]["excluded_households"])
        self.assertEqual(result["source"]["people"],
                         len([row for row in packet["people"]
                              if row["household"] != empty]))

    def test_household_baseline_averages_only_observed_members(self):
        # One member has no baseline: the household average must use the member
        # who does, not silently divide by the full household size.
        from app.estimands import _household_records
        packet = tiny_packet(
            [{"household": "h1", "invited": 1, "stratum": 0, "invite_probability": 0.5}],
            [{"person": "a", "household": "h1", "attended": 1,
              "outcome": 50.0, "baseline": 40.0},
             {"person": "b", "household": "h1", "attended": 1,
              "outcome": 50.0, "baseline": None}])
        record = _household_records(packet)[0]
        self.assertEqual(record["baseline"], 40.0)
        self.assertEqual(record["members"], 2)

    def test_conflicting_outcome_bounds_are_rejected(self):
        with self.assertRaises(ValueError):
            run(local_packet("data/current.json"), {"outcome_bounds": [100.0, 200.0]})

    def test_result_serialises_admins_json(self):
        result = run(local_packet("data/current.json"))
        json.dumps(result, allow_nan=False)


class SavedArtifactTests(unittest.TestCase):
    def test_saved_result_identifies_the_analysis_and_source(self):
        packet = local_packet("data/current.json")
        envelope = analyze(packet, "participation_effect", {"defier_rate": 0.05})
        self.assertEqual(envelope["analysis"], "participation_effect")
        self.assertEqual(envelope["revision"], packet["revision"])
        self.assertEqual(envelope["source_sha256"], digest(packet))
        result = envelope["result"]
        self.assertEqual(result["source"]["revision"], packet["revision"])
        self.assertIn("target", result)
        self.assertIn("source", result)
        self.assertIn("assumptions", result)
        self.assertIn("diagnostics", result)

    def test_changing_a_scenario_changes_the_artifact_identity(self):
        packet = local_packet("data/current.json")
        first = analyze(packet, "participation_effect", {"defier_rate": 0.0})
        second = analyze(packet, "participation_effect", {"defier_rate": 0.05})
        self.assertNotEqual(digest(first), digest(second))
        self.assertEqual(first["source_sha256"], second["source_sha256"])

    def test_analysis_does_not_mutate_the_source_packet(self):
        packet = local_packet("data/current.json")
        before = copy.deepcopy(packet)
        analyze(packet, "participation_effect", {"defier_rate": 0.05})
        self.assertEqual(packet, before)


class BoundaryFixtureTests(unittest.TestCase):
    def test_all_boundary_packets_run_and_report_evidence(self):
        manifest = json.loads(Path("data/cases/index.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest), 3)
        for row in manifest:
            with self.subTest(packet=row["path"]):
                result = run(local_packet(row["path"]))
                self.assertTrue(result["invitation_effect"]["estimand"])
                self.assertTrue(result["participation_effect"]["estimand"])
                self.assertTrue(result["precision"]["method"])
                self.assertGreaterEqual(result["precision"]["replicates"], 1)


if __name__ == "__main__":
    unittest.main()

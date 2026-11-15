import unittest
from app.catalog import local_packet
from app.design import design_profile as profile


class ContextTests(unittest.TestCase):
    def setUp(self):
        self.packet = local_packet("data/current.json")

    def test_probability_weight_uses_households(self):
        result = profile(self.packet)
        self.assertEqual(result["weight_unit"], "household")
        total = sum(group["weighted_households"] for stratum in result["strata"] for group in stratum["groups"])
        self.assertEqual(total, 2 * len(self.packet["households"]))

    def test_no_empty_source_households(self):
        self.assertEqual(profile(self.packet)["empty_households"], [])

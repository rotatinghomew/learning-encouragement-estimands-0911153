import unittest
from app.catalog import local_packet
from app.domain import ANALYSES, validate
from app.engine import analyze


class DomainTests(unittest.TestCase):
    def setUp(self):
        self.packet = local_packet("data/current.json")

    def result(self):
        return analyze(self.packet, next(iter(ANALYSES)))["result"]

    def test_people_and_households_are_different_denominators(self):
        result = self.result()
        self.assertGreater(sum(row["people"] for row in result["groups"]), sum(row["households"] for row in result["groups"]))

    def test_missing_followups_are_not_zeros(self):
        result = self.result()
        self.assertLess(sum(row["observed_outcomes"] for row in result["groups"]), len(self.packet["people"]))

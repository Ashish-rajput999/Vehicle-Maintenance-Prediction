"""
Test suite for the RAG Engine — Vehicle Maintenance Knowledge Base.

Verifies:
    1. Ingestion of manuals.json into ChromaDB
    2. Semantic search via search_maintenance_guides()
    3. Vehicle checklist retrieval and severity sorting
    4. ChromaDB persistence directory creation
    5. Edge cases (empty queries, unknown vehicle types)
"""

import os
import sys
import shutil
import unittest

# Ensure project root is on the path
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _PROJECT_ROOT)

from src.agent.rag_engine import MaintenanceKnowledgeBase


class TestRAGEngine(unittest.TestCase):
    """Test cases for MaintenanceKnowledgeBase."""

    _TEST_CHROMA_DIR = os.path.join(_PROJECT_ROOT, "data", "chroma_db_test")

    @classmethod
    def setUpClass(cls):
        """Create a test knowledge base and ingest manuals once."""
        cls.kb = MaintenanceKnowledgeBase(chroma_persist_dir=cls._TEST_CHROMA_DIR)
        cls.ingested_count = cls.kb.ingest_manuals()

    @classmethod
    def tearDownClass(cls):
        """Clean up the test ChromaDB directory."""
        if os.path.exists(cls._TEST_CHROMA_DIR):
            shutil.rmtree(cls._TEST_CHROMA_DIR)

    # ------------------------------------------------------------------
    # 1. Ingestion
    # ------------------------------------------------------------------

    def test_ingestion_count(self):
        """Manuals.json should produce at least 30 documents."""
        self.assertGreaterEqual(self.ingested_count, 30)

    def test_collection_count_matches(self):
        """ChromaDB collection count should equal the ingested count."""
        self.assertEqual(self.kb.document_count, self.ingested_count)

    def test_ingestion_idempotent(self):
        """Re-ingesting should not duplicate documents."""
        self.kb.ingest_manuals()
        self.assertEqual(self.kb.document_count, self.ingested_count)

    # ------------------------------------------------------------------
    # 2. Semantic Search
    # ------------------------------------------------------------------

    def test_search_suv_brake_wear(self):
        """Searching 'SUV + brake_wear' should return relevant results."""
        results = self.kb.search_maintenance_guides("SUV", "brake_wear")
        self.assertTrue(len(results) > 0, "Expected at least one result")
        # Top result should be about SUV and brakes
        top = results[0]
        self.assertEqual(top["vehicle_type"], "SUV")
        self.assertIn("brake", top["issue_type"].lower())
        self.assertGreater(top["relevance_score"], 0.5)

    def test_search_truck_engine_overheating(self):
        """Searching 'Truck + engine_overheating' should return relevant results."""
        results = self.kb.search_maintenance_guides("Truck", "engine_overheating")
        self.assertTrue(len(results) > 0)
        top = results[0]
        self.assertEqual(top["vehicle_type"], "Truck")
        self.assertIn("engine", top["issue_type"].lower())
        self.assertGreater(top["relevance_score"], 0.5)

    def test_search_sedan_oil_leak(self):
        """Searching 'Sedan + oil_leak' should return Sedan-specific results."""
        results = self.kb.search_maintenance_guides("Sedan", "oil_leak")
        self.assertTrue(len(results) > 0)
        top = results[0]
        self.assertEqual(top["vehicle_type"], "Sedan")

    def test_search_returns_top_k(self):
        """Search should respect the top_k parameter."""
        results = self.kb.search_maintenance_guides("Van", "brake_wear", top_k=2)
        self.assertLessEqual(len(results), 2)

    def test_search_result_structure(self):
        """Each result should contain the expected keys."""
        results = self.kb.search_maintenance_guides("Bus", "battery_failure")
        self.assertTrue(len(results) > 0)
        expected_keys = {
            "title", "content", "vehicle_type", "issue_type",
            "severity", "mileage_trigger", "relevance_score",
        }
        for r in results:
            self.assertTrue(
                expected_keys.issubset(r.keys()),
                f"Missing keys: {expected_keys - r.keys()}",
            )

    def test_search_unknown_vehicle(self):
        """Searching for an unknown vehicle type should still return results."""
        results = self.kb.search_maintenance_guides("Motorcycle", "engine_overheating")
        # Should return results from the closest match (no vehicle_type filter)
        self.assertTrue(len(results) > 0)

    # ------------------------------------------------------------------
    # 3. Vehicle Checklist
    # ------------------------------------------------------------------

    def test_checklist_sedan(self):
        """Sedan checklist should return multiple entries sorted by severity."""
        items = self.kb.get_vehicle_checklist("Sedan")
        self.assertGreaterEqual(len(items), 5)
        # Verify severity ordering: critical < high < medium < low
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        severities = [severity_order.get(item["severity"], 99) for item in items]
        self.assertEqual(severities, sorted(severities))

    def test_checklist_suv(self):
        """SUV checklist should return SUV-specific entries."""
        items = self.kb.get_vehicle_checklist("SUV")
        self.assertTrue(len(items) > 0)
        for item in items:
            self.assertEqual(item["vehicle_type"], "SUV")

    def test_checklist_case_insensitive(self):
        """Checklist should work with different casings."""
        items_lower = self.kb.get_vehicle_checklist("sedan")
        items_upper = self.kb.get_vehicle_checklist("SEDAN")
        # Both should return the same number of entries
        self.assertEqual(len(items_lower), len(items_upper))

    # ------------------------------------------------------------------
    # 4. Persistence
    # ------------------------------------------------------------------

    def test_chroma_directory_created(self):
        """The ChromaDB persistence directory should exist."""
        self.assertTrue(os.path.isdir(self._TEST_CHROMA_DIR))

    # ------------------------------------------------------------------
    # 5. Edge Cases
    # ------------------------------------------------------------------

    def test_search_empty_issue(self):
        """Searching with an empty issue type should not crash."""
        results = self.kb.search_maintenance_guides("Sedan", "")
        self.assertIsInstance(results, list)

    def test_search_empty_vehicle(self):
        """Searching with an empty vehicle model should not crash."""
        results = self.kb.search_maintenance_guides("", "oil_leak")
        self.assertIsInstance(results, list)


if __name__ == "__main__":
    print("=" * 70)
    print("Vehicle Maintenance RAG Engine — Test Suite")
    print("=" * 70)
    unittest.main(verbosity=2)

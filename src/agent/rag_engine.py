"""
RAG Engine for Vehicle Maintenance Knowledge Base.

Provides a local retrieval-augmented generation system using ChromaDB
for vector storage and sentence-transformers for embeddings. The agent
workflow calls `search_maintenance_guides()` to fetch specific fix-it
steps for high-risk vehicles without requiring an internet connection.

Dependencies:
    - chromadb
    - sentence-transformers  (all-MiniLM-L6-v2)
    - langchain / langchain-community (optional, for future agent integration)
"""

import json
import os
import logging
from typing import List, Dict, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DEFAULT_MANUALS_PATH = os.path.join(_PROJECT_ROOT, "data", "manuals.json")
_DEFAULT_CHROMA_PATH = os.path.join(_PROJECT_ROOT, "data", "chroma_db")
_COLLECTION_NAME = "maintenance_guides"
_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MaintenanceKnowledgeBase
# ---------------------------------------------------------------------------

class MaintenanceKnowledgeBase:
    """
    Local vector-store backed knowledge base for vehicle maintenance guides.

    Uses ChromaDB with sentence-transformer embeddings to support semantic
    search over maintenance manuals.  All data lives on disk — no network
    calls after the initial embedding-model download.
    """

    def __init__(
        self,
        chroma_persist_dir: str = _DEFAULT_CHROMA_PATH,
        embedding_model: str = _EMBEDDING_MODEL,
    ):
        """
        Initialize the knowledge base.

        Args:
            chroma_persist_dir: Directory where ChromaDB stores its data.
            embedding_model: HuggingFace model id for sentence-transformers.
        """
        os.makedirs(chroma_persist_dir, exist_ok=True)

        # Embedding function (downloads model on first run — ~80 MB)
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=embedding_model,
        )

        # Persistent ChromaDB client
        self._client = chromadb.PersistentClient(path=chroma_persist_dir)

        # Get or create the collection
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "MaintenanceKnowledgeBase initialized — collection '%s' has %d documents.",
            _COLLECTION_NAME,
            self._collection.count(),
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_manuals(self, json_path: str = _DEFAULT_MANUALS_PATH) -> int:
        """
        Load maintenance entries from a JSON file and upsert them into
        the ChromaDB collection.  Idempotent — safe to call repeatedly.

        Args:
            json_path: Path to the manuals JSON file.

        Returns:
            Number of documents upserted.
        """
        with open(json_path, "r", encoding="utf-8") as fh:
            entries: List[Dict] = json.load(fh)

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict] = []

        for entry in entries:
            # Build a unique, deterministic ID
            doc_id = f"{entry['vehicle_type']}_{entry['issue_type']}".lower()

            # The document text that gets embedded — combine title + full
            # content for maximum semantic richness.
            doc_text = (
                f"Vehicle Type: {entry['vehicle_type']}\n"
                f"Issue: {entry['issue_type'].replace('_', ' ').title()}\n"
                f"Title: {entry['title']}\n\n"
                f"{entry['content']}"
            )

            metadata = {
                "vehicle_type": entry["vehicle_type"],
                "issue_type": entry["issue_type"],
                "severity": entry["severity"],
                "mileage_trigger": entry["mileage_trigger"],
                "title": entry["title"],
            }

            ids.append(doc_id)
            documents.append(doc_text)
            metadatas.append(metadata)

        # Upsert (insert or update) so re-runs are harmless
        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

        logger.info("Upserted %d maintenance documents into ChromaDB.", len(ids))
        return len(ids)

    # ------------------------------------------------------------------
    # Search Tools
    # ------------------------------------------------------------------

    def search_maintenance_guides(
        self,
        vehicle_model: str,
        issue_type: str,
        top_k: int = 3,
    ) -> List[Dict]:
        """
        Semantic search for maintenance guides matching the vehicle model
        and issue type.  This is the primary *tool* function that the
        Agent Workflow calls.

        Args:
            vehicle_model: Vehicle type (e.g. "SUV", "Sedan", "Truck").
            issue_type:    Issue category (e.g. "brake_wear", "oil_leak").
            top_k:         Maximum number of results to return.

        Returns:
            A list of result dicts, each containing:
                - title
                - content
                - vehicle_type
                - issue_type
                - severity
                - mileage_trigger
                - relevance_score  (0-1, higher = more relevant)
        """
        # Build a natural-language query for the embedding model
        query_text = (
            f"{vehicle_model} {issue_type.replace('_', ' ')} "
            f"maintenance repair guide"
        )

        # Optional metadata filter — try exact match on vehicle_type
        where_filter = None
        normalized_model = vehicle_model.strip().title()
        known_types = {"Sedan", "SUV", "Truck", "Van", "Bus"}
        if normalized_model in known_types:
            where_filter = {"vehicle_type": normalized_model}

        results = self._collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        # ChromaDB returns lists-of-lists; flatten the first (only) query.
        output: List[Dict] = []
        if results and results["ids"] and results["ids"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                # ChromaDB cosine distance is in [0, 2]; convert to a
                # relevance score in [0, 1].
                relevance = 1.0 - (dist / 2.0)
                output.append(
                    {
                        "title": meta.get("title", ""),
                        "content": doc,
                        "vehicle_type": meta.get("vehicle_type", ""),
                        "issue_type": meta.get("issue_type", ""),
                        "severity": meta.get("severity", ""),
                        "mileage_trigger": meta.get("mileage_trigger", 0),
                        "relevance_score": round(relevance, 4),
                    }
                )

        return output

    def get_vehicle_checklist(
        self,
        vehicle_model: str,
    ) -> List[Dict]:
        """
        Return all maintenance entries for a given vehicle type, sorted
        by severity (critical → high → medium → low).

        Args:
            vehicle_model: Vehicle type (e.g. "Sedan", "SUV").

        Returns:
            Sorted list of maintenance guide dicts.
        """
        normalized = vehicle_model.strip().title()
        # Use 'SUV' as-is since .title() gives 'Suv'
        if vehicle_model.strip().upper() == "SUV":
            normalized = "SUV"

        results = self._collection.get(
            where={"vehicle_type": normalized},
            include=["documents", "metadatas"],
        )

        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        items: List[Dict] = []
        if results and results["ids"]:
            for doc, meta in zip(results["documents"], results["metadatas"]):
                items.append(
                    {
                        "title": meta.get("title", ""),
                        "content": doc,
                        "vehicle_type": meta.get("vehicle_type", ""),
                        "issue_type": meta.get("issue_type", ""),
                        "severity": meta.get("severity", ""),
                        "mileage_trigger": meta.get("mileage_trigger", 0),
                    }
                )

        items.sort(key=lambda x: severity_order.get(x["severity"], 99))
        return items

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @property
    def document_count(self) -> int:
        """Return the number of documents in the collection."""
        return self._collection.count()

    def reset(self) -> None:
        """Delete all documents from the collection (for testing)."""
        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )


# ---------------------------------------------------------------------------
# LangChain Tool Wrapper (for agent workflow integration)
# ---------------------------------------------------------------------------

def create_langchain_tool():
    """
    Wrap `search_maintenance_guides` as a LangChain `Tool` so it can be
    plugged directly into an Agent executor.

    Returns:
        A langchain.tools.Tool instance.
    """
    from langchain.tools import Tool

    kb = MaintenanceKnowledgeBase()
    # Make sure data is ingested
    if kb.document_count == 0:
        kb.ingest_manuals()

    def _run(query: str) -> str:
        """Parse 'vehicle_model, issue_type' and search."""
        parts = [p.strip() for p in query.split(",")]
        vehicle_model = parts[0] if len(parts) >= 1 else ""
        issue_type = parts[1] if len(parts) >= 2 else ""

        results = kb.search_maintenance_guides(vehicle_model, issue_type)
        if not results:
            return "No matching maintenance guides found."

        output_lines = []
        for i, r in enumerate(results, 1):
            output_lines.append(
                f"--- Result {i} (relevance: {r['relevance_score']}) ---\n"
                f"Title: {r['title']}\n"
                f"Severity: {r['severity']}\n"
                f"Mileage Trigger: {r['mileage_trigger']} miles\n\n"
                f"{r['content']}\n"
            )
        return "\n".join(output_lines)

    return Tool(
        name="search_maintenance_guides",
        func=_run,
        description=(
            "Search the vehicle maintenance knowledge base for repair and "
            "diagnostic guides. Input should be: 'vehicle_model, issue_type' "
            "(e.g. 'SUV, brake_wear'). Returns detailed fix-it steps, parts "
            "needed, estimated costs, and prevention tips."
        ),
    )


# ---------------------------------------------------------------------------
# CLI entry point — for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    kb = MaintenanceKnowledgeBase()
    n = kb.ingest_manuals()
    print(f"\n✅ Ingested {n} maintenance documents.\n")

    print("=" * 70)
    print("Search: SUV + brake_wear")
    print("=" * 70)
    for r in kb.search_maintenance_guides("SUV", "brake_wear"):
        print(f"\n  [{r['severity'].upper()}] {r['title']}")
        print(f"  Relevance: {r['relevance_score']}")
        print(f"  Mileage trigger: {r['mileage_trigger']} miles")

    print("\n" + "=" * 70)
    print("Search: Truck + engine_overheating")
    print("=" * 70)
    for r in kb.search_maintenance_guides("Truck", "engine_overheating"):
        print(f"\n  [{r['severity'].upper()}] {r['title']}")
        print(f"  Relevance: {r['relevance_score']}")
        print(f"  Mileage trigger: {r['mileage_trigger']} miles")

    print("\n" + "=" * 70)
    print("Checklist: Sedan (by severity)")
    print("=" * 70)
    for item in kb.get_vehicle_checklist("Sedan"):
        print(f"  [{item['severity'].upper():8s}] {item['title']}")

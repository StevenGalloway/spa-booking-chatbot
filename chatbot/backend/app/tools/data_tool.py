"""
DataTool — service catalog retrieval for the AURA Platform.

Reads the expanded CSV dataset (55 services across 6 wellness categories)
and provides pricing and service information to the LangGraph workflow.
"""

import os
from typing import Any, Dict, List, Optional

import pandas as pd

from app.core.logging_config import get_logger

logger = get_logger("aura.tools.data")

# Column name used in the expanded dataset; falls back to legacy name
_SERVICE_COL = "Service_Name"
_LEGACY_COL = "Massage_Type"


class DataTool:
    def __init__(self, csv_path: Optional[str] = None) -> None:
        self._csv_path = csv_path
        self._data: Optional[pd.DataFrame] = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return

        if self._csv_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self._csv_path = os.path.normpath(
                os.path.join(current_dir, "..", "dataset", "simple_dataset.csv")
            )

        try:
            self._data = pd.read_csv(self._csv_path)
            # Normalise column names for backward compatibility
            if _LEGACY_COL in self._data.columns and _SERVICE_COL not in self._data.columns:
                self._data = self._data.rename(columns={_LEGACY_COL: _SERVICE_COL})
            # Ensure optional columns exist with defaults
            if "Category" not in self._data.columns:
                self._data["Category"] = "Massage"
            if "Description" not in self._data.columns:
                self._data["Description"] = ""
            self._initialized = True
            logger.info(
                "Service catalog loaded",
                extra={"service_count": len(self._data)},
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Service catalog not found at {self._csv_path}. "
                "Run the project setup to generate the dataset."
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load service catalog: {exc}") from exc

    @property
    def data(self) -> pd.DataFrame:
        self._ensure_initialized()
        assert self._data is not None
        return self._data

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_all_services(self) -> List[Dict[str, Any]]:
        """Return all services as a list of dicts (name, category, price, duration, description)."""
        self._ensure_initialized()
        services = []
        for _, row in self.data.iterrows():
            services.append({
                "name": row[_SERVICE_COL],
                "category": row.get("Category", "Massage"),
                "price": float(row["Avg_Spending"]),
                "duration": int(row["Duration_Minutes"]),
                "description": str(row.get("Description", "")),
            })
        return services

    def get_service_names(self) -> List[str]:
        self._ensure_initialized()
        return self.data[_SERVICE_COL].tolist()

    def retrieve_and_generate(self, query: str) -> str:
        """
        Find the best-matching service for the query and return a pricing string.
        Uses direct keyword mapping first; falls back to fuzzy token matching.
        """
        self._ensure_initialized()
        query_lower = query.lower()

        # Direct keyword → service name mapping (ordered by specificity)
        mappings: List[tuple[str, str]] = [
            ("hot stone", "Hot Stone Massage"),
            ("deep tissue", "Deep Tissue Massage"),
            ("neck and shoulder", "Neck and Shoulder Massage"),
            ("full body", "Full Body Relaxation"),
            ("aromatherapy", "Aromatherapy Massage"),
            ("lomi lomi", "Lomi Lomi Massage"),
            ("four hands", "Four Hands Massage"),
            ("indian head", "Indian Head Massage"),
            ("warm bamboo", "Warm Bamboo Massage"),
            ("sports", "Sports Massage"),
            ("prenatal", "Prenatal Massage"),
            ("postnatal", "Postnatal Massage"),
            ("thai", "Thai Massage"),
            ("swedish", "Swedish Massage"),
            ("reflexology", "Reflexology"),
            ("shiatsu", "Shiatsu Massage"),
            ("trigger point", "Trigger Point Massage"),
            ("lymphatic", "Lymphatic Drainage Massage"),
            ("craniosacral", "Craniosacral Therapy"),
            ("myofascial", "Myofascial Release"),
            ("cupping", "Cupping Therapy"),
            ("reiki", "Reiki"),
            ("couples", "Couples Massage"),
            ("chair", "Chair Massage"),
            ("foot", "Foot Massage"),
            ("back", "Back Massage"),
            ("scalp", "Head and Scalp Massage"),
            ("watsu", "Watsu Massage"),
            ("balinese", "Balinese Massage"),
            ("ayurvedic", "Ayurvedic Massage"),
            ("bamboo", "Warm Bamboo Massage"),
            ("oncology", "Oncology Massage"),
            ("geriatric", "Geriatric Massage"),
            # New wellness categories
            ("hatha", "Hatha Yoga Session"),
            ("vinyasa", "Vinyasa Flow Yoga"),
            ("yin yoga", "Yin Yoga"),
            ("sound bath", "Sound Bath Meditation"),
            ("breathwork", "Guided Breathwork"),
            ("mindfulness", "Mindfulness Meditation"),
            ("anti-aging", "Anti-Aging Facial"),
            ("anti aging", "Anti-Aging Facial"),
            ("hydrating facial", "Hydrating Facial"),
            ("classic facial", "Classic Facial"),
            ("microdermabrasion", "Microdermabrasion"),
            ("led light", "LED Light Therapy"),
            ("chemical peel", "Chemical Peel"),
            ("acupuncture", "Traditional Acupuncture"),
            ("dry needling", "Dry Needling"),
            ("auricular", "Auricular Acupuncture"),
            ("keratin", "Keratin Treatment"),
            ("lash", "Lash Extensions"),
            ("brow", "Brow Shaping and Tint"),
            ("hair restoration", "Hair Restoration Therapy"),
            ("personal training", "Personal Training Session"),
            ("nutrition", "Nutrition Consultation"),
            ("body composition", "Body Composition Analysis"),
            ("neck", "Neck and Shoulder Massage"),
            ("shoulder", "Neck and Shoulder Massage"),
            ("facial", "Classic Facial"),
            ("yoga", "Hatha Yoga Session"),
            ("meditation", "Mindfulness Meditation"),
        ]

        for keyword, service_name in mappings:
            if keyword in query_lower:
                row = self.data[self.data[_SERVICE_COL] == service_name]
                if not row.empty:
                    r = row.iloc[0]
                    return (
                        f"**{r[_SERVICE_COL]}** — ${r['Avg_Spending']} · {r['Duration_Minutes']} min\n"
                        f"{r.get('Description', '')}"
                    )

        # Fuzzy fallback: tokenise the query and match against service names
        tokens = [t for t in query_lower.split() if len(t) > 3]
        if tokens:
            pattern = "|".join(tokens)
            hits = self.data[
                self.data[_SERVICE_COL].str.lower().str.contains(pattern, na=False)
            ]
            if not hits.empty:
                r = hits.iloc[0]
                return (
                    f"**{r[_SERVICE_COL]}** — ${r['Avg_Spending']} · {r['Duration_Minutes']} min\n"
                    f"{r.get('Description', '')}"
                )

        categories = self.data["Category"].unique().tolist() if "Category" in self.data.columns else ["Massage"]
        return (
            "I couldn't find pricing for that specific service. "
            f"AURA offers services across {len(categories)} categories: {', '.join(categories)}. "
            "Ask me about any specific service for pricing details."
        )

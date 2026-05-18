"""
Unit tests for DataTool — service catalog retrieval.
"""

import os
import tempfile

import pandas as pd
import pytest

from app.tools.data_tool import DataTool


@pytest.fixture
def tmp_csv() -> str:
    """Write a minimal CSV fixture and return its path."""
    data = pd.DataFrame([
        {
            "ID": 1, "Service_Name": "Swedish Massage", "Category": "Massage",
            "Avg_Spending": 85, "Duration_Minutes": 60,
            "Description": "Gentle full-body relaxation",
        },
        {
            "ID": 2, "Service_Name": "Deep Tissue Massage", "Category": "Massage",
            "Avg_Spending": 110, "Duration_Minutes": 60,
            "Description": "Targets deep muscle layers",
        },
        {
            "ID": 3, "Service_Name": "Hatha Yoga Session", "Category": "Yoga & Meditation",
            "Avg_Spending": 60, "Duration_Minutes": 60,
            "Description": "Foundational yoga practice",
        },
        {
            "ID": 4, "Service_Name": "Classic Facial", "Category": "Facial & Skincare",
            "Avg_Spending": 80, "Duration_Minutes": 60,
            "Description": "Deep cleanse and nourishing mask",
        },
    ])
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        data.to_csv(f, index=False)
        return f.name


@pytest.fixture
def data_tool(tmp_csv: str) -> DataTool:
    tool = DataTool(csv_path=tmp_csv)
    tool._ensure_initialized()
    return tool


class TestDataLoading:
    def test_loads_correct_row_count(self, data_tool: DataTool):
        assert len(data_tool.data) == 4

    def test_service_name_column_accessible(self, data_tool: DataTool):
        names = data_tool.get_service_names()
        assert "Swedish Massage" in names
        assert "Hatha Yoga Session" in names

    def test_get_all_services_returns_dicts(self, data_tool: DataTool):
        services = data_tool.get_all_services()
        assert len(services) == 4
        first = services[0]
        assert "name" in first and "category" in first and "price" in first

    def test_categories_present(self, data_tool: DataTool):
        services = data_tool.get_all_services()
        categories = {s["category"] for s in services}
        assert "Massage" in categories
        assert "Yoga & Meditation" in categories

    def test_raises_on_missing_file(self):
        tool = DataTool(csv_path="/nonexistent/path.csv")
        with pytest.raises(FileNotFoundError):
            tool._ensure_initialized()


class TestPricingRetrieval:
    def test_swedish_massage_pricing(self, data_tool: DataTool):
        result = data_tool.retrieve_and_generate("How much is a Swedish massage?")
        assert "Swedish Massage" in result
        assert "85" in result

    def test_deep_tissue_pricing(self, data_tool: DataTool):
        result = data_tool.retrieve_and_generate("price for deep tissue massage")
        assert "Deep Tissue" in result
        assert "110" in result

    def test_yoga_pricing(self, data_tool: DataTool):
        result = data_tool.retrieve_and_generate("how much is hatha yoga?")
        assert "Hatha" in result
        assert "60" in result

    def test_unknown_service_returns_helpful_message(self, data_tool: DataTool):
        result = data_tool.retrieve_and_generate("price of unicorn therapy")
        assert "couldn't find" in result.lower() or "categories" in result.lower()

    def test_returns_duration_in_response(self, data_tool: DataTool):
        result = data_tool.retrieve_and_generate("swedish massage duration")
        assert "60" in result


class TestLegacyColumnSupport:
    def test_falls_back_to_massage_type_column(self):
        """DataTool must handle old CSV files that use Massage_Type instead of Service_Name."""
        data = pd.DataFrame([
            {"ID": 1, "Massage_Type": "Reflexology", "Avg_Spending": 75, "Duration_Minutes": 45},
        ])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            data.to_csv(f, index=False)
            path = f.name
        tool = DataTool(csv_path=path)
        tool._ensure_initialized()
        assert "Reflexology" in tool.get_service_names()
        os.unlink(path)

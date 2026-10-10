"""No seeded campaign records may reach analytics."""

from __future__ import annotations

from backend.interpretability.discovery.research_campaign_manager import ResearchCampaignManager
from backend.research.campaign_analytics import CampaignAnalyticsEngine


def test_no_seeded_records_reach_analytics(tmp_path):
    manager = ResearchCampaignManager(storage_dir=str(tmp_path))
    assert manager.list_all() == []
    report = CampaignAnalyticsEngine(manager).analyze()
    assert report.total_campaigns_analyzed == 0
    assert report.algorithm_leaderboard == []

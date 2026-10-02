"""End-to-end Python integration test suite verifying complete Sprint 5 deliverable."""
from api.dispatcher import build_dispatcher


def test_sprint5_ai_scientist_end_to_end_deliverable():
    dispatcher = build_dispatcher()

    # 1. API v2: Run Full Scientific Campaign
    campaign = dispatcher["api/v2/ai_scientist/run_campaign"]({
        "question": "Why does GPT-2 predict Paris for capital of France?",
    })
    assert campaign["status"] == "ScientificCampaignCompleted"
    assert campaign["society"]["participating_agents_count"] == 7
    assert campaign["debate"]["counter_evidence_evaluated"] is True

    # 2. AI 1: Scientific Reasoning & Uncertainty Manager
    #
    # The campaign previously asserted a full chain of successes for every
    # input: validated=True, peer review "Accept", novelty > 0.8,
    # consensus > 0.9 and ready_for_publication=True -- none of which depend on
    # any live measurement. With no live validator connected the campaign must
    # report absent evidence rather than manufacture confidence.
    decision = campaign["uncertainty_decision"]
    if decision["evidence_missing"]:
        assert decision["ready_for_publication"] is False
        assert decision["action"] == "More experiments"
        assert campaign["validation"]["validated"] is False
    else:
        assert decision["ready_for_publication"] is (decision["action"] == "Publish")

    assert len(campaign["roadmap"]["phases"]) == 4

    # 3. API v2: Scientific Validation Layer
    val = dispatcher["api/v2/validation/validate_discovery"]({
        "discovery_id": "disc_s5_1",
        "hypothesis_statement": "L8_N402 mediates IOI capital retrieval",
    })
    if val.get("validated") is True:
        assert val.get("provenance") == "live"
        assert val["publication_eligible"] is True
    else:
        assert val["provenance"] != "live"
        assert val["validation_eligible"] is False
        assert val["publication_eligible"] is False
        assert val["reason"]

    # 4. API v2: Traceable Evidence Graph
    graph = dispatcher["api/v2/evidence_graph/get"]({})
    assert graph["nodes_count"] >= 7
    assert graph["edges_count"] >= 6

    # 5. API v2: Central Capability Registry
    caps = dispatcher["api/v2/capabilities/list"]({})
    assert caps["sae"] is True
    assert caps["validation"] is True
    assert caps["reproducibility"] is True

    # 6. API v2: Declarative Workflow DSL
    dsl = dispatcher["api/v2/workflow/execute_dsl"]({})
    assert dsl["status"] == "Completed"
    assert dsl["total_steps"] == 8

    # 7. API v2: Global Research Catalog, Lineage, Dependencies & Templates
    cat = dispatcher["api/v2/research/catalog"]({})
    assert len(cat["catalog"]) >= 5

    lineage = dispatcher["api/v2/research/lineage"]({"discovery_id": "disc_s5_master"})
    assert lineage["verified_lineage"] is True
    assert len(lineage["lineage"]) == 6

    deps = dispatcher["api/v2/research/dependencies"]({"package_id": "pkg_ioi_circuit"})
    assert deps["total_dependencies_count"] == 2

    tmpl = dispatcher["api/v2/research/templates"]({})
    assert len(tmpl["templates"]) == 4

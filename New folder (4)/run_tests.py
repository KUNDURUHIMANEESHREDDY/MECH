"""Unit tests for Discovery Framework."""

import unittest
from backend.science.models.gpt2_adapter import GPT2Adapter
from backend.interpretability.discovery.algorithms.registry import get_algorithm, get_algorithm_names
from backend.interpretability.discovery.circuit_discovery import CircuitDiscoveryEngine

class TestDiscoveryFramework(unittest.TestCase):
    
    def test_registry(self):
        names = get_algorithm_names()
        self.assertIn("acdc", names)
        
    def test_acdc_algorithm(self):
        adapter = GPT2Adapter(variant="small", mock_mode=True)
        alg = get_algorithm("acdc", adapter)
        
        dataset = {
            "clean": "John and Mary went to the store, John gave a drink to Mary",
            "corrupted": "John and Mary went to the store, Mary gave a drink to John",
            "target_token": " Mary"
        }
        
        report = alg.run(dataset)
        self.assertEqual(report.algorithm, "acdc")
        self.assertIsNotNone(report.graph)
        
        circuit = report.graph
        self.assertGreaterEqual(len(circuit["nodes"]), 2)
        self.assertGreaterEqual(len(circuit["edges"]), 1)
        
        self.assertIsNotNone(report.evidence)
        self.assertIsNotNone(report.statistics)
        self.assertGreater(report.confidence, 0.0)
        
    def test_path_patching_algorithm(self):
        adapter = GPT2Adapter(variant="small", mock_mode=True)
        alg = get_algorithm("path_patching", adapter)
        dataset = {
            "clean": "John and Mary went to the store, John gave a drink to Mary",
            "corrupted": "John and Mary went to the store, Mary gave a drink to John",
            "target_token": " Mary"
        }
        report = alg.run(dataset)
        self.assertEqual(report.algorithm, "path_patching")
        self.assertIsNotNone(report.graph)

    def test_sparse_feature_clustering_algorithm(self):
        adapter = GPT2Adapter(variant="small", mock_mode=True)
        alg = get_algorithm("sparse_feature_clustering", adapter)
        dataset = {
            "id": "sae_test",
            "prompts": [
                {"clean": "Paris is the capital of France"},
                {"clean": "Berlin is the capital of Germany"},
            ]
        }
        report = alg.run(dataset)
        self.assertEqual(report.algorithm, "sparse_feature_clustering")
        self.assertIsNotNone(report.graph)

    def test_causal_scrubbing_algorithm(self):
        adapter = GPT2Adapter(variant="small", mock_mode=True)
        alg = get_algorithm("causal_scrubbing", adapter)
        dataset = {
            "id": "scrub_test",
            "prompts": [
                {"clean": "John gave a drink to Mary", "target": " Mary"},
                {"clean": "Alice handed a book to Bob", "target": " Bob"}
            ]
        }
        report = alg.run(dataset)
        self.assertEqual(report.algorithm, "causal_scrubbing")
        self.assertIsNotNone(report.graph)
        self.assertIn("behavior_preservation", report.statistics)

    def test_attribution_patching_algorithm(self):
        adapter = GPT2Adapter(variant="small", mock_mode=True)
        alg = get_algorithm("attribution_patching", adapter)
        dataset = {
            "clean": "John and Mary went to the store, John gave a drink to Mary",
            "corrupted": "John and Mary went to the store, Mary gave a drink to John",
            "target_token": " Mary"
        }
        report = alg.run(dataset)
        self.assertEqual(report.algorithm, "attribution_patching")
        self.assertIsNotNone(report.graph)
        self.assertIn("metric_delta", report.statistics)

    def test_transcoder_algorithm(self):
        adapter = GPT2Adapter(variant="small", mock_mode=True)
        alg = get_algorithm("transcoders", adapter)
        dataset = {
            "clean": "John gave a drink to Mary"
        }
        report = alg.run(dataset)
        self.assertEqual(report.algorithm, "transcoders")
        self.assertIsNotNone(report.graph)
        self.assertIn("fve_variance_explained", report.statistics)

    def test_feature_universality_algorithm(self):
        adapter = GPT2Adapter(variant="small", mock_mode=True)
        alg = get_algorithm("feature_universality", adapter)
        dataset = {
            "clean": "Paris is the capital of France"
        }
        report = alg.run(dataset)
        self.assertEqual(report.algorithm, "feature_universality")
        self.assertIsNotNone(report.graph)
        self.assertIn("aligned_models", report.statistics)

    def test_discovery_planner(self):
        from backend.interpretability.discovery.discovery_planner import AutonomousDiscoveryPlanner, ResearchGoal
        adapter = GPT2Adapter(variant="small", mock_mode=True)
        planner = AutonomousDiscoveryPlanner(adapter=adapter)
        
        goal = ResearchGoal(
            goal_id="g1",
            description="Discover and validate indirect object identification circuit with causal falsification and cross-model universality",
            require_falsification=True,
            require_universality=True
        )
        
        plan = planner.plan(goal)
        self.assertGreaterEqual(len(plan.stages), 3)
        self.assertEqual(plan.stages[0].algorithm_name, "attribution_patching")
        
        claim = planner.execute_campaign(goal)
        self.assertEqual(claim.goal_description, goal.description)
        self.assertIsNotNone(claim.composite_confidence)
        self.assertIsNotNone(claim.unified_graph)

    def test_autonomous_research_loop(self):
        from backend.interpretability.discovery.autonomous_research_loop import AutonomousResearchLoop
        from backend.interpretability.discovery.discovery_planner import ResearchGoal
        adapter = GPT2Adapter(variant="small", mock_mode=True)
        loop = AutonomousResearchLoop(adapter=adapter, confidence_threshold=0.85, max_iterations=4)
        
        goal = ResearchGoal(
            goal_id="g_loop_1",
            description="Interactively interpret and validate IOI circuit",
            require_falsification=True
        )
        
        report = loop.run_campaign(goal)
        self.assertGreaterEqual(report.total_iterations, 1)
        self.assertIsNotNone(report.final_composite_confidence)
        self.assertIsNotNone(report.reasoning_trace)
        self.assertEqual(report.reasoning_trace["observation_id"], goal.goal_id)

    def test_information_gain_scheduler(self):
        from backend.interpretability.discovery.information_gain_scheduler import InformationGainScheduler
        from backend.interpretability.discovery.discovery_planner import ResearchGoal
        adapter = GPT2Adapter(variant="small", mock_mode=True)
        scheduler = InformationGainScheduler(adapter=adapter, min_eu_threshold=0.02, max_steps=4)
        
        goal = ResearchGoal(
            goal_id="g_eu_1",
            description="Decision-theoretic active learning campaign for IOI circuit",
            model_id="gpt2"
        )
        
        # Test candidate evaluation & Expected Utility
        candidates = scheduler.evaluate_candidates(current_uncertainty=0.50, executed_algorithms={"acdc": 2}, goal=goal)
        self.assertGreater(len(candidates), 3)
        self.assertTrue(hasattr(candidates[0], "expected_utility"))
        self.assertTrue(hasattr(candidates[0], "scientific_novelty"))

        # Verify repeat ACDC has lower EU due to diminishing returns & compute cost
        acdc_cand = next(c for c in candidates if c.algorithm_name == "acdc")
        self.assertIn("PRUNED", acdc_cand.rationale)

        # Run scheduled campaign
        campaign_report = scheduler.run_scheduled_campaign(goal)
        self.assertGreaterEqual(len(campaign_report.steps_executed), 1)
        self.assertLess(campaign_report.final_uncertainty, campaign_report.initial_uncertainty)
        self.assertGreater(campaign_report.total_information_gained, 0.0)

    def test_discovery_memory_engine(self):
        from backend.interpretability.discovery.discovery_memory import DiscoveryMemoryEngine
        memory = DiscoveryMemoryEngine()
        
        results = memory.search("circuits involving induction")
        self.assertGreater(results.total_results, 0)
        self.assertGreater(len(results.mechanism_claims) + len(results.neurons) + len(results.circuits), 0)

        results_ioi = memory.search("IOI Name Mover")
        self.assertGreater(results_ioi.total_results, 0)

    def test_dag_discovery_planner(self):
        from backend.interpretability.discovery.dag_discovery_planner import DynamicDAGPlanner
        from backend.interpretability.discovery.discovery_planner import ResearchGoal
        adapter = GPT2Adapter(variant="small", mock_mode=True)
        planner = DynamicDAGPlanner(adapter=adapter)
        
        goal = ResearchGoal(
            goal_id="g_dag_1",
            description="Dynamic DAG discovery campaign for IOI circuit with transcoders and universality",
            require_falsification=True,
            require_universality=True
        )
        
        dag = planner.build_dag(goal)
        self.assertIn("attr", dag.nodes)
        self.assertIn("acdc", dag.nodes)
        self.assertIn("scrub", dag.nodes)
        self.assertIn("fusion", dag.nodes)

        levels = dag.get_topological_levels()
        self.assertGreaterEqual(len(levels), 3)

        execution_report = planner.execute_dag(goal)
        self.assertGreaterEqual(execution_report["total_levels"], 3)
        self.assertGreater(execution_report["nodes_count"], 3)

    def test_autonomous_paper_replicator(self):
        from backend.interpretability.discovery.autonomous_paper_replicator import AutonomousPaperReplicator
        adapter = GPT2Adapter(variant="small", mock_mode=True)
        replicator = AutonomousPaperReplicator(adapter=adapter)
        
        # Test parse paper spec
        spec = replicator.parse_paper_spec("Interpretability in the Wild: IOI Circuit (Wang et al. 2022)")
        self.assertEqual(spec.paper_id, "paper_wang_2022")
        self.assertIn("L9H9", spec.claimed_circuit_components)

        # Run paper replication
        report = replicator.replicate_paper("Conmy et al. 2023 ACDC")
        self.assertEqual(report.status, "Validated")
        self.assertGreaterEqual(report.fidelity_score, 90.0)
        self.assertGreaterEqual(len(report.reproduced_figures), 1)

    def test_scientific_publication_engine(self):
        from backend.interpretability.discovery.scientific_publication_engine import ScientificPublicationEngine
        pub_engine = ScientificPublicationEngine()
        
        # Generate replication paper
        paper_repl = pub_engine.generate_replication_paper(paper_title="Interpretability in the Wild", fidelity_score=97.2)
        self.assertEqual(paper_repl.paper_type, "REPLICATION")
        self.assertIn("97.2%", paper_repl.markdown_content)
        self.assertIn("\\documentclass", paper_repl.latex_content)

        # Generate discovery paper
        paper_disc = pub_engine.generate_discovery_paper(mechanism_name="Sparse Transcoder Induction Mechanism", model_name="Gemma-2B")
        self.assertEqual(paper_disc.paper_type, "DISCOVERY")
        self.assertIn("Gemma-2B", paper_disc.markdown_content)

    def test_circuit_discovery_engine(self):
        adapter = GPT2Adapter(variant="small", mock_mode=True)
        engine = CircuitDiscoveryEngine(adapter)
        report = engine.discover_circuit(dataset_name="ioi", prompt_id="ioi_0001", algorithm_name="acdc")
        self.assertEqual(report["algorithm"], "acdc")
        self.assertIsNotNone(report["graph"])

    def test_real_acdc_algorithm(self):
        from backend.interpretability.discovery.algorithms.acdc import ACDCAlgorithm
        adapter = GPT2Adapter(variant="small", mock_mode=True)
        acdc_alg = ACDCAlgorithm(adapter=adapter)

        dataset_item = {
            "clean": "John gave a drink to Mary",
            "corrupted": "John gave a drink to John",
            "target_token": " Mary"
        }

        report = acdc_alg.run(dataset_item)
        self.assertEqual(report.algorithm, "acdc")
        self.assertGreater(report.statistics["total_candidate_components"], 0)
        self.assertGreater(report.statistics["retained_components"], 0)
        self.assertGreaterEqual(report.statistics["logit_recovery_fidelity"], 0.90)
        self.assertIn("greedy_reverse_topological_edge_pruning", report.provenance["metric"])

    def test_research_campaign_manager(self):
        from backend.interpretability.discovery.research_campaign_manager import ResearchCampaignManager
        from backend.interpretability.discovery.discovery_planner import ResearchGoal
        
        manager = ResearchCampaignManager()
        campaigns = manager.list_all()
        self.assertGreater(len(campaigns), 0)

        # Create new campaign
        goal = ResearchGoal(goal_id="g_camp_01", description="Test campaign for induction heads", model_id="gemma")
        new_camp = manager.create_campaign("Test Campaign", goal, initial_budget_usd=15.0)
        self.assertEqual(new_camp.status, "Running")
        self.assertEqual(new_camp.budget_remaining_usd, 15.0)

        # Record experiment run
        updated = manager.record_experiment_run(
            campaign_id=new_camp.campaign_id,
            algorithm_name="attribution_patching",
            target_model="gemma",
            success=True,
            compute_flops=1.5e12,
            runtime_ms=1500.0,
            evidence={"isolated_heads": ["L5H1"]},
            uncertainty_after=0.30
        )
        self.assertEqual(len(updated.completed_experiments), 1)
        self.assertEqual(updated.remaining_uncertainty, 0.30)

    def test_bayesian_belief_engine(self):
        from backend.interpretability.reasoning.bayesian_belief_engine import BayesianBeliefEngine, MechanismBelief
        engine = BayesianBeliefEngine()

        belief = MechanismBelief(claim_id="ioi_test", title="IOI Name Mover Test", prior=0.50, posterior=0.50)
        self.assertEqual(belief.update_count, 0)

        # Apply supporting update
        updated_1 = engine.update_belief(belief, algorithm_name="attribution_patching", evidence_id="ev_01", is_supporting=True)
        self.assertEqual(updated_1.update_count, 1)
        self.assertGreater(updated_1.posterior, 0.50)
        self.assertLess(updated_1.uncertainty, 0.50)

        # Apply second supporting update (ACDC)
        updated_2 = engine.update_belief(updated_1, algorithm_name="acdc", evidence_id="ev_02", is_supporting=True)
        self.assertGreater(updated_2.posterior, updated_1.posterior)

        # Apply contradictory update
        updated_3 = engine.update_belief(updated_2, algorithm_name="causal_scrubbing", evidence_id="ev_03", is_supporting=False)
        self.assertLess(updated_3.posterior, updated_2.posterior)
        self.assertEqual(len(updated_3.history), 3)

    def test_meta_learning_engine(self):
        from backend.research.campaign_analytics import CampaignAnalyticsEngine
        from backend.research.meta_learning_engine import MetaLearningEngine
        from backend.research.planner_optimizer import PlannerOptimizer
        from backend.interpretability.discovery.discovery_planner import ResearchGoal

        analytics = CampaignAnalyticsEngine()
        report = analytics.analyze()
        self.assertGreaterEqual(report.total_campaigns_analyzed, 1)

        meta_engine = MetaLearningEngine(analytics_engine=analytics)
        policy = meta_engine.learn_policy(task_category="ioi")
        self.assertEqual(policy.task_category, "ioi")
        self.assertIn("attribution_patching", policy.recommended_sequence)
        self.assertGreater(len(policy.algorithm_priority_weights), 3)

        optimizer = PlannerOptimizer(meta_engine=meta_engine)
        goal = ResearchGoal(goal_id="g_meta_01", description="IOI Meta-Optimized Campaign", dataset_name="ioi")
        optimized_dag = optimizer.build_meta_optimized_dag(goal)
        self.assertIn("attr", optimized_dag.nodes)

    def test_distributed_workflow_engine(self):
        from backend.distributed import (
            ResourceManager,
            DistributedScheduler,
            EvidenceAggregator,
            DistributedCampaignOrchestrator
        )
        from backend.interpretability.discovery.discovery_planner import ResearchGoal

        # Test Resource Manager
        rm = ResourceManager()
        status = rm.get_cluster_status()
        self.assertGreaterEqual(len(status), 3)

        optimal_worker = rm.select_optimal_worker(model_id="gpt2", estimated_vram_gb=4.0)
        self.assertIsNotNone(optimal_worker.worker_id)

        # Test Distributed Campaign Orchestration
        orchestrator = DistributedCampaignOrchestrator(resource_manager=rm)
        goal = ResearchGoal(goal_id="g_dist_01", description="Distributed IOI Circuit Campaign", model_id="gpt2")
        summary = orchestrator.run_distributed_campaign(goal)

        self.assertEqual(summary.successful_tasks, 5)
        self.assertGreater(len(summary.cluster_workers_used), 0)
        self.assertGreaterEqual(summary.aggregated_evidence["fused_confidence"], 0.85)

    def test_scientific_knowledge_graph(self):
        from backend.knowledge_graph import (
            GraphStore,
            NodeType,
            EdgeType,
            GraphQueryEngine,
            ProvenanceGraph
        )

        store = GraphStore()
        self.assertGreater(len(store.nodes), 4)

        query_engine = GraphQueryEngine(store=store)

        res_exp = query_engine.query_experiments_for_claim("IOI Name Mover")
        self.assertGreaterEqual(res_exp.match_count, 1)

        res_paper = query_engine.query_papers_for_head("L9H9")
        self.assertGreaterEqual(res_paper.match_count, 1)

        prov_engine = ProvenanceGraph(store=store)
        chains = prov_engine.Trace_provenance_chain("paper_wang2022")
        self.assertGreaterEqual(len(chains), 1)

    def test_continuous_validation_engine(self):
        from backend.validation import (
            ValidationMonitor,
            ValidationBenchmarkScheduler,
            RegressionDetector,
            AlertEngine,
            HealthDashboardEngine,
        )

        # Environment detection
        monitor = ValidationMonitor()
        triggers = monitor.detect_triggers(previous_env=None)
        self.assertGreater(len(triggers), 0)

        # Benchmark scheduling
        scheduler = ValidationBenchmarkScheduler()
        results = scheduler.execute_validation_suite()
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r.status == "PASS" for r in results))
        self.assertTrue(all(r.current_fidelity > 0 for r in results))

        # Regression detection (all PASS → no regressions)
        detector = RegressionDetector()
        regressions = detector.analyze_results(results)
        self.assertEqual(len(regressions), 0)

        # Alert engine on empty regression list → INFO alert
        alert_engine = AlertEngine()
        alerts = alert_engine.generate_alerts(regressions)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0].severity, "INFO")

        # Full health dashboard run
        engine = HealthDashboardEngine()
        report = engine.run_continuous_validation()
        self.assertGreaterEqual(report.platform_health_score, 95.0)
        self.assertEqual(report.passed_benchmarks, 5)
        self.assertEqual(len(report.regression_events), 0)

    def test_plugin_sdk(self):
        from backend.plugins import (
            MechPlugin, PluginManifest,
            PluginLoader, PluginLoadError,
            PluginRegistry, PluginAlreadyRegisteredError, PluginNotFoundError,
            PluginHookBus, ALL_HOOKS,
            HOOK_EXPERIMENT_PLANNED, HOOK_CAMPAIGN_COMPLETED, HOOK_PAPER_GENERATED,
        )

        # --- 1. Concrete plugin implementation inline ---
        class _TestPlugin(MechPlugin):
            _MF = PluginManifest(
                plugin_id="test.unit.sdk",
                name="SDK Unit Test Plugin",
                version="1.0.0",
                author="test",
                description="Testing.",
                hooks=[HOOK_EXPERIMENT_PLANNED, HOOK_CAMPAIGN_COMPLETED, HOOK_PAPER_GENERATED],
            )
            def __init__(self):
                self.loaded = False
                self.unloaded = False
                self.campaigns_seen = []
                self.plans_seen = []

            @property
            def manifest(self):
                return self._MF

            def on_load(self):
                self.loaded = True

            def on_unload(self):
                self.unloaded = True

            def on_experiment_planned(self, plan):
                self.plans_seen.append(plan)
                return None   # pass-through

            def on_campaign_completed(self, campaign):
                self.campaigns_seen.append(campaign)

            def on_paper_generated(self, paper):
                return paper.get("abstract", "") + " [PLUGIN_ENRICHED]"

        # --- 2. Registry lifecycle ---
        registry = PluginRegistry()
        plugin = _TestPlugin()

        registry.register(plugin)
        self.assertTrue(plugin.loaded, "on_load() must be called on register")
        self.assertIn("test.unit.sdk", registry)
        self.assertEqual(len(registry), 1)

        # Duplicate registration raises
        with self.assertRaises(PluginAlreadyRegisteredError):
            registry.register(_TestPlugin())

        # Allow override
        registry.register(_TestPlugin(), allow_override=True)

        # Lookup
        fetched = registry.get("test.unit.sdk")
        self.assertIsInstance(fetched, MechPlugin)

        # Not found raises
        with self.assertRaises(PluginNotFoundError):
            registry.get("nonexistent.plugin")

        # Manifests list
        manifests = registry.list_manifests()
        self.assertEqual(len(manifests), 1)
        self.assertEqual(manifests[0].plugin_id, "test.unit.sdk")

        # --- 3. Hook bus dispatch ---
        bus = PluginHookBus(registry=registry)

        # Notification hook
        result = bus.on_campaign_completed({"campaign_id": "c_001", "goal": "IOI"})
        self.assertEqual(len(result.errors), 0)

        # Mutating hook — experiment planned (pass-through → returns None)
        plan_payload = {"name": "IOI: L9H9 patching"}
        result = bus.on_experiment_planned(plan_payload)
        self.assertEqual(result.final_payload, plan_payload)

        # Mutating hook — paper generated → abstract enriched
        paper = {"title": "IOI Circuits", "abstract": "We study IOI."}
        result = bus.on_paper_generated(paper)
        self.assertIn("PLUGIN_ENRICHED", result.final_payload)

        # All hooks constant must contain 8 entries
        self.assertEqual(len(ALL_HOOKS), 8)

        # --- 4. Loader: bad path raises ---
        loader = PluginLoader()
        with self.assertRaises(PluginLoadError):
            loader.load_from_path("totally.nonexistent.module.xyz")

        # --- 5. Unregister ---
        registry.unregister("test.unit.sdk")
        self.assertNotIn("test.unit.sdk", registry)
        self.assertEqual(len(registry), 0)

    def test_dataset_manager(self):
        from backend.datasets.dataset_manager import DatasetManager
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            dm = DatasetManager(tmpdir)
            datasets = dm.list_datasets()
            self.assertIn("ioi", datasets)
            self.assertIn("induction", datasets)
            self.assertIn("greater_than", datasets)
            
            prompts = dm.load("ioi", num_prompts=10)
            self.assertEqual(len(prompts), 10)
            
            stats = dm.stats("ioi")
            self.assertEqual(stats["num_prompts"], 10)

if __name__ == "__main__":
    unittest.main(verbosity=2)

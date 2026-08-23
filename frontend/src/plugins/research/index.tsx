import React from 'react';
import { pluginRegistry } from '../../panel-system/pluginRegistry';
import type { FC, PanelContext } from '../../shared/types';
import { ActiveInvestigationOverview } from '../../components/research/ActiveInvestigationOverview';
import { HypothesisLab } from '../../components/research/HypothesisLab';
import { InterventionLab } from '../../components/research/InterventionLab';
import { ExperimentBuilder } from '../../components/research/ExperimentBuilder';
import { ExperimentMonitor } from '../../components/research/ExperimentMonitor';
import { EvidenceExplorer } from '../../components/research/EvidenceExplorer';
import { ResearchGraph } from '../../components/research/ResearchGraph';
import { ResearchNotebook } from '../../components/research/ResearchNotebook';
import { AIResearchAssistant } from '../../components/research/AIResearchAssistant';
import { ResearchSearch } from '../../components/research/ResearchSearch';
import { EvidenceGraphView } from '../../components/research/EvidenceGraphView';
import { MechanismBuilder } from '../../components/research/MechanismBuilder';
import { ModelExplorerView } from '../../components/research/ModelExplorerView';
import { ComputeCenterView } from '../../components/research/ComputeCenterView';
import { ArtifactsReportView } from '../../components/research/ArtifactsReportView';
import { MechanismGraphCenterpiece } from '../../components/research/MechanismGraphCenterpiece';
import { ResearchQueueView } from '../../components/research/ResearchQueueView';
import { ComparisonWorkspaceView } from '../../components/research/ComparisonWorkspaceView';
import { GroundTruthBenchmarkView } from '../../components/research/GroundTruthBenchmarkView';
import { MechanismDiffView } from '../../components/research/MechanismDiffView';
import { MechanismCriticPanel } from '../../components/research/MechanismCriticPanel';

// Reasoning Agent Components
import { HypothesisGenerator } from '../../components/reasoning/HypothesisGenerator';
import { ExperimentPlanner } from '../../components/reasoning/ExperimentPlanner';
import { FalsificationDesigner } from '../../components/reasoning/FalsificationDesigner';
import { EvidenceReasoner } from '../../components/reasoning/EvidenceReasoner';
import { CandidatePrioritizer } from '../../components/reasoning/CandidatePrioritizer';
import { ResearchLoopController } from '../../components/reasoning/ResearchLoopController';

// 1. Active Investigation Overview
const ActiveInvestigationBody: FC<PanelContext> = () => <ActiveInvestigationOverview />;
pluginRegistry.register({
  id: 'active_investigation',
  title: 'Active Investigation',
  icon: 'Brain',
  category: 'workspace',
  resourceKinds: ['workspace', 'session', 'model'],
  defaultDock: 'center',
  fullWidth: true,
  Body: ActiveInvestigationBody,
});

// 2. Inspectable Mechanism Graph Centerpiece
const MechanismGraphBody: FC<PanelContext> = () => <MechanismGraphCenterpiece />;
pluginRegistry.register({
  id: 'mechanism_graph',
  title: 'Mechanism Circuit Graph',
  icon: 'Layers',
  category: 'research',
  resourceKinds: ['circuit', 'model'],
  defaultDock: 'center',
  fullWidth: true,
  Body: MechanismGraphBody,
});

// 3. What Should I Test Next? (Research Queue)
const ResearchQueueBody: FC<PanelContext> = () => <ResearchQueueView />;
pluginRegistry.register({
  id: 'research_queue',
  title: 'Research Queue (Next Experiments)',
  icon: 'Sparkles',
  category: 'research',
  resourceKinds: ['experiment', 'session'],
  defaultDock: 'center',
  fullWidth: true,
  Body: ResearchQueueBody,
});

// 4. Comparison Workspace (Side-by-Side)
const ComparisonWorkspaceBody: FC<PanelContext> = () => <ComparisonWorkspaceView />;
pluginRegistry.register({
  id: 'comparison_workspace',
  title: 'Comparison Workspace',
  icon: 'Scale',
  category: 'research',
  resourceKinds: ['experiment', 'session', 'circuit'],
  defaultDock: 'center',
  fullWidth: true,
  Body: ComparisonWorkspaceBody,
});

// 5. Known Literature Mechanisms (Ground Truth Benchmarks)
const GroundTruthBenchmarksBody: FC<PanelContext> = () => <GroundTruthBenchmarkView />;
pluginRegistry.register({
  id: 'ground_truth_benchmarks',
  title: 'Known Mechanisms & Benchmarks',
  icon: 'Award',
  category: 'research',
  resourceKinds: ['session', 'paper'],
  defaultDock: 'center',
  fullWidth: true,
  Body: GroundTruthBenchmarksBody,
});

// 6. Mechanism Diff & Lineage
const MechanismDiffBody: FC<PanelContext> = () => <MechanismDiffView />;
pluginRegistry.register({
  id: 'mechanism_diff',
  title: 'Mechanism Version Diff',
  icon: 'GitCompare',
  category: 'research',
  resourceKinds: ['circuit'],
  defaultDock: 'center',
  fullWidth: true,
  Body: MechanismDiffBody,
});

// 7. Mechanism Critic & Epistemic Audit
const MechanismCriticBody: FC<PanelContext> = () => <MechanismCriticPanel />;
pluginRegistry.register({
  id: 'mechanism_critic',
  title: 'Mechanism Critic & Audit',
  icon: 'ShieldCheck',
  category: 'research',
  resourceKinds: ['session'],
  defaultDock: 'center',
  fullWidth: true,
  Body: MechanismCriticBody,
});

// 8. Hypothesis Lab
const HypothesisLabBody: FC<PanelContext> = () => <HypothesisLab />;
pluginRegistry.register({
  id: 'hypothesis_lab',
  title: 'Hypothesis Lab & Falsification',
  icon: 'FlaskConical',
  category: 'research',
  resourceKinds: ['experiment', 'session'],
  defaultDock: 'center',
  fullWidth: true,
  Body: HypothesisLabBody,
});

// 9. Causal Intervention Lab
const InterventionLabBody: FC<PanelContext> = () => <InterventionLab />;
pluginRegistry.register({
  id: 'intervention_lab',
  title: 'Causal Intervention & Patching Lab',
  icon: 'Zap',
  category: 'research',
  resourceKinds: ['model', 'experiment', 'head'],
  defaultDock: 'center',
  fullWidth: true,
  Body: InterventionLabBody,
});

// 10. Evidence Graph
const EvidenceGraphBody: FC<PanelContext> = () => <EvidenceGraphView />;
pluginRegistry.register({
  id: 'evidence_graph',
  title: 'Evidence Graph & Provenance',
  icon: 'Network',
  category: 'research',
  resourceKinds: ['circuit', 'session'],
  defaultDock: 'center',
  fullWidth: true,
  Body: EvidenceGraphBody,
});

// 11. Mechanism Builder
const MechanismBuilderBody: FC<PanelContext> = () => <MechanismBuilder />;
pluginRegistry.register({
  id: 'mechanism_builder',
  title: 'Visual Mechanism Builder',
  icon: 'Layers',
  category: 'research',
  resourceKinds: ['circuit', 'model'],
  defaultDock: 'center',
  fullWidth: true,
  Body: MechanismBuilderBody,
});

// 12. Model Architecture Explorer
const ModelExplorerBody: FC<PanelContext> = () => <ModelExplorerView />;
pluginRegistry.register({
  id: 'model_explorer',
  title: 'Model Architecture Explorer',
  icon: 'Search',
  category: 'model',
  resourceKinds: ['model'],
  defaultDock: 'center',
  fullWidth: true,
  Body: ModelExplorerBody,
});

// 13. Compute & Job Center
const ComputeCenterBody: FC<PanelContext> = () => <ComputeCenterView />;
pluginRegistry.register({
  id: 'compute_center',
  title: 'Compute & Job Center',
  icon: 'Cpu',
  category: 'system',
  resourceKinds: ['workspace'],
  defaultDock: 'center',
  fullWidth: true,
  Body: ComputeCenterBody,
});

// 14. Scientific Report Mode
const ReportModeBody: FC<PanelContext> = () => <ArtifactsReportView />;
pluginRegistry.register({
  id: 'report_mode',
  title: 'Publication & Report Generator',
  icon: 'FileText',
  category: 'workspace',
  resourceKinds: ['paper', 'workspace'],
  defaultDock: 'center',
  fullWidth: true,
  Body: ReportModeBody,
});

// 15. Experiment Builder
const ExperimentBuilderBody: FC<PanelContext> = () => <ExperimentBuilder />;
pluginRegistry.register({
  id: 'experiment_builder',
  title: 'Experiment Builder',
  icon: 'FlaskConical',
  category: 'research',
  resourceKinds: ['experiment', 'session'],
  defaultDock: 'center',
  fullWidth: true,
  Body: ExperimentBuilderBody,
});

// 16. Experiment Monitor
const ExperimentMonitorBody: FC<PanelContext> = () => <ExperimentMonitor />;
pluginRegistry.register({
  id: 'experiment_monitor',
  title: 'Experiment Monitor',
  icon: 'Activity',
  category: 'research',
  resourceKinds: ['experiment', 'session'],
  defaultDock: 'center',
  fullWidth: true,
  Body: ExperimentMonitorBody,
});

// 17. Evidence Explorer
const EvidenceExplorerBody: FC<PanelContext> = () => <EvidenceExplorer />;
pluginRegistry.register({
  id: 'evidence_explorer',
  title: 'Evidence Explorer & Hypothesis Mapping',
  icon: 'Search',
  category: 'research',
  resourceKinds: ['evidence', 'experiment', 'session'],
  defaultDock: 'center',
  fullWidth: true,
  Body: EvidenceExplorerBody,
});

// 18. Research Graph
const ResearchGraphBody: FC<PanelContext> = () => <ResearchGraph />;
pluginRegistry.register({
  id: 'research_graph',
  title: 'Research Graph & Pipeline',
  icon: 'GitBranch',
  category: 'research',
  resourceKinds: ['research', 'session'],
  defaultDock: 'center',
  fullWidth: true,
  Body: ResearchGraphBody,
});

// 19. Research Notebook
const ResearchNotebookBody: FC<PanelContext> = () => <ResearchNotebook />;
pluginRegistry.register({
  id: 'research_notebook',
  title: 'Research Notebook',
  icon: 'NotebookPen',
  category: 'research',
  resourceKinds: ['research', 'session'],
  defaultDock: 'center',
  fullWidth: true,
  Body: ResearchNotebookBody,
});

// 20. AI Research Assistant
const AIResearchAssistantBody: FC<PanelContext> = () => <AIResearchAssistant />;
pluginRegistry.register({
  id: 'ai_research_assistant',
  title: 'AI Research Assistant',
  icon: 'Sparkles',
  category: 'research',
  resourceKinds: ['research', 'session'],
  defaultDock: 'center',
  fullWidth: true,
  Body: AIResearchAssistantBody,
});

// 21. Research Search
const ResearchSearchBody: FC<PanelContext> = () => <ResearchSearch isOpen={true} onClose={() => {}} onSelect={() => {}} />;
pluginRegistry.register({
  id: 'research_search',
  title: 'Research Search',
  icon: 'Search',
  category: 'research',
  resourceKinds: ['research', 'session'],
  defaultDock: 'center',
  fullWidth: true,
  Body: ResearchSearchBody,
});

// 22. Hypothesis Generator (Reasoning Agent)
const HypothesisGeneratorBody: FC<PanelContext> = () => <HypothesisGenerator />;
pluginRegistry.register({
  id: 'hypothesis_generator',
  title: 'Hypothesis Generator',
  icon: 'Brain',
  category: 'research',
  resourceKinds: ['research', 'session'],
  defaultDock: 'center',
  fullWidth: true,
  Body: HypothesisGeneratorBody,
});

// 23. Experiment Planner (Reasoning Agent)
const ExperimentPlannerBody: FC<PanelContext> = () => <ExperimentPlanner />;
pluginRegistry.register({
  id: 'experiment_planner',
  title: 'Experiment Planner',
  icon: 'ClipboardList',
  category: 'research',
  resourceKinds: ['research', 'session'],
  defaultDock: 'center',
  fullWidth: true,
  Body: ExperimentPlannerBody,
});

// 24. Falsification Designer (Reasoning Agent)
const FalsificationDesignerBody: FC<PanelContext> = () => <FalsificationDesigner />;
pluginRegistry.register({
  id: 'falsification_designer',
  title: 'Falsification Designer',
  icon: 'ShieldAlert',
  category: 'research',
  resourceKinds: ['research', 'session'],
  defaultDock: 'center',
  fullWidth: true,
  Body: FalsificationDesignerBody,
});

// 25. Evidence Reasoner (Reasoning Agent)
const EvidenceReasonerBody: FC<PanelContext> = () => <EvidenceReasoner />;
pluginRegistry.register({
  id: 'evidence_reasoner',
  title: 'Evidence Reasoner',
  icon: 'Network',
  category: 'research',
  resourceKinds: ['research', 'session'],
  defaultDock: 'center',
  fullWidth: true,
  Body: EvidenceReasonerBody,
});

// 26. Candidate Prioritizer (Reasoning Agent)
const CandidatePrioritizerBody: FC<PanelContext> = () => <CandidatePrioritizer />;
pluginRegistry.register({
  id: 'candidate_prioritizer',
  title: 'Candidate Prioritizer',
  icon: 'BarChart3',
  category: 'research',
  resourceKinds: ['research', 'session'],
  defaultDock: 'center',
  fullWidth: true,
  Body: CandidatePrioritizerBody,
});

// 27. Research Loop Controller (Reasoning Agent)
const ResearchLoopControllerBody: FC<PanelContext> = () => <ResearchLoopController />;
pluginRegistry.register({
  id: 'research_loop_controller',
  title: 'Research Loop Controller',
  icon: 'Repeat',
  category: 'research',
  resourceKinds: ['research', 'session'],
  defaultDock: 'center',
  fullWidth: true,
  Body: ResearchLoopControllerBody,
});

// 28. UI Adversarial Tester
import { UIAdversarialTester } from '../../components/research/UIAdversarialTester';
const UIAdversarialTesterBody: FC<PanelContext> = () => <UIAdversarialTester />;
pluginRegistry.register({
  id: 'ui_adversarial_tester',
  title: 'UI Adversarial Tester',
  icon: 'Shield',
  category: 'research',
  resourceKinds: ['research', 'session'],
  defaultDock: 'center',
  fullWidth: true,
  Body: UIAdversarialTesterBody,
});

// 29. Acceptance Test Verifier
import { AcceptanceTestVerifier } from '../../components/research/AcceptanceTestVerifier';
const AcceptanceTestVerifierBody: FC<PanelContext> = () => <AcceptanceTestVerifier />;
pluginRegistry.register({
  id: 'acceptance_test_verifier',
  title: 'Acceptance Test Verifier',
  icon: 'CheckCircle',
  category: 'research',
  resourceKinds: ['research', 'session'],
  defaultDock: 'center',
  fullWidth: true,
  Body: AcceptanceTestVerifierBody,
});

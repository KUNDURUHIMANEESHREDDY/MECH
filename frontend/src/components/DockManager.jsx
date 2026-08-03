import React, { useState, useEffect } from 'react';
import { BarChart3, Brain, ClipboardPen, Clapperboard, Dna, FileText, Flame, FlaskConical, Glasses, Globe, Hourglass, Map, Mic, Network, Presentation, Rewind, Scale, Search, Sparkles, Timer, TreePine, Users, Waves, X, Zap } from 'lucide-react';
import { panelRegistry } from '../utils/panelRegistry';
import { eventBus } from '../utils/eventBus';
import InferenceTimeline from './panels/InferenceTimeline';
import SAEFeaturePanel from './panels/SAEFeaturePanel';
import CircuitGraphPanel from './panels/CircuitGraphPanel';
import ResidualStreamViewer from './panels/ResidualStreamViewer';
import LogitLensViewer from './panels/LogitLensViewer';
import LayerComparisonPanel from './panels/LayerComparisonPanel';
import ModelComparisonPanel from './panels/ModelComparisonPanel';
import CircuitExplorerPanel from './panels/CircuitExplorerPanel';
import CausalTraceViewerPanel from './panels/CausalTraceViewerPanel';
import FeatureAtlasPanel from './panels/FeatureAtlasPanel';
import InterventionTimelinePanel from './panels/InterventionTimelinePanel';
import TokenJourneyPanel from './panels/TokenJourneyPanel';
import EmbeddingViewerPanel from './panels/EmbeddingViewerPanel';
import VersionControlPanel from './panels/VersionControlPanel';
import ArtifactManagerPanel from './panels/ArtifactManagerPanel';
import PublicationBuilderPanel from './panels/PublicationBuilderPanel';
import ResearchGraphPanel from './panels/ResearchGraphPanel';
import CircuitAnimationPanel from './panels/CircuitAnimationPanel';
import KnowledgeGraphViewerPanel from './panels/KnowledgeGraphViewerPanel';
import VR3DNetworkViewerPanel from './panels/VR3DNetworkViewerPanel';
import PublicationDashboardPanel from './panels/PublicationDashboardPanel';
import PresentationModePanel from './panels/PresentationModePanel';
import LiveCollaborationPanel from './panels/LiveCollaborationPanel';
import KnowledgeUniversePanel from './panels/KnowledgeUniversePanel';
import MechanismSimulatorPanel from './panels/MechanismSimulatorPanel';
import ResearchReplayPanel from './panels/ResearchReplayPanel';
import DiscoveryTimelinePanel from './panels/DiscoveryTimelinePanel';
import ConferenceModePanel from './panels/ConferenceModePanel';
import ImmersiveCollabPanel from './panels/ImmersiveCollabPanel';
import PublicationStudioPanel from './panels/PublicationStudioPanel';
import CausalGraphEditorPanel from './panels/CausalGraphEditorPanel';
import TimeTravelDebuggerPanel from './panels/TimeTravelDebuggerPanel';
import LayerEvolutionPanel from './panels/LayerEvolutionPanel';
import FeatureGenealogyExplorerPanel from './panels/FeatureGenealogyExplorerPanel';
import ConfidenceHeatmapPanel from './panels/ConfidenceHeatmapPanel';
import DiscoveryComparisonPanel from './panels/DiscoveryComparisonPanel';
import InteractiveFiguresPanel from './panels/InteractiveFiguresPanel';

export default function DockManager({ isVisible, onClose }) {
  const [panels, setPanels] = useState([]);
  const [activePanelId, setActivePanelId] = useState('inference-timeline');
  const [expanded, setExpanded] = useState(true);

  useEffect(() => {
    panelRegistry.register({ id: 'inference-timeline', name: 'Inference Stepper', icon: Timer, desc: 'Step through forward inference layers', order: 10, component: InferenceTimeline });
    panelRegistry.register({ id: 'sae-feature', name: 'SAE Features', icon: Dna, desc: 'Sparse Autoencoder feature inspector', order: 20, component: SAEFeaturePanel });
    panelRegistry.register({ id: 'circuit-graph', name: 'Circuit Graph', icon: Network, desc: 'Interactive circuit graph mapper', order: 30, component: CircuitGraphPanel });
    panelRegistry.register({ id: 'causal-graph-editor', name: 'Causal Graph Editor', icon: Zap, desc: 'Interactive node/edge pruning editor', order: 35, component: CausalGraphEditorPanel });
    panelRegistry.register({ id: 'residual-stream', name: 'Residual Stream', icon: BarChart3, desc: 'Residual stream vector evolution', order: 40, component: ResidualStreamViewer });
    panelRegistry.register({ id: 'layer-evolution', name: 'Layer Evolution', icon: Clapperboard, desc: 'Residual stream trajectory animation', order: 45, component: LayerEvolutionPanel });
    panelRegistry.register({ id: 'logit-lens', name: 'Projection Lens', icon: Search, desc: 'Logit Lens and Tuned Lens viewer', order: 50, component: LogitLensViewer });
    panelRegistry.register({ id: 'time-travel-debugger', name: 'Time-Travel Debugger', icon: Rewind, desc: 'Step backward/forward through activations', order: 55, component: TimeTravelDebuggerPanel });
    panelRegistry.register({ id: 'circuit-explorer', name: 'Circuit Explorer', icon: Zap, desc: 'Interactive causal circuit graph explorer', order: 80, component: CircuitExplorerPanel });
    panelRegistry.register({ id: 'causal-trace-viewer', name: 'Causal Trace', icon: Waves, desc: 'Animated causal flow viewer', order: 90, component: CausalTraceViewerPanel });
    panelRegistry.register({ id: 'feature-atlas', name: 'Feature Atlas', icon: Map, desc: 'Searchable 16k SAE feature catalog', order: 100, component: FeatureAtlasPanel });
    panelRegistry.register({ id: 'feature-genealogy-explorer', name: 'Genealogy Explorer', icon: TreePine, desc: 'Feature lineage tree explorer', order: 105, component: FeatureGenealogyExplorerPanel });
    panelRegistry.register({ id: 'confidence-heatmap', name: 'Confidence Heatmap', icon: Flame, desc: 'Statistical confidence heatmaps', order: 108, component: ConfidenceHeatmapPanel });
    panelRegistry.register({ id: 'research-graph', name: 'Research Graph', icon: Globe, desc: 'Interactive project DAG explorer', order: 110, component: ResearchGraphPanel });
    panelRegistry.register({ id: 'circuit-animation', name: 'Circuit Animation', icon: Clapperboard, desc: 'Activation propagation scrubber', order: 120, component: CircuitAnimationPanel });
    panelRegistry.register({ id: 'knowledge-graph', name: 'Knowledge Graph', icon: Brain, desc: 'Connected discoveries visualizer', order: 130, component: KnowledgeGraphViewerPanel });
    panelRegistry.register({ id: 'discovery-comparison', name: 'Discovery Comparison', icon: Scale, desc: 'Side-by-side discovery comparison', order: 135, component: DiscoveryComparisonPanel });
    panelRegistry.register({ id: 'vr-3d-network', name: 'VR / 3D Mode', icon: Glasses, desc: '3D neural network manifold', order: 140, component: VR3DNetworkViewerPanel });
    panelRegistry.register({ id: 'publication-dashboard', name: 'Publication Dashboard', icon: FileText, desc: 'Unified paper & figure hub', order: 150, component: PublicationDashboardPanel });
    panelRegistry.register({ id: 'interactive-figures', name: 'Interactive Figures', icon: BarChart3, desc: 'Live figures linked to raw data', order: 155, component: InteractiveFiguresPanel });
    panelRegistry.register({ id: 'presentation-mode', name: 'Presentation Mode', icon: Presentation, desc: 'Interactive slide deck generator', order: 160, component: PresentationModePanel });
    panelRegistry.register({ id: 'live-collaboration', name: 'Live Collaboration', icon: Users, desc: 'Multi-user real-time view', order: 170, component: LiveCollaborationPanel });
    panelRegistry.register({ id: 'knowledge-universe', name: 'Knowledge Universe', icon: Sparkles, desc: 'Zoomable knowledge universe visualizer', order: 180, component: KnowledgeUniversePanel });
    panelRegistry.register({ id: 'mechanism-simulator', name: 'Mechanism Simulator', icon: FlaskConical, desc: 'Interactive intervention simulator', order: 190, component: MechanismSimulatorPanel });
    panelRegistry.register({ id: 'research-replay', name: 'Research Replay', icon: Rewind, desc: 'Campaign time-travel research replay', order: 200, component: ResearchReplayPanel });
    panelRegistry.register({ id: 'discovery-timeline', name: 'Discovery Timeline', icon: Hourglass, desc: 'Discovery evolution timeline', order: 210, component: DiscoveryTimelinePanel });
    panelRegistry.register({ id: 'conference-mode', name: 'Conference Mode', icon: Mic, desc: 'Live keynote presentation generator', order: 220, component: ConferenceModePanel });
    panelRegistry.register({ id: 'immersive-collab', name: 'Immersive Collab', icon: Globe, desc: 'Synchronized multi-user view', order: 230, component: ImmersiveCollabPanel });
    panelRegistry.register({ id: 'publication-studio', name: 'Publication Studio', icon: ClipboardPen, desc: 'Live-linked manuscript editor', order: 240, component: PublicationStudioPanel });

    setPanels(panelRegistry.getPanels());

    const unsub = eventBus.on('panels:updated', (updated) => {
      setPanels(updated);
    });

    return unsub;
  }, []);

  if (!isVisible) return null;

  const currentPanel = panels.find((p) => p.id === activePanelId) || panels[0];
  const PanelComponent = currentPanel?.component;

  return (
    <div className="dock-manager" data-testid="dock-manager">
      <div className="dock-header">
        <div className="dock-tabs" style={{ flexWrap: 'wrap' }}>
          {panels.map((p) => (
            <button
              key={p.id}
              className={`dock-tab ${activePanelId === p.id ? 'active' : ''}`}
              onClick={() => setActivePanelId(p.id)}
            >
              <span className="dock-icon">{typeof p.icon === 'string' ? p.icon : p.icon ? <p.icon size={16} /> : null}</span> {p.name}
            </button>
          ))}
        </div>
        <div className="dock-actions">
          <button
            className="btn btn-secondary btn-sm"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? 'Collapse' : 'Expand'}
          </button>
          <button className="btn btn-secondary btn-sm" onClick={onClose}><X size={14} /></button>
        </div>
      </div>

      {expanded && (
        <div className="dock-body">
          {PanelComponent ? <PanelComponent /> : (
            <div className="panel-content"><p className="hint">No panel component bound.</p></div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * Visualization DTO Adapter.
 * Transforms raw backend DTOs into clean UI rendering DTOs.
 */

export class VisualizationAdapter {
  static adaptGraphNode(backendNode) {
    return {
      id: backendNode.id,
      label: backendNode.label || backendNode.title || backendNode.id,
      type: backendNode.type || 'Node',
      color: backendNode.type === 'Circuit' ? 'var(--accent)' : backendNode.type === 'Feature' ? 'var(--success)' : 'var(--purple)',
    };
  }

  static adaptAnimationFrame(backendStep) {
    return {
      stepIndex: backendStep.step || backendStep.layer || 0,
      activeNodesCount: backendStep.active_nodes_count || 10,
      activationValue: backendStep.activation || 0.85,
    };
  }
}

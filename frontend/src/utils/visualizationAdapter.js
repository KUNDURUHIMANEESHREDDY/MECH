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
      color: backendNode.type === 'Circuit' ? '#3b82f6' : backendNode.type === 'Feature' ? '#10b981' : '#a855f7',
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

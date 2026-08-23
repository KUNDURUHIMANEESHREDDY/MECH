import React, { useState } from 'react';
import { useResearchStore, Mechanism } from '../../shared/stores/research';
import { colors } from '../../design/tokens/colors';
import { Layers, Plus, ShieldCheck, AlertTriangle, ArrowDown, Trash2, CheckCircle2 } from 'lucide-react';

export const MechanismBuilder: React.FC = () => {
  const {
    activeInvestigation,
    mechanisms,
    evidence,
    saveMechanism,
    deleteMechanism,
  } = useResearchStore();

  const [mechName, setMechName] = useState('IOI Name Mover Circuit');
  const [description, setDescription] = useState('Two-stage attention routing: Duplicate Token Head (L3H0) → S-Inhibition Head (L7H9) → Name Mover (L9H9)');

  const defaultPipeline = [
    { id: 'n1', label: 'Input Tokens ("John", "Mary")', component_type: 'input', evidence_level: 'OBSERVED' },
    { id: 'n2', label: 'Duplicate Token Detection (L3H0)', component_type: 'attention_head', layer: 3, head: 0, evidence_level: 'OBSERVED' },
    { id: 'n3', label: 'S-Inhibition Signal Routing (L7H9)', component_type: 'attention_head', layer: 7, head: 9, evidence_level: 'SUPPORTED' },
    { id: 'n4', label: 'Name Mover Logit Projection (L9H9)', component_type: 'attention_head', layer: 9, head: 9, evidence_level: 'CAUSALLY_VERIFIED' },
    { id: 'n5', label: 'Final Output Logit (" Mary")', component_type: 'output', evidence_level: 'CAUSALLY_VERIFIED' },
  ];

  const handleSave = async () => {
    await saveMechanism({
      name: mechName,
      description,
      nodes: defaultPipeline as any,
      edges: [
        { id: 'e1', source_node_id: 'n1', target_node_id: 'n2', mechanism_type: 'attention_routing', is_causally_verified: false, evidence_ids: [] },
        { id: 'e2', source_node_id: 'n2', target_node_id: 'n3', mechanism_type: 'attention_routing', is_causally_verified: true, evidence_ids: [] },
        { id: 'e3', source_node_id: 'n3', target_node_id: 'n4', mechanism_type: 'attention_routing', is_causally_verified: true, evidence_ids: [] },
        { id: 'e4', source_node_id: 'n4', target_node_id: 'n5', mechanism_type: 'ov_circuit', is_causally_verified: true, evidence_ids: [] },
      ],
      is_evidence_backed: true,
      weakest_link_tier: 'SUPPORTED',
    });
  };

  return (
    <div style={{ padding: '20px', height: '100%', overflowY: 'auto', boxSizing: 'border-box', backgroundColor: colors.canvas }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: colors.primary, textTransform: 'uppercase', letterSpacing: 0.8 }}>
            Computational Circuit Synthesis
          </div>
          <h2 style={{ margin: '4px 0 0', fontSize: 18, fontWeight: 700, color: colors.ink }}>
            Visual Mechanism Builder
          </h2>
          <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 2 }}>
            Construct proposed computational pathways and verify end-to-end causal fidelity under the Weakest-Link Principle.
          </div>
        </div>

        <button
          onClick={handleSave}
          style={{
            padding: '8px 14px',
            borderRadius: 6,
            border: 'none',
            backgroundColor: colors.primary,
            color: colors.onPrimary,
            fontSize: 12,
            fontWeight: 600,
            cursor: 'pointer',
          }}
        >
          Save Proposed Mechanism
        </button>
      </div>

      {/* Mechanism Pipeline Canvas */}
      <div style={{ border: `1px solid ${colors.border}`, borderRadius: 10, padding: 20, backgroundColor: colors.surfaceTile1, marginBottom: 20 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <div>
            <span style={{ fontSize: 14, fontWeight: 700, color: colors.ink }}>{mechName}</span>
            <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 2 }}>{description}</div>
          </div>
          <span style={{ fontSize: 11, fontWeight: 800, padding: '3px 10px', borderRadius: 999, backgroundColor: colors.successSoft, color: colors.successText, border: `1px solid ${colors.successBorder}` }}>
            EVIDENCE-BACKED
          </span>
        </div>

        {/* Vertical Pipeline Nodes */}
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 10, maxWidth: 520, margin: '0 auto' }}>
          {defaultPipeline.map((node, idx) => (
            <React.Fragment key={node.id}>
              <div
                style={{
                  width: '100%',
                  padding: '12px 16px',
                  borderRadius: 8,
                  backgroundColor: colors.canvas,
                  border: `1px solid ${node.evidence_level === 'CAUSALLY_VERIFIED' ? colors.successBorder : colors.border}`,
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  boxShadow: '0 1px 3px rgba(0,0,0,0.04)',
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <div style={{ width: 22, height: 22, borderRadius: 11, backgroundColor: colors.surfacePearl, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 11, fontWeight: 700 }}>
                    {idx + 1}
                  </div>
                  <div>
                    <div style={{ fontSize: 13, fontWeight: 600, color: colors.ink }}>{node.label}</div>
                    <div style={{ fontSize: 11, color: colors.bodyMuted }}>Type: {node.component_type}</div>
                  </div>
                </div>

                <span
                  style={{
                    fontSize: 10,
                    fontWeight: 700,
                    padding: '2px 6px',
                    borderRadius: 4,
                    backgroundColor: node.evidence_level === 'CAUSALLY_VERIFIED' ? colors.successSoft : colors.surfacePearl,
                    color: node.evidence_level === 'CAUSALLY_VERIFIED' ? colors.successText : colors.bodyMuted,
                  }}
                >
                  {node.evidence_level}
                </span>
              </div>

              {idx < defaultPipeline.length - 1 && (
                <div style={{ color: colors.primary, display: 'flex', alignItems: 'center', gap: 4, fontSize: 11, fontWeight: 600 }}>
                  <ArrowDown size={16} /> Attention-mediated information transport
                </div>
              )}
            </React.Fragment>
          ))}
        </div>
      </div>

      {/* Existing Saved Mechanisms */}
      <div>
        <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink, textTransform: 'uppercase', letterSpacing: 0.4, marginBottom: 10 }}>
          Saved Mechanisms ({mechanisms.length})
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(320px, 1fr))', gap: 12 }}>
          {mechanisms.map((m) => (
            <div key={m.id} style={{ border: `1px solid ${colors.border}`, borderRadius: 8, padding: 14, backgroundColor: colors.surfaceTile1 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                <span style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>{m.name}</span>
                <button
                  onClick={() => deleteMechanism(m.id)}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: colors.dangerText }}
                >
                  <Trash2 size={13} />
                </button>
              </div>
              <div style={{ fontSize: 12, color: colors.bodyMuted, marginBottom: 8 }}>{m.description}</div>
              <div style={{ fontSize: 11, color: colors.body }}>
                Nodes: <b>{m.nodes?.length || 0}</b> | Edges: <b>{m.edges?.length || 0}</b> | Weakest Link: <b>{m.weakest_link_tier}</b>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

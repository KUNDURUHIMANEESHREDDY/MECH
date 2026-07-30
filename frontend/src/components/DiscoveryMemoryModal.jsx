import React, { useState } from 'react';

/**
 * DiscoveryMemoryModal - Multi-Modal Semantic Search across all Research Memory
 *
 * Allows researchers to type queries like "circuits involving induction" or "French cities"
 * and instantly receive multi-modal search results across Claims, SAE Features, Neurons,
 * Circuits, Counterexamples, and Papers.
 */

const MOCK_MEMORY_DATABASE = {
  induction: {
    claims: [
      { id: 'claim_induction_heads', title: 'Induction Head Sequence Repeater', summary: 'Previous-token head L4H2 attends to token K-1 while Induction Head L5H1 copies token K.', confidence: 0.94 }
    ],
    features: [
      { id: 'SAE_Feat_402', title: 'Induction Prefix Matcher', summary: 'Fires on repeated token sequences K-1 -> K', score: 0.92 }
    ],
    neurons: [
      { id: 'Head_L5H1', title: 'Gemma Induction Head L5H1', summary: 'Copies token K from previous sequence match', score: 0.95 }
    ],
    circuits: [
      { id: 'Circ_Induction_01', title: 'Induction Circuit L4H2 -> L5H1', summary: '2-head computational subgraph for sequence repetition', score: 0.94 }
    ],
    counterexamples: [
      { id: 'CE_Induction_01', title: 'Non-repeating sequence prompt', summary: 'The cat sat on the mat. The dog ran across...', score: 0.88 }
    ],
    papers: [
      { id: 'Paper_Olsson_2022', title: 'In-context Learning and Induction Heads', summary: 'Olsson et al. 2022 - Mechanism for prefix matching and in-context learning.', score: 0.96 }
    ]
  },
  ioi: {
    claims: [
      { id: 'claim_ioi_name_mover', title: 'IOI Name Mover Circuit', summary: 'L9H9 and L10H0 act as primary Name Mover Heads writing directly to IO token logits.', confidence: 0.96 }
    ],
    features: [
      { id: 'SAE_Feat_182', title: 'French Cities / Proper Nouns Feature', summary: 'Fires on European geography and city names', score: 0.91 }
    ],
    neurons: [
      { id: 'Head_L9H9', title: 'GPT-2 Name Mover Head L9H9', summary: 'Copies indirect object name to logit output', score: 0.96 }
    ],
    circuits: [
      { id: 'Circ_IOI_01', title: 'IOI Name Mover Subgraph', summary: 'L9H9 + L10H0 Name Mover circuit', score: 0.95 }
    ],
    counterexamples: [
      { id: 'CE_IOI_01', title: 'Dallas syntax prompt', summary: 'I flew to Dallas for the weekend.', score: 0.89 }
    ],
    papers: [
      { id: 'Paper_Wang_2022', title: 'Interpretability in the Wild: IOI Circuit', summary: 'Wang et al. 2022 - Full circuit breakdown for indirect object identification in GPT-2.', score: 0.98 }
    ]
  }
};

export default function DiscoveryMemoryModal({ isOpen, onClose, onNavigate }) {
  const [query, setQuery] = useState('circuits involving induction');
  const [activeCategory, setActiveCategory] = useState('all');

  if (!isOpen) return null;

  const isIoi = query.toLowerCase().includes('ioi') || query.toLowerCase().includes('name') || query.toLowerCase().includes('french');
  const db = isIoi ? MOCK_MEMORY_DATABASE.ioi : MOCK_MEMORY_DATABASE.induction;

  const totalResults = db.claims.length + db.features.length + db.neurons.length + db.circuits.length + db.counterexamples.length + db.papers.length;

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      background: 'rgba(5, 5, 15, 0.85)', backdropFilter: 'blur(8px)',
      display: 'flex', justifyContent: 'center', alignItems: 'center', zIndex: 1000,
      fontFamily: "'Inter', sans-serif"
    }}>
      <div style={{
        background: '#12122a', width: 800, maxHeight: '85vh', borderRadius: 16,
        border: '1px solid #3a3a5a', padding: 24, display: 'flex', flexDirection: 'column',
        boxShadow: '0 20px 50px rgba(0,0,0,0.6)', color: '#e0e0ff'
      }}>
        {/* Search Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16 }}>
          <h2 style={{ fontSize: 18, fontWeight: 700, margin: 0, color: '#d0c0ff', display: 'flex', alignItems: 'center', gap: 8 }}>
            🧠 Discovery Memory Search
          </h2>
          <button onClick={onClose} style={{ background: 'transparent', border: 'none', color: '#888', cursor: 'pointer', fontSize: 18 }}>✕</button>
        </div>

        {/* Input Bar */}
        <div style={{ marginBottom: 16 }}>
          <input
            autoFocus
            style={{
              width: '100%', padding: '12px 16px', background: '#1a1a2e', color: '#fff',
              border: '1px solid #5cd4c4', borderRadius: 10, fontSize: 14, outline: 'none'
            }}
            placeholder="Type research question e.g. 'circuits involving induction', 'French cities', 'Name Mover'..."
            value={query}
            onChange={e => setQuery(e.target.value)}
          />
        </div>

        {/* Category Tabs */}
        <div style={{ display: 'flex', gap: 6, marginBottom: 16, borderBottom: '1px solid #2a2a4a', paddingBottom: 10 }}>
          {['all', 'claims', 'features', 'neurons', 'circuits', 'counterexamples', 'papers'].map(cat => (
            <button
              key={cat}
              onClick={() => setActiveCategory(cat)}
              style={{
                padding: '4px 12px', borderRadius: 6, border: 'none', cursor: 'pointer', fontSize: 11, fontWeight: 600,
                background: activeCategory === cat ? '#2a2a5a' : '#151528',
                color: activeCategory === cat ? '#5cd4c4' : '#888', textTransform: 'capitalize'
              }}
            >
              {cat}
            </button>
          ))}
        </div>

        {/* Results List */}
        <div style={{ flex: 1, overflowY: 'auto', display: 'flex', flexDirection: 'column', gap: 12, paddingRight: 6 }}>
          
          {/* Claims */}
          {(activeCategory === 'all' || activeCategory === 'claims') && db.claims.map(item => (
            <div key={item.id} style={{ background: '#151528', borderLeft: '3px solid #2ea043', borderRadius: '0 8px 8px 0', padding: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <span style={{ fontSize: 11, color: '#2ea043', fontWeight: 700 }}>🏛️ MECHANISM CLAIM</span>
                <span style={{ fontSize: 11, color: '#888' }}>Confidence: {Math.round(item.confidence * 100)}%</span>
              </div>
              <div style={{ fontSize: 14, fontWeight: 600, color: '#fff', marginBottom: 4 }}>{item.title}</div>
              <div style={{ fontSize: 12, color: '#bbb' }}>{item.summary}</div>
            </div>
          ))}

          {/* SAE Features */}
          {(activeCategory === 'all' || activeCategory === 'features') && db.features.map(item => (
            <div key={item.id} style={{ background: '#151528', borderLeft: '3px solid #d0c0ff', borderRadius: '0 8px 8px 0', padding: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <span style={{ fontSize: 11, color: '#d0c0ff', fontWeight: 700 }}>🧬 SAE FEATURE</span>
                <span style={{ fontSize: 11, color: '#888' }}>Match: {Math.round(item.score * 100)}%</span>
              </div>
              <div style={{ fontSize: 14, fontWeight: 600, color: '#fff', marginBottom: 4 }}>{item.title} ({item.id})</div>
              <div style={{ fontSize: 12, color: '#bbb' }}>{item.summary}</div>
            </div>
          ))}

          {/* Neurons */}
          {(activeCategory === 'all' || activeCategory === 'neurons') && db.neurons.map(item => (
            <div key={item.id} style={{ background: '#151528', borderLeft: '3px solid #5cd4c4', borderRadius: '0 8px 8px 0', padding: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <span style={{ fontSize: 11, color: '#5cd4c4', fontWeight: 700 }}>🧠 ATTENTION HEAD / NEURON</span>
                <span style={{ fontSize: 11, color: '#888' }}>Match: {Math.round(item.score * 100)}%</span>
              </div>
              <div style={{ fontSize: 14, fontWeight: 600, color: '#fff', marginBottom: 4 }}>{item.title}</div>
              <div style={{ fontSize: 12, color: '#bbb' }}>{item.summary}</div>
            </div>
          ))}

          {/* Circuits */}
          {(activeCategory === 'all' || activeCategory === 'circuits') && db.circuits.map(item => (
            <div key={item.id} style={{ background: '#151528', borderLeft: '3px solid #feca57', borderRadius: '0 8px 8px 0', padding: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <span style={{ fontSize: 11, color: '#feca57', fontWeight: 700 }}>🔍 COMPUTATIONAL CIRCUIT</span>
                <span style={{ fontSize: 11, color: '#888' }}>Score: {Math.round(item.score * 100)}%</span>
              </div>
              <div style={{ fontSize: 14, fontWeight: 600, color: '#fff', marginBottom: 4 }}>{item.title}</div>
              <div style={{ fontSize: 12, color: '#bbb' }}>{item.summary}</div>
            </div>
          ))}

          {/* Counterexamples */}
          {(activeCategory === 'all' || activeCategory === 'counterexamples') && db.counterexamples.map(item => (
            <div key={item.id} style={{ background: '#151528', borderLeft: '3px solid #ff6b6b', borderRadius: '0 8px 8px 0', padding: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <span style={{ fontSize: 11, color: '#ff6b6b', fontWeight: 700 }}>🎯 FALSIFICATION COUNTEREXAMPLE</span>
                <span style={{ fontSize: 11, color: '#888' }}>Score: {Math.round(item.score * 100)}%</span>
              </div>
              <div style={{ fontSize: 14, fontWeight: 600, color: '#fff', marginBottom: 4 }}>{item.title}</div>
              <div style={{ fontSize: 12, color: '#bbb', fontFamily: 'monospace' }}>"{item.summary}"</div>
            </div>
          ))}

          {/* Papers */}
          {(activeCategory === 'all' || activeCategory === 'papers') && db.papers.map(item => (
            <div key={item.id} style={{ background: '#151528', borderLeft: '3px solid #58a6ff', borderRadius: '0 8px 8px 0', padding: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 4 }}>
                <span style={{ fontSize: 11, color: '#58a6ff', fontWeight: 700 }}>📄 LITERATURE CITATION</span>
                <span style={{ fontSize: 11, color: '#888' }}>Match: {Math.round(item.score * 100)}%</span>
              </div>
              <div style={{ fontSize: 14, fontWeight: 600, color: '#fff', marginBottom: 4 }}>{item.title}</div>
              <div style={{ fontSize: 12, color: '#bbb' }}>{item.summary}</div>
            </div>
          ))}

        </div>

        {/* Footer */}
        <div style={{ marginTop: 16, paddingTop: 12, borderTop: '1px solid #2a2a4a', display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: 12, color: '#888' }}>
          <span>Found {totalResults} memory entries for "{query}"</span>
          <button onClick={onClose} style={{ padding: '6px 14px', borderRadius: 6, background: '#2a2a5a', border: 'none', color: '#fff', cursor: 'pointer' }}>Close</button>
        </div>
      </div>
    </div>
  );
}

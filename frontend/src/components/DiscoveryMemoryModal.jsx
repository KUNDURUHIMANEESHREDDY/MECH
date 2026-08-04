import React, { useState } from 'react';
import { X } from 'lucide-react';

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

const CATEGORIES = ['all', 'claims', 'features', 'neurons', 'circuits', 'counterexamples', 'papers'];

export default function DiscoveryMemoryModal({ isOpen, onClose, onNavigate }) {
  const [query, setQuery] = useState('circuits involving induction');
  const [activeCategory, setActiveCategory] = useState('all');

  if (!isOpen) return null;

  const isIoi = query.toLowerCase().includes('ioi') || query.toLowerCase().includes('name') || query.toLowerCase().includes('french');
  const db = isIoi ? MOCK_MEMORY_DATABASE.ioi : MOCK_MEMORY_DATABASE.induction;

  const totalResults = db.claims.length + db.features.length + db.neurons.length + db.circuits.length + db.counterexamples.length + db.papers.length;

  return (
    <div className="modal-overlay">
      <div className="modal-container">
        <div className="modal-header">
          <h2>Discovery Memory Search</h2>
          <button onClick={onClose} className="modal-close-btn"><X size={14} /></button>
        </div>

        <div style={{ marginBottom: 16 }}>
          <input
            autoFocus
            className="search-input"
            placeholder="Type research question e.g. 'circuits involving induction', 'French cities', 'Name Mover'..."
            value={query}
            onChange={e => setQuery(e.target.value)}
          />
        </div>

        <div className="category-tabs">
          {CATEGORIES.map(cat => (
            <button
              key={cat}
              onClick={() => setActiveCategory(cat)}
              className={'category-tab' + (activeCategory === cat ? ' active' : '')}
            >
              {cat}
            </button>
          ))}
        </div>

        <div className="results-list">
          {(activeCategory === 'all' || activeCategory === 'claims') && db.claims.map(item => (
            <div key={item.id} className="result-item claim">
              <div className="result-type">
                <span style={{ color: 'var(--success)' }}>MECHANISM CLAIM</span>
                <span style={{ color: 'var(--text-dim)' }}>Confidence: {Math.round(item.confidence * 100)}%</span>
              </div>
              <div className="result-title">{item.title}</div>
              <div className="result-summary">{item.summary}</div>
            </div>
          ))}

          {(activeCategory === 'all' || activeCategory === 'features') && db.features.map(item => (
            <div key={item.id} className="result-item feature">
              <div className="result-type">
                <span style={{ color: 'var(--warning)' }}>SAE FEATURE</span>
                <span style={{ color: 'var(--text-dim)' }}>Match: {Math.round(item.score * 100)}%</span>
              </div>
              <div className="result-title">{item.title} ({item.id})</div>
              <div className="result-summary">{item.summary}</div>
            </div>
          ))}

          {(activeCategory === 'all' || activeCategory === 'neurons') && db.neurons.map(item => (
            <div key={item.id} className="result-item neuron">
              <div className="result-type">
                <span style={{ color: 'var(--accent-hover)' }}>ATTENTION HEAD / NEURON</span>
                <span style={{ color: 'var(--text-dim)' }}>Match: {Math.round(item.score * 100)}%</span>
              </div>
              <div className="result-title">{item.title}</div>
              <div className="result-summary">{item.summary}</div>
            </div>
          ))}

          {(activeCategory === 'all' || activeCategory === 'circuits') && db.circuits.map(item => (
            <div key={item.id} className="result-item circuit">
              <div className="result-type">
                <span style={{ color: 'var(--success)' }}>COMPUTATIONAL CIRCUIT</span>
                <span style={{ color: 'var(--text-dim)' }}>Score: {Math.round(item.score * 100)}%</span>
              </div>
              <div className="result-title">{item.title}</div>
              <div className="result-summary">{item.summary}</div>
            </div>
          ))}

          {(activeCategory === 'all' || activeCategory === 'counterexamples') && db.counterexamples.map(item => (
            <div key={item.id} className="result-item counterexample">
              <div className="result-type">
                <span style={{ color: 'var(--danger)' }}>FALSIFICATION COUNTEREXAMPLE</span>
                <span style={{ color: 'var(--text-dim)' }}>Score: {Math.round(item.score * 100)}%</span>
              </div>
              <div className="result-title">{item.title}</div>
              <div className="result-summary mono">"{item.summary}"</div>
            </div>
          ))}

          {(activeCategory === 'all' || activeCategory === 'papers') && db.papers.map(item => (
            <div key={item.id} className="result-item paper">
              <div className="result-type">
                <span style={{ color: 'var(--accent)' }}>LITERATURE CITATION</span>
                <span style={{ color: 'var(--text-dim)' }}>Match: {Math.round(item.score * 100)}%</span>
              </div>
              <div className="result-title">{item.title}</div>
              <div className="result-summary">{item.summary}</div>
            </div>
          ))}
        </div>

        <div className="modal-footer">
          <span>Found {totalResults} memory entries for "{query}"</span>
          <button onClick={onClose} className="btn">Close</button>
        </div>
      </div>
    </div>
  );
}

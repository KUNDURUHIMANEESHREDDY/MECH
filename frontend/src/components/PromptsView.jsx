import React, { useState } from 'react';

export default function PromptsView({ api }) {
  const [promptText, setPromptText] = useState('When Mary and John went to the store, John gave a book to');
  const [prompts, setPrompts] = useState([
    { id: '1', title: 'IOI Task (Indirect Object Identification)', prompt: 'When Mary and John went to the store, John gave a book to Mary' },
    { id: '2', title: 'Greater-Than Temporal Constraint', prompt: 'The War lasted from 1914 to 1918. The next year was 19' },
    { id: '3', title: 'Gender Bias Probing', prompt: 'The doctor called the nurse because she was' }
  ]);

  const handleSave = () => {
    if (!promptText.trim()) return;
    const newP = {
      id: String(Date.now()),
      title: promptText.slice(0, 30) + '...',
      prompt: promptText
    };
    setPrompts([newP, ...prompts]);
  };

  return (
    <div className="prompts-view" data-testid="prompts-view">
      <div className="section-header">
        <h2>Prompt & Dataset Manager</h2>
        <p>Curate input sequences for circuit discovery and activation patching.</p>
      </div>

      <div className="card" style={{ marginBottom: '20px' }}>
        <h3>Prompt Editor</h3>
        <textarea
          className="input-textarea"
          rows={3}
          value={promptText}
          onChange={(e) => setPromptText(e.target.value)}
          placeholder="Enter prompt string for activation probing..."
        />
        <div style={{ marginTop: '10px', display: 'flex', gap: '8px' }}>
          <button className="btn btn-primary" onClick={handleSave}>Save Prompt Template</button>
        </div>
      </div>

      <div className="card">
        <h3>Saved Research Prompts</h3>
        <div className="prompt-list">
          {prompts.map((p) => (
            <div key={p.id} className="prompt-item" onClick={() => setPromptText(p.prompt)}>
              <strong>{p.title}</strong>
              <p className="prompt-code">"{p.prompt}"</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

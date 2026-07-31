import React, { useState } from 'react';
import { Bell, Check, HeartPulse, Play, Settings2, Trophy, Zap } from 'lucide-react';

/**
 * ScientificHealthView - Continuous Validation & Regression Monitoring Dashboard
 *
 * Displays continuous platform health metrics, golden benchmark pass rates, 
 * actionable scientific alerts, and environment version snapshots.
 */

const MOCK_HEALTH = {
  health_score: 98.6,
  reproducibility_score: 97.8,
  pass_rate: 100.0,
  total_benchmarks: 5,
  passed_benchmarks: 5,
  environment: {
    torch_version: '2.3.0+cu121',
    transformers_version: '4.39.0',
    cuda_version: '12.1',
    python_version: '3.10.12',
    models: 'GPT2-S, GPT2-M, Gemma-2B, Llama3-8B'
  },
  benchmarks: [
    { id: 'bm_ioi', name: 'IOI Circuit Recovery', model: 'GPT2-S', baseline: '0.880', current: '0.876', delta: '-0.4%', runtime: '1.2s', status: 'PASS' },
    { id: 'bm_ind', name: 'Induction Head Sequence Repeater', model: 'Gemma-2B', baseline: '0.940', current: '0.935', delta: '-0.5%', runtime: '2.1s', status: 'PASS' },
    { id: 'bm_gt', name: 'Greater-Than Comparative Circuit', model: 'GPT2-M', baseline: '0.860', current: '0.858', delta: '-0.2%', runtime: '1.8s', status: 'PASS' },
    { id: 'bm_arith', name: 'Multi-Digit Arithmetic Circuit', model: 'Llama3-8B', baseline: '0.910', current: '0.908', delta: '-0.2%', runtime: '3.4s', status: 'PASS' },
    { id: 'bm_sae', name: 'SAE Feature Dictionary Recovery', model: 'GPT2-S', baseline: '0.895', current: '0.891', delta: '-0.4%', runtime: '1.5s', status: 'PASS' },
  ],
  alerts: [
    {
      id: 'a1',
      title: 'Platform Health Optimal',
      severity: 'INFO',
      desc: 'All 5 golden benchmarks passing baseline fidelity and latency specs.',
      cause: 'Stable runtime environment state.',
      action: 'No action required.'
    }
  ]
};

export default function ScientificHealthView({ api, onNavigate }) {
  const [runningSuite, setRunningSuite] = useState(false);

  const handleRunSuite = () => {
    setRunningSuite(true);
    setTimeout(() => {
      setRunningSuite(false);
    }, 1200);
  };

  return (
    <div style={{ padding: 28, background: '#0b0b1a', color: '#e0e0ff', height: '100%', overflowY: 'auto', fontFamily: "'Inter', sans-serif" }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <div>
          <h1 style={{ fontSize: 24, fontWeight: 700, margin: '0 0 6px 0', color: '#d0c0ff', display: 'flex', alignItems: 'center', gap: 8 }}><HeartPulse size={20} /> Scientific Health & Continuous Validation</h1>
          <p style={{ margin: 0, color: '#888', fontSize: 13 }}>
            Continuous regression monitoring across PyTorch, CUDA, Transformers, and Golden Benchmarks
          </p>
        </div>
        <button
          onClick={handleRunSuite}
          disabled={runningSuite}
          style={{
            background: runningSuite ? '#1a1a3a' : '#2ea043',
            color: '#fff', border: 'none', padding: '10px 18px', borderRadius: 8,
            cursor: runningSuite ? 'default' : 'pointer', fontSize: 13, fontWeight: 700
          }}
        >
          {runningSuite ? <><Zap size={14} style={{ verticalAlign: 'middle', marginRight: 6 }} />Running Validation...</> : <><Play size={14} style={{ verticalAlign: 'middle', marginRight: 6 }} />Run Validation Suite</>}
        </button>
      </div>

      {/* Health Overview Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
        <div style={{ background: '#12122a', border: '1px solid #2a2a4a', padding: 20, borderRadius: 12, textAlign: 'center' }}>
          <div style={{ fontSize: 11, color: '#888', textTransform: 'uppercase' }}>Platform Health Score</div>
          <div style={{ fontSize: 26, fontWeight: 800, color: '#2ea043', marginTop: 4 }}>{MOCK_HEALTH.health_score}%</div>
        </div>
        <div style={{ background: '#12122a', border: '1px solid #2a2a4a', padding: 20, borderRadius: 12, textAlign: 'center' }}>
          <div style={{ fontSize: 11, color: '#888', textTransform: 'uppercase' }}>Reproducibility Score</div>
          <div style={{ fontSize: 26, fontWeight: 800, color: '#5cd4c4', marginTop: 4 }}>{MOCK_HEALTH.reproducibility_score}%</div>
        </div>
        <div style={{ background: '#12122a', border: '1px solid #2a2a4a', padding: 20, borderRadius: 12, textAlign: 'center' }}>
          <div style={{ fontSize: 11, color: '#888', textTransform: 'uppercase' }}>Benchmark Pass Rate</div>
          <div style={{ fontSize: 26, fontWeight: 800, color: '#2ea043', marginTop: 4 }}>{MOCK_HEALTH.pass_rate}%</div>
        </div>
        <div style={{ background: '#12122a', border: '1px solid #2a2a4a', padding: 20, borderRadius: 12, textAlign: 'center' }}>
          <div style={{ fontSize: 11, color: '#888', textTransform: 'uppercase' }}>Active Regressions</div>
          <div style={{ fontSize: 26, fontWeight: 800, color: '#58a6ff', marginTop: 4 }}>0 Active</div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: 20 }}>
        
        {/* Left Column: Golden Benchmarks Matrix Table */}
        <div style={{ background: '#12122a', borderRadius: 12, border: '1px solid #2a2a4a', padding: 20 }}>
          <h3 style={{ fontSize: 15, color: '#d0c0ff', margin: '0 0 14px 0', display: 'flex', alignItems: 'center', gap: 6 }}><Trophy size={15} /> Golden Benchmarks Validation Matrix</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, textAlign: 'left' }}>
            <thead>
              <tr style={{ background: '#151528', color: '#888', borderBottom: '1px solid #2a2a4a' }}>
                <th style={{ padding: 10 }}>Benchmark</th>
                <th style={{ padding: 10 }}>Model</th>
                <th style={{ padding: 10 }}>Baseline</th>
                <th style={{ padding: 10 }}>Current</th>
                <th style={{ padding: 10 }}>Delta</th>
                <th style={{ padding: 10 }}>Runtime</th>
                <th style={{ padding: 10 }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {MOCK_HEALTH.benchmarks.map((bm, idx) => (
                <tr key={bm.id} style={{ borderBottom: '1px solid #1a1a3a', background: idx % 2 === 0 ? '#12122a' : '#151528' }}>
                  <td style={{ padding: 10, fontWeight: 700, color: '#fff' }}>{bm.name}</td>
                  <td style={{ padding: 10, color: '#5cd4c4', fontWeight: 600 }}>{bm.model}</td>
                  <td style={{ padding: 10, color: '#aaa' }}>{bm.baseline}</td>
                  <td style={{ padding: 10, fontWeight: 700, color: '#2ea043' }}>{bm.current}</td>
                  <td style={{ padding: 10, color: '#888' }}>{bm.delta}</td>
                  <td style={{ padding: 10, color: '#888' }}>{bm.runtime}</td>
                  <td style={{ padding: 10 }}>
                    <span style={{ background: '#1c3d27', color: '#2ea043', border: '1px solid #2ea043', padding: '2px 8px', borderRadius: 4, fontSize: 10, fontWeight: 700 }}>
                      <Check size={11} style={{ verticalAlign: 'middle', marginRight: 2 }} /> {bm.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Right Column: Actionable Alerts & Environment Profile */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
          
          {/* Actionable Alerts Feed */}
          <div style={{ background: '#12122a', borderRadius: 12, border: '1px solid #2a2a4a', padding: 20 }}>
            <h3 style={{ fontSize: 15, color: '#2ea043', margin: '0 0 12px 0', display: 'flex', alignItems: 'center', gap: 6 }}><Bell size={15} /> Actionable Scientific Alerts</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              {MOCK_HEALTH.alerts.map(a => (
                <div key={a.id} style={{ background: '#151528', border: '1px solid #2a2a4a', borderRadius: 8, padding: 12, borderLeft: '3px solid #2ea043' }}>
                  <div style={{ fontSize: 13, fontWeight: 700, color: '#fff', marginBottom: 4 }}>{a.title}</div>
                  <div style={{ fontSize: 12, color: '#bbb', marginBottom: 6 }}>{a.desc}</div>
                  <div style={{ fontSize: 11, color: '#5cd4c4' }}>Action: {a.action}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Environment Version Snapshot */}
          <div style={{ background: '#12122a', borderRadius: 12, border: '1px solid #2a2a4a', padding: 20 }}>
            <h3 style={{ fontSize: 15, color: '#d0c0ff', margin: '0 0 12px 0', display: 'flex', alignItems: 'center', gap: 6 }}><Settings2 size={15} /> Runtime Environment Profile</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, fontSize: 12 }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', background: '#151528', padding: 8, borderRadius: 6 }}>
                <span style={{ color: '#888' }}>PyTorch Version</span>
                <span style={{ color: '#5cd4c4', fontWeight: 600 }}>{MOCK_HEALTH.environment.torch_version}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', background: '#151528', padding: 8, borderRadius: 6 }}>
                <span style={{ color: '#888' }}>Transformers Version</span>
                <span style={{ color: '#5cd4c4', fontWeight: 600 }}>{MOCK_HEALTH.environment.transformers_version}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', background: '#151528', padding: 8, borderRadius: 6 }}>
                <span style={{ color: '#888' }}>CUDA Toolkit</span>
                <span style={{ color: '#5cd4c4', fontWeight: 600 }}>{MOCK_HEALTH.environment.cuda_version}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', background: '#151528', padding: 8, borderRadius: 6 }}>
                <span style={{ color: '#888' }}>Validated Models</span>
                <span style={{ color: '#d0c0ff', fontWeight: 600 }}>{MOCK_HEALTH.environment.models}</span>
              </div>
            </div>
          </div>

        </div>

      </div>
    </div>
  );
}


import React from 'react';
import { BarChart3, Cpu, Zap } from 'lucide-react';
import { useModel } from '../hooks/useModel';
import './HomeDashboard.css';

export const HomeDashboard: React.FC = () => {
  const { state: model } = useModel();

  return (
    <div className="home-dashboard">
      <div className="home-hero">
        <h1 className="home-title">MECH Platform</h1>
        <p className="home-subtitle">Mechanistic Interpretability Research Environment</p>
      </div>

      <div className="home-grid">
        <div className="home-card">
          <Cpu size={22} />
          <div>
            <div className="home-card-title">Model</div>
            <div className="home-card-value">{model.modelInfo?.model_name || 'Not loaded'}</div>
          </div>
        </div>

        <div className="home-card">
          <Zap size={22} />
          <div>
            <div className="home-card-title">Status</div>
            <div className="home-card-value">{model.loaded ? 'Ready' : 'Idle'}</div>
          </div>
        </div>

        <div className="home-card">
          <BarChart3 size={22} />
          <div>
            <div className="home-card-title">Layers</div>
            <div className="home-card-value">{model.modelInfo?.num_layers ?? '—'}</div>
          </div>
        </div>
      </div>
    </div>
  );
};

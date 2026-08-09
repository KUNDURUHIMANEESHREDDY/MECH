import React from 'react';
import { useAppStore } from '../store/useAppStore';
import { Activity, Zap } from 'lucide-react';
import './StatusBar.css';

export const StatusBar: React.FC = () => {
  const { pythonStatus, activeModel, gpuUtil, memoryUtil } = useAppStore();

  return (
    <footer className="status-bar">
      <div className="status-bar-left">
        <span className="status-item">
          <Zap size={12} />
          {activeModel || 'No Model'}
        </span>
      </div>

      <div className="status-bar-center">
        <span className="status-item">
          GPU: {gpuUtil ?? 0}%
        </span>
        <span className="status-item">
          Mem: {memoryUtil ?? 0}%
        </span>
      </div>

      <div className="status-bar-right">
        <span className={`status-item python-status ${pythonStatus ? 'online' : 'offline'}`} data-testid="python-status">
          <Activity size={12} />
          {pythonStatus ? 'Python Ready' : 'Python Offline'}
        </span>
      </div>
    </footer>
  );
};

import React from 'react';
import './PythonSettings.css';

interface PythonSettingsProps {
  settings: Record<string, unknown>;
  onChange: (settings: Record<string, unknown>) => void;
}

export const PythonSettings: React.FC<PythonSettingsProps> = ({ settings, onChange }) => {
  return (
    <div className="python-settings">
      <h1>Python Settings</h1>
      <p>Configure Python backend connection.</p>
    </div>
  );
};

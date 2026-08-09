import React from 'react';
import './ThemeSettings.css';

interface ThemeSettingsProps {
  settings: { theme: string };
  onChange: (settings: { theme: string }) => void;
}

export const ThemeSettings: React.FC<ThemeSettingsProps> = ({ settings, onChange }) => {
  return (
    <div className="theme-settings">
      <h1>Theme Settings</h1>
      <select
        data-testid="theme-select"
        value={settings.theme}
        onChange={(e) => onChange({ theme: e.target.value })}
        className="theme-select"
      >
        <option value="light">Light</option>
        <option value="dark">Dark</option>
        <option value="system">System</option>
      </select>
    </div>
  );
};

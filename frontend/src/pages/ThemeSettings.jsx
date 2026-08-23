import React, { useState } from 'react';
import './ThemeSettings.css';

export const ThemeSettings = ({ settings = { theme: 'system' }, onChange }) => {
  const [theme, setTheme] = useState(settings.theme || 'system');
  const [accentColor, setAccentColor] = useState(settings.accentColor || 'emerald');
  const [editorFontSize, setEditorFontSize] = useState(settings.editorFontSize || 14);
  const [reduceMotion, setReduceMotion] = useState(settings.reduceMotion ?? false);
  const [highContrast, setHighContrast] = useState(settings.highContrast ?? false);

  const handleUpdate = (patch) => {
    if (onChange) {
      onChange({ theme, accentColor, editorFontSize, reduceMotion, highContrast, ...patch });
    }
  };

  return (
    <div className="theme-settings settings-page-container">
      <div className="settings-page-header">
        <h1>Theme & Appearance</h1>
        <p>Customize the visual interface, dark/light contrast modes, color palette, and typography.</p>
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="theme-select">Interface Theme</label>
        <select
          id="theme-select"
          data-testid="theme-select"
          value={theme}
          onChange={(e) => {
            setTheme(e.target.value);
            handleUpdate({ theme: e.target.value });
          }}
          className="settings-select theme-select"
        >
          <option value="system">System Synchronized</option>
          <option value="dark">Dark Theme (Deep Slate)</option>
          <option value="light">Light Theme (Clean Paper)</option>
        </select>
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="accent-color-select">Visual Accent Palette</label>
        <select
          id="accent-color-select"
          data-testid="accent-color-select"
          className="settings-select"
          value={accentColor}
          onChange={(e) => {
            setAccentColor(e.target.value);
            handleUpdate({ accentColor: e.target.value });
          }}
        >
          <option value="emerald">Emerald Green (Default)</option>
          <option value="blue">Deep Indigo Blue</option>
          <option value="violet">Cyber Violet</option>
          <option value="amber">Warm Amber</option>
        </select>
      </div>

      <div className="settings-section">
        <label className="settings-label" htmlFor="font-size-slider">
          Editor & Code Font Size: {editorFontSize} px
        </label>
        <input
          id="font-size-slider"
          data-testid="font-size-slider"
          type="range"
          min={11}
          max={22}
          step={1}
          className="settings-slider"
          value={editorFontSize}
          onChange={(e) => {
            const val = parseInt(e.target.value, 10);
            setEditorFontSize(val);
            handleUpdate({ editorFontSize: val });
          }}
        />
      </div>

      <div className="settings-section checkbox-section">
        <label className="checkbox-label">
          <input
            data-testid="high-contrast-toggle"
            type="checkbox"
            checked={highContrast}
            onChange={(e) => {
              setHighContrast(e.target.checked);
              handleUpdate({ highContrast: e.target.checked });
            }}
          />
          <span>High Contrast Mode (Enhance border and text readability)</span>
        </label>
      </div>

      <div className="settings-section checkbox-section">
        <label className="checkbox-label">
          <input
            data-testid="reduce-motion-toggle"
            type="checkbox"
            checked={reduceMotion}
            onChange={(e) => {
              setReduceMotion(e.target.checked);
              handleUpdate({ reduceMotion: e.target.checked });
            }}
          />
          <span>Reduce UI Animations & Transitions</span>
        </label>
      </div>
    </div>
  );
};

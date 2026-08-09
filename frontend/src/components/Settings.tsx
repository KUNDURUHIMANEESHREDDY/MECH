import React, { useEffect, useState } from 'react';
import { Settings2, Save, Check } from 'lucide-react';
import { colors } from '../design/tokens/colors';

type AppTheme = 'system' | 'dark' | 'light' | 'midnight' | 'nord' | 'dracula' | 'solarized-dark' | 'solarized-light';

interface SettingsForm {
  theme: AppTheme;
  gpuEnabled: boolean;
  cachePath: string;
  modelPath: string;
  workspacePath: string;
}

const THEMES: AppTheme[] = ['system', 'dark', 'light', 'midnight', 'nord', 'dracula', 'solarized-dark', 'solarized-light'];

const DEFAULT_SETTINGS: SettingsForm = { theme: 'system', gpuEnabled: false, cachePath: '', modelPath: '', workspacePath: '' };

const rowStyle: React.CSSProperties = { display: 'flex', alignItems: 'center', gap: 10, padding: '8px 0', borderBottom: `1px solid ${colors.dividerSoft}` };
const labelStyle: React.CSSProperties = { fontSize: 12, color: colors.ink, fontWeight: 600, width: 140, flexShrink: 0 };
const inputStyle: React.CSSProperties = { flex: 1, padding: 6, borderRadius: 6, border: `1px solid ${colors.hairline}`, fontSize: 12, background: colors.canvas, color: colors.ink };

const toAppSettings = (s: SettingsForm): DeepPartial<AppSettings> => ({
  theme: s.theme,
  gpu: { enabled: s.gpuEnabled },
  cache: { location: s.cachePath },
  models: { defaultModel: s.modelPath },
  paths: { workspace: s.workspacePath },
});

export const Settings: React.FC = () => {
  const [settings, setSettings] = useState<SettingsForm>(DEFAULT_SETTINGS);
  const [loaded, setLoaded] = useState(false);
  const [saved, setSaved] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    window.appApi
      .getSettings()
      .then((s) => {
        setSettings({
          theme: (s.theme as AppTheme) || 'system',
          gpuEnabled: s.gpu?.enabled ?? false,
          cachePath: s.cache?.location ?? '',
          modelPath: s.models?.defaultModel ?? '',
          workspacePath: s.paths?.workspace ?? '',
        });
        setLoaded(true);
      })
      .catch(() => setLoaded(true));
  }, []);

  const save = async () => {
    setSaving(true);
    try {
      await window.appApi.setSettings(toAppSettings(settings));
      setSaved(new Date().toLocaleTimeString());
    } catch {
      void 0;
    } finally {
      setSaving(false);
    }
  };

  const set = <K extends keyof SettingsForm>(key: K, value: SettingsForm[K]) => setSettings((s) => ({ ...s, [key]: value }));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body, padding: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Settings2 size={16} color={colors.primary} />
        <span style={{ fontWeight: 700, color: colors.ink, fontSize: 15 }}>System Settings</span>
        <span style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
          {saved && (
            <span style={{ fontSize: 11, color: colors.success, display: 'inline-flex', alignItems: 'center', gap: 4 }}>
              <Check size={12} /> saved {saved}
            </span>
          )}
          <button onClick={() => void save()} disabled={!loaded || saving} style={{ background: colors.primary, color: colors.onPrimary, border: 'none', borderRadius: 6, padding: '6px 14px', fontSize: 12, fontWeight: 600, cursor: 'pointer', display: 'inline-flex', alignItems: 'center', gap: 6, opacity: !loaded || saving ? 0.6 : 1 }}>
            <Save size={13} /> {saving ? 'Saving…' : 'Save'}
          </button>
        </span>
      </div>

      {!loaded && <div style={{ fontSize: 12, color: colors.bodyMuted }}>Loading settings from the Electron store…</div>}

      <div style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: '4px 12px' }}>
        <div style={rowStyle}>
          <span style={labelStyle}>Theme</span>
          <select value={settings.theme} onChange={(e) => set('theme', e.target.value as AppTheme)} style={{ ...inputStyle, width: 200 }}>
            {THEMES.map((t) => (
              <option key={t} value={t}>{t}</option>
            ))}
          </select>
        </div>
        <div style={rowStyle}>
          <span style={labelStyle}>GPU enabled</span>
          <input type="checkbox" checked={settings.gpuEnabled} onChange={(e) => set('gpuEnabled', e.target.checked)} style={{ width: 16, height: 16 }} />
        </div>
        <div style={rowStyle}>
          <span style={labelStyle}>Cache path</span>
          <input value={settings.cachePath} onChange={(e) => set('cachePath', e.target.value)} placeholder="/path/to/cache" style={inputStyle} />
        </div>
        <div style={rowStyle}>
          <span style={labelStyle}>Default model</span>
          <input value={settings.modelPath} onChange={(e) => set('modelPath', e.target.value)} placeholder="GPT-2 Small" style={inputStyle} />
        </div>
        <div style={rowStyle}>
          <span style={labelStyle}>Workspace path</span>
          <input value={settings.workspacePath} onChange={(e) => set('workspacePath', e.target.value)} placeholder="default" style={inputStyle} />
        </div>
      </div>

      <div style={{ fontSize: 11, color: colors.bodyMuted }}>
        Persisted to the SQLite app store via the `settings:set` bridge; the sidecar reads these values on boot.
      </div>
    </div>
  );
};
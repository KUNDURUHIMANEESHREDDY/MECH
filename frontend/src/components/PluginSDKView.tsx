import React from 'react';
import { Puzzle } from 'lucide-react';
import { colors } from '../design/tokens/colors';
import { pluginRegistry } from '../panel-system/pluginRegistry';

const dockColor = (dock: string) => (dock === 'left' || dock === 'right' ? colors.purple : dock === 'bottom' ? colors.warningText : colors.primary);

export const PluginSDKView: React.FC = () => {
  const plugins = pluginRegistry.list();
  const categories = Array.from(new Set(plugins.map((p) => p.category)));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, fontSize: 13, color: colors.body, padding: 4 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <Puzzle size={16} color={colors.primary} />
        <span style={{ fontWeight: 700, color: colors.ink, fontSize: 15 }}>Plugin SDK</span>
        <span style={{ marginLeft: 'auto', fontSize: 11, color: colors.bodyMuted }}>{plugins.length} registered</span>
      </div>

      {plugins.length === 0 && (
        <div style={{ fontSize: 12, color: colors.bodyMuted, border: `1px dashed ${colors.hairline}`, borderRadius: 8, padding: 16 }}>
          No plugins registered. Panels register themselves through <code>pluginRegistry.register</code>.
        </div>
      )}

      {categories.map((cat) => (
        <div key={cat} style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          <div style={{ fontWeight: 700, color: colors.ink, textTransform: 'capitalize', fontSize: 12 }}>{cat}</div>
          {plugins
            .filter((p) => p.category === cat)
            .map((p) => (
              <div key={p.id} style={{ border: `1px solid ${colors.hairline}`, borderRadius: 8, background: colors.canvas, padding: 10, display: 'flex', flexDirection: 'column', gap: 6 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontWeight: 700, color: colors.ink }}>{p.title}</span>
                  <span style={{ fontSize: 11, color: colors.bodyMuted, fontFamily: 'monospace' }}>{p.id}</span>
                  <span style={{ marginLeft: 'auto', fontSize: 10, textTransform: 'uppercase', letterSpacing: 0.4, color: dockColor(p.defaultDock), background: colors.surfacePearl, borderRadius: 10, padding: '2px 8px' }}>{p.defaultDock}</span>
                </div>
                <div style={{ fontSize: 11, color: colors.bodyMuted, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                  {p.resourceKinds.map((k) => (
                    <span key={k} style={{ background: colors.canvasParchment, border: `1px solid ${colors.hairline}`, borderRadius: 10, padding: '1px 8px' }}>{k}</span>
                  ))}
                </div>
              </div>
            ))}
        </div>
      ))}
    </div>
  );
};

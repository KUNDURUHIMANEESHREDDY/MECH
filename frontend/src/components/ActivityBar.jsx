import React from 'react';
import { LayoutGrid, Search, Bug, FileCode2, Settings } from 'lucide-react';

const TOP_ICONS = [
  { id: 'explorer', icon: LayoutGrid, label: 'Explorer', section: 'explorer' },
  { id: 'search', icon: Search, label: 'Search' },
  { id: 'debug', icon: Bug, label: 'Debug', section: 'debugger' },
  { id: 'code', icon: FileCode2, label: 'Code' },
];

export default function ActivityBar({ active, onSelect }) {
  return (
    <div className="activity-bar">
      <div className="activity-bar-top">
        <div className="activity-logo" title="MECH Platform">
          <span>M</span>
        </div>
        {TOP_ICONS.map(item => {
          const Icon = item.icon;
          return (
            <button
              key={item.id}
              className={'activity-icon' + (active === item.section ? ' active' : '')}
              onClick={() => item.section && onSelect(item.section)}
              title={item.label}
            >
              <Icon size={18} strokeWidth={1.75} />
            </button>
          );
        })}
      </div>
      <div className="activity-bar-bottom">
        <button
          className={'activity-icon' + (active === 'settings' ? ' active' : '')}
          onClick={() => onSelect('settings')}
          title="Settings"
        >
          <Settings size={18} strokeWidth={1.75} />
        </button>
      </div>
    </div>
  );
}

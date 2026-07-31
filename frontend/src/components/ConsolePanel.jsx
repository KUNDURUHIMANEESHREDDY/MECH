import React, { useState } from 'react';
import { Terminal, ChevronUp, ChevronDown } from 'lucide-react';

export default function ConsolePanel({ logs = [] }) {
  const [collapsed, setCollapsed] = useState(true);

  return (
    <div className="console-panel">
      <div className="console-bar" onClick={() => setCollapsed(!collapsed)}>
        <span><Terminal size={13} /> Console</span>
        <span>
          {collapsed ? <ChevronUp size={13} /> : <ChevronDown size={13} />} {logs.length} lines
        </span>
      </div>
      {!collapsed && (
        <div className="console-body">
          {logs.length === 0 && <div className="console-line">[System ready]</div>}
          {logs.map((line, i) => (
            <div key={i} className={'console-line' + (line.type ? ' ' + line.type : '')}>
              &gt; {line.text || line}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

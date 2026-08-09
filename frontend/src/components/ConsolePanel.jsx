import React, { useState, useEffect } from 'react';
import { X, Trash2 } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { commandBus } from '../utils/eventBus';
import './ConsolePanel.css';

export const ConsolePanel: React.FC = () => {
  const { consoleLogs, clearConsole } = useAppStore();
  const [filter, setFilter] = useState<'all' | 'info' | 'warn' | 'error'>('all');

  useEffect(() => {
    const handler = (entry: { level: 'info' | 'warn' | 'error'; message: string; timestamp: number }) => {
      // console logs are managed by the store
    };
    commandBus.on('console:log', handler);
    return () => commandBus.off('console:log', handler);
  }, []);

  const filtered = filter === 'all' ? consoleLogs : consoleLogs.filter((l) => l.level === filter);

  return (
    <div className="console-panel">
      <div className="console-header">
        <span className="console-title">Console</span>
        <div className="console-actions">
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value as typeof filter)}
            className="console-filter"
          >
            <option value="all">All</option>
            <option value="info">Info</option>
            <option value="warn">Warnings</option>
            <option value="error">Errors</option>
          </select>
          <button className="console-clear" onClick={clearConsole} title="Clear console">
            <Trash2 size={14} />
          </button>
        </div>
      </div>

      <div className="console-body">
        {filtered.length === 0 && (
          <div className="console-empty">No console output.</div>
        )}
        {filtered.map((entry, idx) => (
          <div key={idx} className={`console-entry console-${entry.level}`}>
            <span className="console-time">
              {new Date(entry.timestamp).toLocaleTimeString()}
            </span>
            <span className="console-level">{entry.level.toUpperCase()}</span>
            <span className="console-message">{entry.message}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

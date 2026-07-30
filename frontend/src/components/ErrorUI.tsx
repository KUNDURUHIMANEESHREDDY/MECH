import React from 'react';

interface Props {
  message: string | null;
  onDismiss: () => void;
}

export function ErrorUI({ message, onDismiss }: Props) {
  if (!message) return null;

  return (
    <div style={{
      position: 'fixed', top: 0, left: 0, right: 0, bottom: 0,
      background: 'rgba(0,0,0,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center',
      zIndex: 100,
    }}>
      <div style={{
        background: '#1a1a2a', border: '1px solid #ff4444', borderRadius: 12,
        padding: '24px 32px', maxWidth: 400, textAlign: 'center',
      }}>
        <div style={{ fontSize: 24, marginBottom: 8 }}>⚠</div>
        <div style={{ color: '#ff6666', fontSize: 14, marginBottom: 16, fontWeight: 600 }}>Error</div>
        <div style={{ color: '#ccc', fontSize: 13, marginBottom: 20 }}>{message}</div>
        <button
          onClick={onDismiss}
          style={{
            background: '#3b82f6', color: '#fff', border: 'none', borderRadius: 6,
            padding: '8px 20px', cursor: 'pointer', fontSize: 13,
          }}
        >
          Dismiss
        </button>
      </div>
    </div>
  );
}

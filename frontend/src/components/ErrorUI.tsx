import React from 'react';
import { TriangleAlert } from 'lucide-react';

interface Props {
  message: string | null;
  onDismiss: () => void;
}

export function ErrorUI({ message, onDismiss }: Props) {
  if (!message) return null;

  return (
    <div className="error-overlay">
      <div className="error-modal">
        <div className="error-icon"><TriangleAlert size={36} /></div>
        <div className="error-title">Error</div>
        <div className="error-message">{message}</div>
        <button onClick={onDismiss} className="btn error-dismiss-btn">
          Dismiss
        </button>
      </div>
    </div>
  );
}

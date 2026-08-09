import React from 'react';
import { AlertCircle, X } from 'lucide-react';
import './ErrorUI.css';

interface ErrorUIProps {
  message: string;
  onDismiss?: () => void;
}

export function ErrorUI({ message, onDismiss }: ErrorUIProps) {
  return (
    <div className="error-overlay" role="dialog" aria-modal="true" aria-labelledby="error-title">
      <div className="error-modal">
        <div className="error-header">
          <AlertCircle size={24} className="error-icon" />
          <h2 id="error-title" className="error-title">Error</h2>
          {onDismiss && (
            <button className="error-dismiss" onClick={onDismiss} aria-label="Dismiss">
              <X size={16} />
            </button>
          )}
        </div>
        <div className="error-body">
          <p className="error-message">{message}</p>
        </div>
        {onDismiss && (
          <div className="error-footer">
            <button className="btn btn-primary" onClick={onDismiss}>Dismiss</button>
          </div>
        )}
      </div>
    </div>
  );
}

import React from 'react';
import {
  Sparkles,
  FlaskConical,
  BookOpen,
  FolderOpen,
  Play,
  ArrowRight,
  ShieldCheck,
  Award,
  Layers,
} from 'lucide-react';
import { colors } from '../../design/tokens/colors';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  onSelectAction: (action: 'NEW_INVESTIGATION' | 'EXPLORE_BENCHMARKS' | 'OPEN_PACKAGE' | 'CONTINUE') => void;
}

export const WelcomeOnboardingModal: React.FC<Props> = ({ isOpen, onClose, onSelectAction }) => {
  if (!isOpen) return null;

  return (
    <div
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.75)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 2200,
        padding: 20,
      }}
    >
      <div
        style={{
          width: '100%',
          maxWidth: 640,
          backgroundColor: colors.canvas,
          borderRadius: 10,
          border: `1px solid ${colors.hairline}`,
          boxShadow: '0 24px 60px rgba(0,0,0,0.6)',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden',
        }}
      >
        {/* Header Banner */}
        <div
          style={{
            padding: '24px 28px',
            backgroundColor: colors.surfaceTile1,
            borderBottom: `1px solid ${colors.hairline}`,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div
              style={{
                width: 36,
                height: 36,
                borderRadius: 8,
                backgroundColor: colors.primary,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#fff',
                fontWeight: 900,
                fontSize: 16,
              }}
            >
              M
            </div>
            <div>
              <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700, color: colors.ink }}>
                Welcome to MECH Research OS
              </h2>
              <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 2 }}>
                A scientific operating system for discovering, intervening on, and validating neural-network mechanisms.
              </div>
            </div>
          </div>
        </div>

        {/* Options Grid */}
        <div style={{ padding: 24, display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 700, color: colors.ink }}>
            What are you investigating today?
          </div>

          {/* Option 1: Start an Investigation */}
          <div
            onClick={() => {
              onSelectAction('NEW_INVESTIGATION');
              onClose();
            }}
            style={{
              padding: 14,
              borderRadius: 8,
              backgroundColor: colors.surfaceTile1,
              border: `1px solid ${colors.hairline}`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              cursor: 'pointer',
              transition: 'background-color 0.15s ease',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.surfaceTile2)}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = colors.surfaceTile1)}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <div
                style={{
                  width: 32,
                  height: 32,
                  borderRadius: 6,
                  backgroundColor: colors.surfaceTile2,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <FlaskConical size={16} color={colors.primary} />
              </div>
              <div>
                <div style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>
                  Start a New Investigation
                </div>
                <div style={{ fontSize: 11, color: colors.bodyMuted }}>
                  Define a research question, load a target model, formulate hypotheses, and execute causal interventions.
                </div>
              </div>
            </div>
            <ArrowRight size={16} color={colors.bodyMuted} />
          </div>

          {/* Option 2: Explore Known Mechanisms */}
          <div
            onClick={() => {
              onSelectAction('EXPLORE_BENCHMARKS');
              onClose();
            }}
            style={{
              padding: 14,
              borderRadius: 8,
              backgroundColor: colors.surfaceTile1,
              border: `1px solid ${colors.hairline}`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              cursor: 'pointer',
              transition: 'background-color 0.15s ease',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.surfaceTile2)}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = colors.surfaceTile1)}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <div
                style={{
                  width: 32,
                  height: 32,
                  borderRadius: 6,
                  backgroundColor: colors.surfaceTile2,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <Award size={16} color={colors.successText} />
              </div>
              <div>
                <div style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>
                  Explore a Known Literature Mechanism
                </div>
                <div style={{ fontSize: 11, color: colors.bodyMuted }}>
                  Run independent ground-truth replications for IOI Circuits, Induction Heads, and Greater-Than Reasoning.
                </div>
              </div>
            </div>
            <ArrowRight size={16} color={colors.bodyMuted} />
          </div>

          {/* Option 3: Open Research Package */}
          <div
            onClick={() => {
              onSelectAction('OPEN_PACKAGE');
              onClose();
            }}
            style={{
              padding: 14,
              borderRadius: 8,
              backgroundColor: colors.surfaceTile1,
              border: `1px solid ${colors.hairline}`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              cursor: 'pointer',
              transition: 'background-color 0.15s ease',
            }}
            onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = colors.surfaceTile2)}
            onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = colors.surfaceTile1)}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
              <div
                style={{
                  width: 32,
                  height: 32,
                  borderRadius: 6,
                  backgroundColor: colors.surfaceTile2,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                }}
              >
                <FolderOpen size={16} color={colors.secondary} />
              </div>
              <div>
                <div style={{ fontSize: 13, fontWeight: 700, color: colors.ink }}>
                  Open Research Package (.mech bundle)
                </div>
                <div style={{ fontSize: 11, color: colors.bodyMuted }}>
                  Import an immutable scientific bundle with verified cryptographic checksums and full provenance logs.
                </div>
              </div>
            </div>
            <ArrowRight size={16} color={colors.bodyMuted} />
          </div>
        </div>

        {/* Footer */}
        <div
          style={{
            padding: '12px 24px',
            backgroundColor: colors.surfaceTile1,
            borderTop: `1px solid ${colors.hairline}`,
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            fontSize: 11,
            color: colors.bodyMuted,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <ShieldCheck size={14} color={colors.successText} />
            <span>Cryptographic Provenance Engine Active</span>
          </div>
          <div>Press <strong style={{ fontFamily: 'monospace' }}>Ctrl+K</strong> anytime for Global Quick Search</div>
        </div>
      </div>
    </div>
  );
};

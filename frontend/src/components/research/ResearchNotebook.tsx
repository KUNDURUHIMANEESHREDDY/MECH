import React, { useState, useMemo } from 'react';
import { useResearchStore } from '../../shared/stores/research';
import { colors } from '../../design/tokens/colors';
import {
  NotebookPen,
  Plus,
  Trash2,
  Search,
  Filter,
  Clock,
  Tag,
  Link,
  ChevronDown,
  ChevronUp,
  Lightbulb,
  FlaskConical,
  Eye,
  ShieldCheck,
  Award,
  MessageSquare,
} from 'lucide-react';

type NoteType = 'USER_NOTE' | 'MODEL_INTERPRETATION' | 'EXPERIMENTAL_OBSERVATION' | 'COMPUTED_EVIDENCE' | 'SCIENTIFIC_CONCLUSION';

interface NotebookNote {
  id: string;
  type: NoteType;
  title: string;
  content: string;
  tags: string[];
  linkedHypothesisId?: string;
  linkedExperimentId?: string;
  createdAt: number;
  updatedAt: number;
}

const NOTE_TYPE_CONFIG: Record<NoteType, { icon: React.ReactNode; color: string; bgColor: string; borderColor: string; label: string }> = {
  USER_NOTE: {
    icon: <MessageSquare size={14} />,
    color: colors.bodyMuted,
    bgColor: colors.surfacePearl,
    borderColor: colors.border,
    label: 'USER NOTE',
  },
  MODEL_INTERPRETATION: {
    icon: <Lightbulb size={14} />,
    color: colors.purpleText,
    bgColor: colors.purpleSoft,
    borderColor: colors.purpleBorder,
    label: 'MODEL INTERPRETATION',
  },
  EXPERIMENTAL_OBSERVATION: {
    icon: <Eye size={14} />,
    color: colors.infoText,
    bgColor: colors.infoSoft,
    borderColor: colors.infoBorder,
    label: 'EXPERIMENTAL OBSERVATION',
  },
  COMPUTED_EVIDENCE: {
    icon: <FlaskConical size={14} />,
    color: colors.successText,
    bgColor: colors.successSoft,
    borderColor: colors.successBorder,
    label: 'COMPUTED EVIDENCE',
  },
  SCIENTIFIC_CONCLUSION: {
    icon: <Award size={14} />,
    color: colors.warningText,
    bgColor: colors.warningSoft,
    borderColor: colors.warningBorder,
    label: 'SCIENTIFIC CONCLUSION',
  },
};

const NOTE_TYPES: NoteType[] = ['USER_NOTE', 'MODEL_INTERPRETATION', 'EXPERIMENTAL_OBSERVATION', 'COMPUTED_EVIDENCE', 'SCIENTIFIC_CONCLUSION'];

export const ResearchNotebook: React.FC = () => {
  const {
    activeInvestigation,
    hypotheses,
    runs,
    loading,
  } = useResearchStore();

  const [notes, setNotes] = useState<NotebookNote[]>([]);
  const [isCreating, setIsCreating] = useState(false);
  const [selectedNoteId, setSelectedNoteId] = useState<string | null>(null);
  const [filterType, setFilterType] = useState<NoteType | 'ALL'>('ALL');
  const [searchQuery, setSearchQuery] = useState('');

  // Form state
  const [formType, setFormType] = useState<NoteType>('USER_NOTE');
  const [formTitle, setFormTitle] = useState('');
  const [formContent, setFormContent] = useState('');
  const [formTags, setFormTags] = useState('');
  const [formLinkedHypothesis, setFormLinkedHypothesis] = useState('');
  const [formLinkedExperiment, setFormLinkedExperiment] = useState('');

  const filteredNotes = useMemo(() => {
    let items = notes;

    if (filterType !== 'ALL') {
      items = items.filter((n) => n.type === filterType);
    }

    if (searchQuery) {
      const q = searchQuery.toLowerCase();
      items = items.filter(
        (n) =>
          n.title.toLowerCase().includes(q) ||
          n.content.toLowerCase().includes(q) ||
          n.tags.some((t) => t.toLowerCase().includes(q))
      );
    }

    return items.sort((a, b) => b.createdAt - a.createdAt);
  }, [notes, filterType, searchQuery]);

  const stats = useMemo(() => ({
    total: notes.length,
    byType: NOTE_TYPES.reduce((acc, type) => {
      acc[type] = notes.filter((n) => n.type === type).length;
      return acc;
    }, {} as Record<NoteType, number>),
  }), [notes]);

  const handleCreate = () => {
    if (!formTitle.trim() && !formContent.trim()) return;

    const newNote: NotebookNote = {
      id: `note_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      type: formType,
      title: formTitle.trim() || 'Untitled Note',
      content: formContent.trim(),
      tags: formTags.split(',').map((t) => t.trim()).filter(Boolean),
      linkedHypothesisId: formLinkedHypothesis || undefined,
      linkedExperimentId: formLinkedExperiment || undefined,
      createdAt: Date.now() / 1000,
      updatedAt: Date.now() / 1000,
    };

    setNotes((prev) => [newNote, ...prev]);
    resetForm();
    setIsCreating(false);
  };

  const handleDelete = (id: string) => {
    setNotes((prev) => prev.filter((n) => n.id !== id));
    if (selectedNoteId === id) {
      setSelectedNoteId(null);
    }
  };

  const resetForm = () => {
    setFormType('USER_NOTE');
    setFormTitle('');
    setFormContent('');
    setFormTags('');
    setFormLinkedHypothesis('');
    setFormLinkedExperiment('');
  };

  const formatTimestamp = (ts: number) => {
    return new Date(ts * 1000).toLocaleString();
  };

  const selectedNote = notes.find((n) => n.id === selectedNoteId);

  return (
    <div style={{ padding: '20px', height: '100%', overflowY: 'auto', boxSizing: 'border-box', backgroundColor: colors.canvas }}>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 16 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: colors.primary, textTransform: 'uppercase', letterSpacing: 0.8 }}>
            Research Notebook
          </div>
          <h2 style={{ margin: '4px 0 0', fontSize: 18, fontWeight: 700, color: colors.ink }}>
            Scientific Notes & Observations
          </h2>
          <div style={{ fontSize: 12, color: colors.bodyMuted, marginTop: 2 }}>
            Record observations, interpretations, and conclusions with strict epistemic type distinctions.
          </div>
        </div>

        <button
          onClick={() => setIsCreating(!isCreating)}
          style={{
            padding: '8px 16px',
            borderRadius: 6,
            border: 'none',
            backgroundColor: colors.primary,
            color: colors.onPrimary,
            fontSize: 12,
            fontWeight: 600,
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}
        >
          <Plus size={14} /> New Note
        </button>
      </div>

      {/* Stats */}
      <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
        <div style={{
          padding: '8px 12px',
          borderRadius: 6,
          backgroundColor: colors.surfaceTile1,
          border: `1px solid ${colors.border}`,
          fontSize: 12,
        }}>
          <b style={{ color: colors.ink }}>{stats.total}</b> <span style={{ color: colors.bodyMuted }}>total notes</span>
        </div>
        {NOTE_TYPES.map((type) => (
          <div
            key={type}
            style={{
              padding: '8px 12px',
              borderRadius: 6,
              backgroundColor: NOTE_TYPE_CONFIG[type].bgColor,
              border: `1px solid ${NOTE_TYPE_CONFIG[type].borderColor}`,
              fontSize: 12,
              display: 'flex',
              alignItems: 'center',
              gap: 4,
            }}
          >
            {NOTE_TYPE_CONFIG[type].icon}
            <b style={{ color: NOTE_TYPE_CONFIG[type].color }}>{stats.byType[type]}</b>
            <span style={{ color: colors.bodyMuted }}>{NOTE_TYPE_CONFIG[type].label}</span>
          </div>
        ))}
      </div>

      {/* Create Form */}
      {isCreating && (
        <div style={{
          border: `1px solid ${colors.primary}`,
          borderRadius: 10,
          padding: 16,
          backgroundColor: colors.surfaceTile1,
          marginBottom: 16,
        }}>
          <div style={{ fontSize: 13, fontWeight: 700, color: colors.ink, marginBottom: 12 }}>New Notebook Entry</div>

          {/* Note Type Selection */}
          <div style={{ marginBottom: 12 }}>
            <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: colors.bodyMuted, marginBottom: 6 }}>
              Note Type (Epistemic Category)
            </label>
            <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
              {NOTE_TYPES.map((type) => {
                const config = NOTE_TYPE_CONFIG[type];
                return (
                  <button
                    key={type}
                    onClick={() => setFormType(type)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                      padding: '6px 12px',
                      borderRadius: 6,
                      border: `1px solid ${formType === type ? config.borderColor : colors.border}`,
                      backgroundColor: formType === type ? config.bgColor : colors.canvas,
                      color: formType === type ? config.color : colors.bodyMuted,
                      fontSize: 11,
                      fontWeight: 600,
                      cursor: 'pointer',
                      transition: 'all 0.15s ease',
                    }}
                  >
                    {config.icon}
                    {config.label}
                  </button>
                );
              })}
            </div>
            <div style={{ marginTop: 6, fontSize: 11, color: colors.bodyMuted }}>
              {formType === 'USER_NOTE' && 'Personal observations, ideas, or questions.'}
              {formType === 'MODEL_INTERPRETATION' && 'AI-generated interpretations of patterns (may not be ground truth).'}
              {formType === 'EXPERIMENTAL_OBSERVATION' && 'Direct measurements from experiment runs.'}
              {formType === 'COMPUTED_EVIDENCE' && 'Quantified causal evidence from interventions.'}
              {formType === 'SCIENTIFIC_CONCLUSION' && 'Researcher conclusions drawn from evidence.'}
            </div>
          </div>

          {/* Title */}
          <div style={{ marginBottom: 10 }}>
            <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: colors.bodyMuted, marginBottom: 4 }}>
              Title
            </label>
            <input
              type="text"
              value={formTitle}
              onChange={(e) => setFormTitle(e.target.value)}
              placeholder="Note title..."
              style={{
                width: '100%',
                padding: '8px 10px',
                borderRadius: 6,
                border: `1px solid ${colors.border}`,
                backgroundColor: colors.canvas,
                color: colors.ink,
                fontSize: 12,
                boxSizing: 'border-box',
                outline: 'none',
              }}
            />
          </div>

          {/* Content */}
          <div style={{ marginBottom: 10 }}>
            <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: colors.bodyMuted, marginBottom: 4 }}>
              Content
            </label>
            <textarea
              value={formContent}
              onChange={(e) => setFormContent(e.target.value)}
              rows={4}
              placeholder="Write your note here..."
              style={{
                width: '100%',
                padding: '8px 10px',
                borderRadius: 6,
                border: `1px solid ${colors.border}`,
                backgroundColor: colors.canvas,
                color: colors.ink,
                fontSize: 12,
                boxSizing: 'border-box',
                outline: 'none',
                resize: 'vertical',
              }}
            />
          </div>

          {/* Tags */}
          <div style={{ marginBottom: 10 }}>
            <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: colors.bodyMuted, marginBottom: 4 }}>
              Tags (comma-separated)
            </label>
            <input
              type="text"
              value={formTags}
              onChange={(e) => setFormTags(e.target.value)}
              placeholder="e.g., IOI, L9H9, causal-evidence"
              style={{
                width: '100%',
                padding: '8px 10px',
                borderRadius: 6,
                border: `1px solid ${colors.border}`,
                backgroundColor: colors.canvas,
                color: colors.ink,
                fontSize: 12,
                boxSizing: 'border-box',
                outline: 'none',
              }}
            />
          </div>

          {/* Linked Entities */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10, marginBottom: 12 }}>
            <div>
              <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: colors.bodyMuted, marginBottom: 4 }}>
                Link to Hypothesis
              </label>
              <select
                value={formLinkedHypothesis}
                onChange={(e) => setFormLinkedHypothesis(e.target.value)}
                style={{
                  width: '100%',
                  padding: '8px 10px',
                  borderRadius: 6,
                  border: `1px solid ${colors.border}`,
                  backgroundColor: colors.canvas,
                  color: colors.ink,
                  fontSize: 12,
                }}
              >
                <option value="">None</option>
                {hypotheses.map((h) => (
                  <option key={h.id} value={h.id}>{h.title}</option>
                ))}
              </select>
            </div>
            <div>
              <label style={{ display: 'block', fontSize: 11, fontWeight: 600, color: colors.bodyMuted, marginBottom: 4 }}>
                Link to Experiment
              </label>
              <select
                value={formLinkedExperiment}
                onChange={(e) => setFormLinkedExperiment(e.target.value)}
                style={{
                  width: '100%',
                  padding: '8px 10px',
                  borderRadius: 6,
                  border: `1px solid ${colors.border}`,
                  backgroundColor: colors.canvas,
                  color: colors.ink,
                  fontSize: 12,
                }}
              >
                <option value="">None</option>
                {runs.map((r) => (
                  <option key={r.id} value={r.id}>Run {r.id.slice(-6)}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Actions */}
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              onClick={handleCreate}
              disabled={!formTitle.trim() && !formContent.trim()}
              style={{
                padding: '8px 16px',
                borderRadius: 6,
                border: 'none',
                backgroundColor: colors.primary,
                color: colors.onPrimary,
                fontSize: 12,
                fontWeight: 600,
                cursor: !formTitle.trim() && !formContent.trim() ? 'default' : 'pointer',
                opacity: !formTitle.trim() && !formContent.trim() ? 0.5 : 1,
              }}
            >
              Save Note
            </button>
            <button
              onClick={() => { setIsCreating(false); resetForm(); }}
              style={{
                padding: '8px 16px',
                borderRadius: 6,
                border: `1px solid ${colors.border}`,
                backgroundColor: colors.canvas,
                color: colors.bodyMuted,
                fontSize: 12,
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Search and Filter */}
      <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
        <div style={{ flex: 1, position: 'relative' }}>
          <Search size={14} style={{ position: 'absolute', left: 10, top: '50%', transform: 'translateY(-50%)', color: colors.bodyMuted }} />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search notes..."
            style={{
              width: '100%',
              padding: '8px 12px 8px 32px',
              borderRadius: 6,
              border: `1px solid ${colors.border}`,
              backgroundColor: colors.canvas,
              color: colors.ink,
              fontSize: 12,
              boxSizing: 'border-box',
              outline: 'none',
            }}
          />
        </div>

        <div style={{ display: 'flex', gap: 4 }}>
          <button
            onClick={() => setFilterType('ALL')}
            style={{
              padding: '6px 12px',
              borderRadius: 6,
              border: `1px solid ${filterType === 'ALL' ? colors.primary : colors.border}`,
              backgroundColor: filterType === 'ALL' ? colors.accentSoft : colors.canvas,
              color: filterType === 'ALL' ? colors.primary : colors.bodyMuted,
              fontSize: 11,
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            ALL
          </button>
          {NOTE_TYPES.map((type) => (
            <button
              key={type}
              onClick={() => setFilterType(type)}
              style={{
                padding: '6px 12px',
                borderRadius: 6,
                border: `1px solid ${filterType === type ? NOTE_TYPE_CONFIG[type].borderColor : colors.border}`,
                backgroundColor: filterType === type ? NOTE_TYPE_CONFIG[type].bgColor : colors.canvas,
                color: filterType === type ? NOTE_TYPE_CONFIG[type].color : colors.bodyMuted,
                fontSize: 11,
                fontWeight: 600,
                cursor: 'pointer',
              }}
            >
              {type.split('_').slice(0, 2).join(' ')}
            </button>
          ))}
        </div>
      </div>

      {/* Notes List */}
      <div style={{ display: 'grid', gridTemplateColumns: selectedNote ? '1fr 1fr' : '1fr', gap: 16 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {filteredNotes.length === 0 ? (
            <div style={{
              padding: 40,
              textAlign: 'center',
              color: colors.bodyMuted,
              backgroundColor: colors.surfacePearl,
              borderRadius: 10,
            }}>
              <NotebookPen size={32} style={{ opacity: 0.3, marginBottom: 8 }} />
              <div style={{ fontSize: 13, fontWeight: 600, color: colors.ink }}>No Notes Found</div>
              <div style={{ fontSize: 12, marginTop: 4 }}>
                {searchQuery || filterType !== 'ALL'
                  ? 'No notes match your search criteria.'
                  : 'Create your first note to start documenting your research.'}
              </div>
            </div>
          ) : (
            filteredNotes.map((note) => {
              const config = NOTE_TYPE_CONFIG[note.type];
              const isSelected = selectedNoteId === note.id;

              return (
                <div
                  key={note.id}
                  onClick={() => setSelectedNoteId(isSelected ? null : note.id)}
                  style={{
                    border: `1px solid ${isSelected ? config.borderColor : colors.border}`,
                    borderRadius: 8,
                    padding: 12,
                    backgroundColor: isSelected ? config.bgColor : colors.surfaceTile1,
                    cursor: 'pointer',
                    transition: 'all 0.15s ease',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 6 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <div style={{ color: config.color }}>{config.icon}</div>
                      <div>
                        <div style={{ fontSize: 13, fontWeight: 600, color: colors.ink }}>{note.title}</div>
                        <div style={{ fontSize: 10, color: config.color, fontWeight: 700, marginTop: 2 }}>{config.label}</div>
                      </div>
                    </div>

                    <button
                      onClick={(e) => { e.stopPropagation(); handleDelete(note.id); }}
                      style={{
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        color: colors.dangerText,
                        padding: 4,
                      }}
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>

                  <div style={{ fontSize: 12, color: colors.body, lineHeight: 1.5, marginBottom: 8 }}>
                    {note.content.slice(0, 150)}{note.content.length > 150 ? '...' : ''}
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                    {note.tags.map((tag) => (
                      <span
                        key={tag}
                        style={{
                          fontSize: 10,
                          padding: '2px 6px',
                          borderRadius: 4,
                          backgroundColor: colors.surfacePearl,
                          color: colors.bodyMuted,
                        }}
                      >
                        <Tag size={8} style={{ marginRight: 2 }} />{tag}
                      </span>
                    ))}
                    {(note.linkedHypothesisId || note.linkedExperimentId) && (
                      <span style={{ fontSize: 10, color: colors.bodyMuted }}>
                        <Link size={10} style={{ marginRight: 4 }} />
                        Linked
                      </span>
                    )}
                    <span style={{ fontSize: 10, color: colors.bodyMuted, marginLeft: 'auto' }}>
                      {formatTimestamp(note.createdAt)}
                    </span>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Note Detail View */}
        {selectedNote && (
          <div style={{
            border: `1px solid ${NOTE_TYPE_CONFIG[selectedNote.type].borderColor}`,
            borderRadius: 10,
            padding: 16,
            backgroundColor: colors.surfaceTile1,
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <span style={{ fontSize: 12, fontWeight: 700, color: colors.ink, textTransform: 'uppercase' }}>
                Note Detail
              </span>
              <button
                onClick={() => setSelectedNoteId(null)}
                style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: 16, color: colors.bodyMuted }}
              >
                ×
              </button>
            </div>

            {/* Type Badge */}
            <div style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              padding: '6px 12px',
              borderRadius: 6,
              backgroundColor: NOTE_TYPE_CONFIG[selectedNote.type].bgColor,
              border: `1px solid ${NOTE_TYPE_CONFIG[selectedNote.type].borderColor}`,
              marginBottom: 12,
            }}>
              {NOTE_TYPE_CONFIG[selectedNote.type].icon}
              <span style={{ fontSize: 11, fontWeight: 700, color: NOTE_TYPE_CONFIG[selectedNote.type].color }}>
                {NOTE_TYPE_CONFIG[selectedNote.type].label}
              </span>
            </div>

            <h3 style={{ margin: '0 0 8px', fontSize: 16, fontWeight: 700, color: colors.ink }}>
              {selectedNote.title}
            </h3>

            <div style={{
              fontSize: 12,
              color: colors.body,
              lineHeight: 1.6,
              padding: 12,
              borderRadius: 6,
              backgroundColor: colors.surfacePearl,
              border: `1px solid ${colors.border}`,
              marginBottom: 12,
              whiteSpace: 'pre-wrap',
            }}>
              {selectedNote.content}
            </div>

            {/* Metadata */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 11, color: colors.bodyMuted }}>
              <div><b>Created:</b> {formatTimestamp(selectedNote.createdAt)}</div>
              <div><b>Updated:</b> {formatTimestamp(selectedNote.updatedAt)}</div>
              {selectedNote.tags.length > 0 && (
                <div><b>Tags:</b> {selectedNote.tags.join(', ')}</div>
              )}
              {selectedNote.linkedHypothesisId && (
                <div><b>Linked Hypothesis:</b> {hypotheses.find((h) => h.id === selectedNote.linkedHypothesisId)?.title || selectedNote.linkedHypothesisId}</div>
              )}
              {selectedNote.linkedExperimentId && (
                <div><b>Linked Experiment:</b> Run {selectedNote.linkedExperimentId.slice(-6)}</div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

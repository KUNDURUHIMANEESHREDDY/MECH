<template>
  <main class="lab-notebook-surface">
    <header class="surface-header">
      <div class="header-copy">
        <p class="surface-kicker">General / Local notes</p>
        <h2 class="surface-title">Lab Notebook</h2>
        <p class="surface-description">Keep experiment notes and logs in this browser. They are local user notes and are not synced to the MECH backend.</p>
      </div>
      <div class="header-actions">
        <span class="surface-status" :class="{ 'is-connected': storageAvailable }" role="status" aria-live="polite">
          <span class="status-dot" aria-hidden="true" />
          {{ storageAvailable ? 'Local storage available' : 'Local storage unavailable' }}
        </span>
        <button class="primary-button" type="button" :disabled="!storageAvailable" @click="createNote">New lab note</button>
      </div>
    </header>

    <div v-if="storageError" class="surface-notice notice-error" role="alert">
      <strong>Local lab notebook storage is unavailable</strong>
      <p>{{ storageError }}</p>
    </div>

    <div v-if="actionMessage" class="surface-notice notice-success" role="status" aria-live="polite">
      <p>{{ actionMessage }}</p>
    </div>

    <div class="lab-layout">
      <section class="surface-panel notes-panel" aria-labelledby="lab-notes-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">This browser only</p>
            <h3 id="lab-notes-title">Lab notes</h3>
            <p class="panel-description">Local notes stored under a namespaced browser key.</p>
          </div>
          <span class="source-badge">LOCAL ONLY</span>
        </header>

        <div class="search-field">
          <label for="lab-note-search">Find a note</label>
          <input
            id="lab-note-search"
            v-model="searchQuery"
            type="search"
            autocomplete="off"
            placeholder="Search note titles or text"
            :disabled="!notes.length"
          >
        </div>

        <div v-if="isLoading" class="panel-message" role="status" aria-live="polite">
          <span class="loading-mark" aria-hidden="true" />
          Loading local lab notes…
        </div>

        <ul v-else-if="filteredNotes.length" class="note-list" aria-label="Local lab notes">
          <li v-for="note in filteredNotes" :key="note.id" class="note-item">
            <button
              class="note-select"
              type="button"
              :class="{ 'is-selected': selectedNoteId === note.id }"
              :aria-pressed="selectedNoteId === note.id"
              @click="selectNote(note.id)"
            >
              <span class="note-heading">
                <span class="note-mark" aria-hidden="true">L</span>
                <span class="note-title">{{ note.title }}</span>
                <span class="local-badge">LOCAL</span>
              </span>
              <span class="note-preview">{{ note.body || 'Empty note' }}</span>
              <span class="note-meta">Updated {{ formatTimestamp(note.updatedAt) }}</span>
            </button>
            <button class="icon-button" type="button" :aria-label="`Delete ${note.title}`" title="Delete lab note" @click="deleteNote(note)">
              <span aria-hidden="true">×</span>
            </button>
          </li>
        </ul>

        <p v-else-if="storageError" class="panel-message panel-message-error">
          Lab notes could not be read from this browser.
        </p>
        <p v-else-if="notes.length" class="panel-message">
          No local lab notes match “{{ searchQuery }}”.
        </p>
        <p v-else class="panel-message">
          No local lab notes have been created in this browser. Choose “New lab note” to start one.
        </p>
      </section>

      <section class="surface-panel editor-panel" aria-labelledby="lab-editor-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Local editor</p>
            <h3 id="lab-editor-title">{{ selectedNoteId ? 'Edit lab note' : 'Select a lab note' }}</h3>
            <p class="panel-description">Nothing is sent to a backend from this window.</p>
          </div>
          <span v-if="selectedNoteId" class="source-badge">LOCAL ONLY</span>
        </header>

        <form v-if="selectedNoteId" class="editor-form" @submit.prevent="saveNote">
          <div class="field">
            <label for="lab-note-title">Title <span class="required-mark" aria-hidden="true">*</span></label>
            <input id="lab-note-title" v-model="draftTitle" type="text" autocomplete="off" required>
          </div>
          <div class="field">
            <label for="lab-note-body">Log or note</label>
            <textarea id="lab-note-body" v-model="draftBody" rows="16" placeholder="Record an observation, setup detail, or follow-up…" />
          </div>
          <p v-if="formError" class="form-message" role="alert">{{ formError }}</p>
          <div class="editor-actions">
            <span class="word-count" role="status" aria-live="polite">{{ wordCount }} words</span>
            <div class="action-buttons">
              <button class="quiet-button" type="button" @click="selectFirst">Select another</button>
              <button class="primary-button" type="submit">Save lab note</button>
            </div>
          </div>
        </form>

        <div v-else class="editor-empty">
          <span class="empty-mark" aria-hidden="true">L</span>
          <h3>No lab note selected</h3>
          <p>Create a local note or choose one from the list to begin editing.</p>
          <button class="quiet-button" type="button" :disabled="!storageAvailable" @click="createNote">Create a lab note</button>
        </div>
      </section>
    </div>

    <section class="surface-panel boundary-panel" aria-labelledby="lab-boundary-title">
      <p class="panel-kicker">Storage boundary</p>
      <h3 id="lab-boundary-title">Local-only experiment notes</h3>
      <div class="boundary-columns">
        <p><strong>Stored here:</strong> the current browser’s local storage, under <code>mech.local.lab-notes.v1</code>.</p>
        <p><strong>Not included:</strong> backend sync, shared workspaces, automatic export, or claims of scientific provenance.</p>
      </div>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';

interface LocalLabNote {
  id: string;
  title: string;
  body: string;
  createdAt: string;
  updatedAt: string;
}

const STORAGE_KEY = 'mech.local.lab-notes.v1';
const notes = ref<LocalLabNote[]>([]);
const selectedNoteId = ref('');
const draftTitle = ref('');
const draftBody = ref('');
const searchQuery = ref('');
const isLoading = ref(true);
const storageAvailable = ref(true);
const formError = ref('');
const storageError = ref('');
const actionMessage = ref('');

const selectedNote = computed(() => notes.value.find((note) => note.id === selectedNoteId.value) ?? null);
const filteredNotes = computed(() => {
  const query = searchQuery.value.trim().toLocaleLowerCase();
  if (!query) return notes.value;
  return notes.value.filter((note) => `${note.title} ${note.body}`.toLocaleLowerCase().includes(query));
});
const wordCount = computed(() => {
  const text = draftBody.value.trim();
  return text ? text.split(/\s+/).length : 0;
});

onMounted(() => {
  loadNotes();
});

function loadNotes() {
  isLoading.value = true;
  if (typeof localStorage === 'undefined') {
    storageAvailable.value = false;
    storageError.value = 'This runtime does not expose browser local storage.';
    isLoading.value = false;
    return;
  }
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      notes.value = [];
      isLoading.value = false;
      return;
    }
    const parsed: unknown = JSON.parse(stored);
    notes.value = Array.isArray(parsed)
      ? parsed.map((value, index) => normalizeNote(value, index)).filter((value): value is LocalLabNote => value !== null)
      : [];
  } catch {
    notes.value = [];
    storageError.value = 'The saved local lab notes could not be read; no sample notes were loaded.';
  } finally {
    isLoading.value = false;
  }
}

function persistNotes(): boolean {
  if (typeof localStorage === 'undefined') {
    storageAvailable.value = false;
    storageError.value = 'This runtime does not expose browser local storage.';
    return false;
  }
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(notes.value));
    storageAvailable.value = true;
    return true;
  } catch {
    storageError.value = 'The browser could not save this local lab note.';
    return false;
  }
}

function createNote() {
  if (!storageAvailable.value) return;
  const now = new Date().toISOString();
  const note: LocalLabNote = {
    id: makeId(),
    title: 'Untitled lab note',
    body: '',
    createdAt: now,
    updatedAt: now,
  };
  notes.value = [note, ...notes.value];
  if (!persistNotes()) {
    notes.value = notes.value.filter((item) => item.id !== note.id);
    return;
  }
  selectNote(note.id);
  actionMessage.value = 'New local lab note created.';
}

function selectNote(id: string) {
  const note = notes.value.find((item) => item.id === id);
  if (!note) return;
  selectedNoteId.value = id;
  draftTitle.value = note.title;
  draftBody.value = note.body;
  formError.value = '';
}

function selectFirst() {
  selectedNoteId.value = '';
  draftTitle.value = '';
  draftBody.value = '';
  formError.value = '';
}

function saveNote() {
  formError.value = '';
  actionMessage.value = '';
  const title = draftTitle.value.trim();
  if (!title) {
    formError.value = 'Enter a title before saving this local lab note.';
    return;
  }
  if (!selectedNoteId.value) {
    formError.value = 'Select a note before saving.';
    return;
  }
  const now = new Date().toISOString();
  notes.value = notes.value.map((note) => note.id === selectedNoteId.value
    ? { ...note, title, body: draftBody.value, updatedAt: now }
    : note);
  if (!persistNotes()) return;
  actionMessage.value = 'Lab note saved in this browser only.';
}

function deleteNote(note: LocalLabNote) {
  if (typeof window !== 'undefined' && !window.confirm(`Delete the local lab note “${note.title}”?`)) return;
  notes.value = notes.value.filter((item) => item.id !== note.id);
  if (!persistNotes()) {
    notes.value = [note, ...notes.value];
    return;
  }
  if (selectedNoteId.value === note.id) selectFirst();
  actionMessage.value = `Deleted ${note.title} from this browser.`;
}

function normalizeNote(value: unknown, index: number): LocalLabNote | null {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  const title = typeof record.title === 'string' ? record.title.trim() : '';
  const body = typeof record.body === 'string' ? record.body : typeof record.content === 'string' ? record.content : '';
  if (!title && !body) return null;
  const now = new Date().toISOString();
  return {
    id: typeof record.id === 'string' && record.id ? record.id : `local-lab-note-${index}`,
    title: title || `Untitled lab note ${index + 1}`,
    body,
    createdAt: typeof record.createdAt === 'string' ? record.createdAt : now,
    updatedAt: typeof record.updatedAt === 'string' ? record.updatedAt : now,
  };
}

function makeId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') return `local-lab-note-${crypto.randomUUID()}`;
  return `local-lab-note-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function formatTimestamp(value: string): string {
  if (!value) return 'an unspecified time';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}
</script>

<style scoped>
.lab-notebook-surface {
  min-height: 100%;
  padding: clamp(18px, 3vw, 30px);
  background: var(--surface);
  color: var(--text);
  color-scheme: light;
}

.surface-header,
.header-actions,
.panel-heading,
.note-heading,
.editor-actions,
.action-buttons,
.boundary-columns {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
}

.surface-header {
  align-items: center;
  margin-bottom: 20px;
}

.header-copy {
  min-width: 0;
}

.surface-kicker,
.panel-kicker {
  margin: 0 0 5px;
  color: var(--text-muted);
  font: 700 10px/1.2 var(--font-mono);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.surface-title {
  margin: 0;
  color: var(--text);
  font-size: clamp(22px, 3vw, 30px);
  font-weight: 750;
  letter-spacing: -0.035em;
  line-height: 1.1;
}

.surface-description,
.panel-description {
  margin: 7px 0 0;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.header-actions {
  flex: 0 0 auto;
  align-items: center;
  flex-wrap: wrap;
  justify-content: flex-end;
}

.surface-status {
  display: inline-flex;
  min-height: 28px;
  align-items: center;
  gap: 7px;
  padding: 0 9px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-2);
  color: var(--text-muted);
  font: 650 10px/1 var(--font-mono);
  white-space: nowrap;
}

.surface-status.is-connected {
  color: var(--text-dim);
}

.status-dot {
  width: 7px;
  height: 7px;
  flex: 0 0 auto;
  border-radius: 50%;
  background: var(--text-muted);
}

.surface-status.is-connected .status-dot {
  background: var(--success);
}

.quiet-button,
.primary-button {
  min-height: 30px;
  padding: 0 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  cursor: pointer;
  font: 700 11px/1 var(--font);
}

.quiet-button {
  background: var(--surface);
  color: var(--text-muted);
}

.quiet-button:hover:not(:disabled) {
  border-color: var(--border-light);
  color: var(--text);
}

.primary-button {
  border-color: var(--primary);
  background: var(--primary);
  color: #ffffff;
}

.primary-button:hover:not(:disabled) {
  border-color: var(--primary-focus);
  background: var(--primary-focus);
}

.quiet-button:disabled,
.primary-button:disabled,
.icon-button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.surface-notice {
  margin-bottom: 14px;
  padding: 12px 14px;
  border: 1px solid var(--border);
  border-left: 3px solid var(--border-light);
  border-radius: var(--radius);
  background: var(--surface-2);
  color: var(--text-dim);
  font-size: 12px;
  line-height: 1.5;
}

.surface-notice strong {
  display: block;
  margin-bottom: 3px;
  color: var(--text);
}

.surface-notice p {
  margin: 0;
}

.notice-error {
  border-left-color: var(--danger);
}

.notice-success {
  border-left-color: var(--success);
}

.lab-layout {
  display: grid;
  grid-template-columns: minmax(240px, 0.72fr) minmax(0, 1.28fr);
  gap: 14px;
  align-items: start;
}

.surface-panel {
  min-width: 0;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}

.notes-panel,
.editor-panel,
.boundary-panel {
  padding: clamp(16px, 2.2vw, 22px);
}

.panel-heading {
  align-items: flex-start;
  margin-bottom: 18px;
}

.panel-heading h3,
.boundary-panel h3,
.editor-empty h3 {
  margin: 0;
  color: var(--text);
  font-size: 17px;
  font-weight: 720;
  letter-spacing: -0.02em;
}

.source-badge,
.local-badge {
  display: inline-flex;
  min-height: 21px;
  align-items: center;
  padding: 0 7px;
  border: 1px solid var(--border);
  border-radius: 999px;
  background: var(--surface-2);
  color: var(--text-muted);
  font: 700 9px/1 var(--font-mono);
  letter-spacing: 0.03em;
  white-space: nowrap;
}

.local-badge {
  margin-left: auto;
  border-color: var(--border-light);
  color: var(--text-dim);
}

.search-field {
  margin-bottom: 14px;
}

.search-field label,
.field label {
  display: block;
  margin-bottom: 6px;
  color: var(--text-dim);
  font-size: 12px;
  font-weight: 700;
}

.search-field input,
.field input,
.field textarea {
  display: block;
  width: 100%;
  border: 1px solid var(--border);
  border-radius: 6px;
  outline: 0;
  background: var(--surface);
  color: var(--text);
  font: 12px/1.45 var(--font);
}

.search-field input,
.field input {
  min-height: 38px;
  padding: 0 10px;
}

.field textarea {
  min-height: 260px;
  padding: 10px;
  resize: vertical;
}

.search-field input:focus,
.field input:focus,
.field textarea:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 2px var(--accent-soft);
}

.search-field input:focus-visible,
.field input:focus-visible,
.field textarea:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
}

.search-field input:disabled,
.field input:disabled,
.field textarea:disabled {
  cursor: not-allowed;
  opacity: 0.55;
}

.note-list {
  display: grid;
  gap: 7px;
  max-height: 560px;
  margin: 0;
  padding: 0 2px 0 0;
  overflow-y: auto;
  list-style: none;
}

.note-item {
  display: flex;
  min-width: 0;
  gap: 5px;
}

.note-select {
  display: block;
  min-width: 0;
  flex: 1;
  padding: 10px 11px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
  color: var(--text);
  cursor: pointer;
  text-align: left;
}

.note-select:hover,
.note-select.is-selected {
  border-color: var(--primary);
  background: var(--accent-soft);
}

.note-heading {
  align-items: center;
  justify-content: flex-start;
  gap: 7px;
}

.note-mark,
.empty-mark {
  display: inline-grid;
  width: 25px;
  height: 25px;
  flex: 0 0 auto;
  place-items: center;
  border: 1px solid var(--border);
  border-radius: 5px;
  background: var(--surface);
  color: var(--primary);
  font: 750 9px/1 var(--font-mono);
}

.note-title {
  min-width: 0;
  overflow: hidden;
  color: var(--text);
  font-size: 12px;
  font-weight: 700;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.note-preview {
  display: -webkit-box;
  margin: 8px 0 0;
  overflow: hidden;
  color: var(--text-dim);
  font-size: 11px;
  line-height: 1.4;
  -webkit-box-orient: vertical;
  -webkit-line-clamp: 2;
}

.note-meta {
  display: block;
  margin-top: 7px;
  color: var(--text-muted);
  font-size: 10px;
}

.icon-button {
  width: 30px;
  min-height: 30px;
  padding: 0;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface);
  color: var(--text-muted);
  cursor: pointer;
  font-size: 16px;
}

.icon-button:hover {
  border-color: var(--danger);
  color: var(--danger);
}

.panel-message {
  display: flex;
  min-height: 86px;
  align-items: center;
  gap: 9px;
  margin: 0;
  padding: 12px;
  border: 1px dashed var(--border-light);
  border-radius: 6px;
  color: var(--text-muted);
  font-size: 12px;
  line-height: 1.5;
}

.panel-message-error {
  border-style: solid;
  color: var(--danger);
}

.loading-mark {
  width: 12px;
  height: 12px;
  flex: 0 0 auto;
  border: 1px solid var(--border-light);
  border-top-color: var(--text);
  border-radius: 50%;
  animation: lab-notebook-spin 800ms linear infinite;
}

@keyframes lab-notebook-spin {
  to { transform: rotate(360deg); }
}

.editor-form {
  display: grid;
  gap: 14px;
}

.field {
  min-width: 0;
}

.required-mark {
  color: var(--danger);
}

.form-message {
  margin: 0;
  color: var(--danger);
  font-size: 11px;
  line-height: 1.45;
}

.editor-actions {
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
}

.word-count {
  color: var(--text-muted);
  font-size: 10px;
}

.action-buttons {
  align-items: center;
  flex-wrap: wrap;
}

.editor-empty {
  display: flex;
  min-height: 300px;
  align-items: center;
  flex-direction: column;
  justify-content: center;
  gap: 9px;
  color: var(--text-muted);
  text-align: center;
}

.editor-empty .empty-mark {
  width: 42px;
  height: 42px;
  margin-bottom: 3px;
  font-size: 14px;
}

.editor-empty p {
  max-width: 310px;
  margin: 0 0 5px;
  font-size: 12px;
  line-height: 1.5;
}

.boundary-panel {
  margin-top: 14px;
}

.boundary-columns {
  align-items: stretch;
  margin-top: 14px;
}

.boundary-columns p {
  flex: 1 1 0;
  margin: 0;
  padding: 11px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.55;
}

.boundary-columns strong {
  color: var(--text-dim);
}

@media (max-width: 820px) {
  .lab-layout {
    grid-template-columns: minmax(0, 1fr);
  }

  .note-list {
    max-height: none;
  }
}

@media (max-width: 580px) {
  .surface-header,
  .header-actions,
  .panel-heading,
  .editor-actions,
  .boundary-columns {
    align-items: flex-start;
    flex-direction: column;
  }

  .header-actions {
    width: 100%;
  }

  .header-actions .primary-button,
  .header-actions .surface-status {
    align-self: flex-start;
  }

  .action-buttons {
    width: 100%;
  }

  .action-buttons button {
    flex: 1 1 0;
  }
}

@media (prefers-reduced-motion: reduce) {
  .loading-mark {
    animation: none;
  }
}
</style>

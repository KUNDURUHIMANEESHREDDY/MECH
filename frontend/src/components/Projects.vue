<template>
  <main class="projects-surface">
    <header class="surface-header">
      <div class="header-copy">
        <p class="surface-kicker">Workspace / Local</p>
        <h2 class="surface-title">Projects</h2>
        <p class="surface-description">Inspect application workspaces and keep browser-only projects separate.</p>
      </div>
      <span
        class="surface-status"
        :class="{ 'is-connected': bridgeAvailable || apiActive }"
        role="status"
        aria-live="polite"
      >
        <span class="status-dot" aria-hidden="true" />
        {{ bridgeAvailable ? 'Bridge detected' : apiActive ? 'Backend API' : 'Bridge unavailable' }}
      </span>
    </header>

    <div v-if="!bridgeAvailable && !apiActive" class="surface-notice notice-warning" role="status">
      <strong>Local application bridge unavailable</strong>
      <p>
        Open MECH in the Electron desktop app to load application project records. Browser-only entries below are
        user-created and remain in this browser's local storage; they are not synced to the desktop database.
      </p>
    </div>

    <div v-if="bridgeError" class="surface-notice notice-error" role="alert">
      <strong>Application projects could not be loaded</strong>
      <p>{{ bridgeError }}</p>
    </div>

    <div v-if="actionMessage" class="surface-notice notice-success" role="status" aria-live="polite">
      <p>{{ actionMessage }}</p>
    </div>

    <div class="projects-layout">
      <section class="surface-panel records-panel" aria-labelledby="application-projects-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Application database</p>
            <h3 id="application-projects-title">Application projects</h3>
            <p class="panel-description">{{ bridgeAvailable ? 'Records returned by the active desktop bridge.' : 'Records returned by the backend database.' }}</p>
          </div>
          <button
            v-if="bridgeCanList || apiActive"
            class="quiet-button"
            type="button"
            :disabled="isLoading"
            @click="refreshProjects"
          >
            {{ isLoading ? 'Refreshing…' : 'Refresh' }}
          </button>
        </header>

        <div v-if="isLoading" class="panel-message" role="status" aria-live="polite">
          <span class="loading-mark" aria-hidden="true" />
          Loading application projects…
        </div>

        <ul v-else-if="applicationProjects.length" class="record-list" aria-label="Application projects">
          <li v-for="project in applicationProjects" :key="project.id" class="record-item">
            <div class="record-main">
              <div class="record-title-line">
                <h4>{{ project.name }}</h4>
                <span class="record-badge">{{ project.source === 'api' ? 'BACKEND API' : 'APPLICATION' }}</span>
              </div>
              <p v-if="project.path" class="record-path" :title="project.path">
                <span aria-hidden="true">↳</span> {{ project.path }}
              </p>
              <p v-else class="record-path record-muted">No path supplied</p>
              <p class="record-meta">
                Updated <time v-if="project.updatedAt" :datetime="project.updatedAt">{{ formatTimestamp(project.updatedAt) }}</time>
                <span v-else>at an unspecified time</span>
              </p>
            </div>
            <div class="record-actions">
              <button
                v-if="project.path && bridgeCanReveal"
                class="icon-button"
                type="button"
                :aria-label="`Show ${project.name} in folder`"
                title="Show in folder"
                @click="showInFolder(project)"
              >
                <span aria-hidden="true">↗</span>
              </button>
              <button
                v-if="bridgeCanRemove"
                class="remove-button"
                type="button"
                :aria-label="`Remove application project ${project.name}`"
                @click="removeApplicationProject(project)"
              >
                Remove
              </button>
            </div>
          </li>
        </ul>

        <p v-else-if="bridgeError" class="panel-message panel-message-error">
          Application project records are unavailable for this request.
        </p>
        <p v-else-if="(bridgeAvailable && bridgeCanList) || apiActive" class="panel-message">
          No application projects have been recorded.
        </p>
        <p v-else-if="bridgeAvailable" class="panel-message">
          The detected bridge does not expose a project reader. No application records are shown.
        </p>
        <p v-else class="panel-message">
          Application project records require the local application bridge.
        </p>
      </section>

      <section class="surface-panel entry-panel" aria-labelledby="add-project-title">
        <header class="panel-heading">
          <div>
            <p class="panel-kicker">Create</p>
            <h3 id="add-project-title">Add project</h3>
            <p class="panel-description">The destination is shown before anything is saved.</p>
          </div>
        </header>

        <form class="entry-form" @submit.prevent="addProject">
          <div class="field">
            <label for="project-name">Project name</label>
            <input
              id="project-name"
              v-model="projectName"
              type="text"
              autocomplete="off"
              placeholder="e.g. Mechanistic study"
              required
            >
          </div>
          <div class="field">
            <label for="project-path">Project path <span class="optional-label">optional</span></label>
            <input
              id="project-path"
              v-model="projectPath"
              type="text"
              autocomplete="off"
              placeholder="C:\\research\\study"
            >
          </div>

          <div class="destination-note" role="note">
            <span class="destination-badge">{{ bridgeCanAdd ? 'APPLICATION BRIDGE' : apiActive ? 'BACKEND API' : 'LOCAL BROWSER' }}</span>
            <span v-if="bridgeCanAdd">This entry will be sent to the active application bridge.</span>
            <span v-else-if="apiActive">This entry will be saved to the backend database.</span>
            <span v-else>This entry will be stored only in this browser until a bridge is available.</span>
          </div>

          <p v-if="formError" class="form-message form-message-error" role="alert">{{ formError }}</p>
          <div class="form-actions">
            <button class="primary-button" type="submit" :disabled="isSaving">
              {{ isSaving ? 'Adding…' : 'Add project' }}
            </button>
            <button
              v-if="bridgeCanChoose"
              class="secondary-button"
              type="button"
              :disabled="isChoosing"
              @click="chooseProject"
            >
              {{ isChoosing ? 'Choosing…' : 'Choose project folder' }}
            </button>
          </div>
        </form>
      </section>
    </div>

    <section class="surface-panel local-panel" aria-labelledby="local-projects-title">
      <header class="panel-heading">
        <div>
          <p class="panel-kicker">This browser only</p>
          <h3 id="local-projects-title">Local projects</h3>
          <p class="panel-description">
            User-created records stored in this browser's local storage. They are not application-database records and do not sync.
          </p>
        </div>
        <span class="local-badge">LOCAL</span>
      </header>

      <div data-testid="projects-list" class="local-list">
        <ul v-if="localProjects.length" class="record-list" aria-label="Local browser projects">
          <li v-for="project in localProjects" :key="project.id" class="record-item">
            <div class="record-main">
              <div class="record-title-line">
                <h4>{{ project.name }}</h4>
                <span class="record-badge record-badge-local">LOCAL BROWSER</span>
              </div>
              <p v-if="project.path" class="record-path" :title="project.path">
                <span aria-hidden="true">↳</span> {{ project.path }}
              </p>
              <p v-else class="record-path record-muted">No path supplied</p>
              <p class="record-meta">
                Created <time v-if="project.createdAt" :datetime="project.createdAt">{{ formatTimestamp(project.createdAt) }}</time>
                <span v-else>at an unspecified time</span>
              </p>
            </div>
            <button
              class="remove-button"
              type="button"
              :aria-label="`Remove local project ${project.name}`"
              @click="removeLocalProject(project)"
            >
              Remove
            </button>
          </li>
        </ul>
        <p v-else class="panel-message">No local projects have been created in this browser.</p>
      </div>
    </section>
  </main>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue';
import { api } from '../services/api';

type UnknownRecord = Record<string, unknown>;
type Bridge = Record<string, unknown>;
type BridgeMethod = (...args: unknown[]) => unknown;
type ProjectSource = 'application' | 'api' | 'local';

interface ProjectRecord {
  id: string;
  name: string;
  path: string;
  createdAt: string;
  updatedAt: string;
  source: ProjectSource;
}

const LOCAL_PROJECTS_KEY = 'mech.local.projects.v1';
const bridge = ref<Bridge | null>(resolveBridge());
const applicationProjects = ref<ProjectRecord[]>([]);
const localProjects = ref<ProjectRecord[]>([]);
const apiActive = ref(false);
const projectName = ref('');
const projectPath = ref('');
const isLoading = ref(false);
const isSaving = ref(false);
const isChoosing = ref(false);
const bridgeError = ref('');
const formError = ref('');
const actionMessage = ref('');

const bridgeAvailable = computed(() => Boolean(bridge.value));
const bridgeCanList = computed(() => Boolean(bridge.value && (hasMethod(bridge.value, 'listProjects') || hasMethod(bridge.value, 'listRecentProjects'))));
const bridgeCanAdd = computed(() => Boolean(bridge.value && (hasMethod(bridge.value, 'addProject') || hasMethod(bridge.value, 'addRecentProject'))));
const bridgeCanRemove = computed(() => hasMethod(bridge.value, 'removeProject'));
const bridgeCanReveal = computed(() => hasMethod(bridge.value, 'showInFolder'));
const bridgeCanChoose = computed(() => hasMethod(bridge.value, 'chooseProject'));

onMounted(() => {
  loadLocalProjects();
  void loadApplicationProjects();
});

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function resolveBridge(): Bridge | null {
  if (typeof window !== 'undefined') {
    const windowApi = (window as Window & { appApi?: unknown }).appApi;
    if (isRecord(windowApi)) return windowApi;
  }
  return null;
}

function hasMethod(value: Bridge | null, name: string): value is Bridge & Record<string, BridgeMethod> {
  return Boolean(value && typeof value[name] === 'function');
}

function getMethod(value: Bridge | null, name: string): BridgeMethod | null {
  if (!hasMethod(value, name)) return null;
  return value[name];
}

function stringValue(value: unknown, fallback = ''): string {
  return typeof value === 'string' ? value.trim() : fallback;
}

function recordArray(value: unknown, key: string): unknown[] {
  if (Array.isArray(value)) return value;
  if (isRecord(value)) {
    if (Array.isArray(value[key])) return value[key];
    if (value.error !== undefined) {
      throw new Error(typeof value.error === 'string' ? value.error : 'The project bridge returned an error.');
    }
  }
  return [];
}

function normalizeProject(value: unknown, _index: number, source: ProjectSource): ProjectRecord | null {
  if (!isRecord(value)) return null;

  const rawId = value.id;
  const id = stringValue(value.id) || (typeof rawId === 'number' && Number.isFinite(rawId) ? String(rawId) : '');
  const name = stringValue(value.name) || stringValue(value.label);
  const path = stringValue(value.path);
  if (!id || !name) return null;

  return {
    id,
    name,
    path,
    createdAt: stringValue(value.createdAt),
    updatedAt: stringValue(value.updatedAt) || stringValue(value.openedAt) || stringValue(value.modifiedAt),
    source,
  };
}

function normalizeProjects(values: unknown[], source: ProjectSource): ProjectRecord[] {
  return values
    .map((value, index) => normalizeProject(value, index, source))
    .filter((value): value is ProjectRecord => value !== null);
}

function makeId(): string {
  if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
    return `local-project-${crypto.randomUUID()}`;
  }
  return `local-project-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function loadLocalProjects() {
  if (typeof localStorage === 'undefined') return;
  try {
    const stored = localStorage.getItem(LOCAL_PROJECTS_KEY);
    if (!stored) return;
    const parsed: unknown = JSON.parse(stored);
    if (Array.isArray(parsed)) {
      localProjects.value = normalizeProjects(parsed, 'local');
    }
  } catch {
    localProjects.value = [];
  }
}

function persistLocalProjects() {
  if (typeof localStorage === 'undefined') return false;
  try {
    const records = localProjects.value.map(({ source: _source, ...record }) => record);
    localStorage.setItem(LOCAL_PROJECTS_KEY, JSON.stringify(records));
    return true;
  } catch {
    formError.value = 'The browser could not save this local project.';
    return false;
  }
}

async function loadApplicationProjects() {
  if (bridge.value) {
    const listProjects = getMethod(bridge.value, 'listProjects') ?? getMethod(bridge.value, 'listRecentProjects');
    if (listProjects) {
      await loadBridgeProjects(listProjects);
      return;
    }
  }
  await loadApiProjects();
}

async function loadBridgeProjects(listProjects: BridgeMethod) {
  isLoading.value = true;
  bridgeError.value = '';
  try {
    const result = await listProjects();
    const values = recordArray(result, 'projects');
    if (!Array.isArray(result) && !isRecord(result)) {
      throw new Error('The project bridge returned no record list.');
    }
    applicationProjects.value = normalizeProjects(values, 'application');
  } catch (error) {
    applicationProjects.value = [];
    bridgeError.value = errorMessage(error);
  } finally {
    isLoading.value = false;
  }
}

async function loadApiProjects() {
  isLoading.value = true;
  bridgeError.value = '';
  try {
    const result = await api.listBackendProjects();
    if (!isRecord(result) || result.status === 'error' || !Array.isArray(result.projects)) {
      throw new Error(
        isRecord(result) && typeof result.error === 'string'
          ? result.error
          : 'The backend returned no project list.',
      );
    }
    apiActive.value = true;
    applicationProjects.value = normalizeProjects(result.projects, 'api');
  } catch (error) {
    apiActive.value = false;
    applicationProjects.value = [];
    bridgeError.value = '';
  } finally {
    isLoading.value = false;
  }
}

async function refreshProjects() {
  if (!bridgeCanList.value && !apiActive.value) {
    await loadApiProjects();
    return;
  }
  await loadApplicationProjects();
}

async function addProject() {
  formError.value = '';
  actionMessage.value = '';
  const name = projectName.value.trim();
  const path = projectPath.value.trim();

  if (!name) {
    formError.value = 'Enter a project name before saving.';
    return;
  }

  if (bridgeCanAdd.value) {
    await addApplicationProject(name, path);
    return;
  }

  if (apiActive.value) {
    await addApiProject(name, path);
    return;
  }

  const record: ProjectRecord = {
    id: makeId(),
    name,
    path,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    source: 'local',
  };
  localProjects.value = [record, ...localProjects.value];
  if (!persistLocalProjects()) return;
  projectName.value = '';
  projectPath.value = '';
  actionMessage.value = 'Project saved as a local browser record.';
}

async function addApplicationProject(name: string, path: string) {
  isSaving.value = true;
  try {
    const addProject = getMethod(bridge.value, 'addProject');
    if (addProject) {
      const result = await addProject({
        id: makeId(),
        name,
        path,
        createdAt: new Date().toISOString(),
      });
      if (isRecord(result) && result.error !== undefined) {
        throw new Error(typeof result.error === 'string' ? result.error : 'The project bridge returned an error.');
      }
      const returned = normalizeProject(result, Date.now(), 'application');
      if (returned) {
        applicationProjects.value = [returned, ...applicationProjects.value.filter((project) => project.id !== returned.id)];
      } else {
        await loadApplicationProjects();
      }
    } else {
      const addRecentProject = getMethod(bridge.value, 'addRecentProject');
      if (!addRecentProject) throw new Error('The project bridge does not expose an add action.');
      if (!path) throw new Error('This bridge requires a project path.');
      const result = await addRecentProject(path, name);
      if (isRecord(result) && result.error !== undefined) {
        throw new Error(typeof result.error === 'string' ? result.error : 'The project bridge returned an error.');
      }
      const returned = normalizeProject(result, Date.now(), 'application');
      if (returned) {
        applicationProjects.value = [returned, ...applicationProjects.value.filter((project) => project.id !== returned.id)];
      } else {
        await loadApplicationProjects();
      }
    }
    projectName.value = '';
    projectPath.value = '';
    actionMessage.value = 'Project added through the local application bridge.';
  } catch (error) {
    formError.value = errorMessage(error);
  } finally {
    isSaving.value = false;
  }
}

async function addApiProject(name: string, path: string) {
  isSaving.value = true;
  try {
    const result = await api.addBackendProject({ name, path });
    if (!isRecord(result) || result.status === 'error' || !isRecord(result.project)) {
      throw new Error(
        isRecord(result) && typeof result.error === 'string'
          ? result.error
          : 'The backend refused the project record.',
      );
    }
    const returned = normalizeProject(result.project, Date.now(), 'api');
    if (returned) {
      applicationProjects.value = [returned, ...applicationProjects.value.filter((project) => project.id !== returned.id)];
    } else {
      await loadApplicationProjects();
    }
    projectName.value = '';
    projectPath.value = '';
    actionMessage.value = 'Project saved to the backend database.';
  } catch (error) {
    formError.value = errorMessage(error);
  } finally {
    isSaving.value = false;
  }
}

async function chooseProject() {
  const choose = getMethod(bridge.value, 'chooseProject');
  if (!choose || isChoosing.value) return;

  isChoosing.value = true;
  formError.value = '';
  try {
    const result = await choose();
    if (!result) {
      actionMessage.value = 'No project folder was selected.';
      return;
    }
    if (isRecord(result) && result.error !== undefined) {
      throw new Error(typeof result.error === 'string' ? result.error : 'The project chooser returned an error.');
    }
    const record = normalizeProject(result, Date.now(), 'application');
    if (!record) throw new Error('The folder chooser returned an unreadable project record.');
    applicationProjects.value = [record, ...applicationProjects.value.filter((project) => project.id !== record.id)];
    actionMessage.value = 'Project folder added through the local application bridge.';
  } catch (error) {
    formError.value = errorMessage(error);
  } finally {
    isChoosing.value = false;
  }
}

async function removeApplicationProject(project: ProjectRecord) {
  const remove = getMethod(bridge.value, 'removeProject');
  if (!remove) return;

  try {
    const result = await remove(project.id);
    if (isRecord(result) && result.error !== undefined) {
      throw new Error(typeof result.error === 'string' ? result.error : 'The project bridge returned an error.');
    }
    applicationProjects.value = applicationProjects.value.filter((item) => item.id !== project.id);
    actionMessage.value = `Removed ${project.name} from the application database.`;
  } catch (error) {
    formError.value = errorMessage(error);
  }
}

function removeLocalProject(project: ProjectRecord) {
  localProjects.value = localProjects.value.filter((item) => item.id !== project.id);
  if (!persistLocalProjects()) return;
  actionMessage.value = `Removed ${project.name} from this browser.`;
}

async function showInFolder(project: ProjectRecord) {
  const reveal = getMethod(bridge.value, 'showInFolder');
  if (!reveal || !project.path) return;
  try {
    const result = await reveal(project.path);
    if (isRecord(result) && result.error !== undefined) {
      throw new Error(typeof result.error === 'string' ? result.error : 'The local application bridge returned an error.');
    }
  } catch (error) {
    formError.value = errorMessage(error);
  }
}

function formatTimestamp(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function errorMessage(error: unknown): string {
  if (error instanceof Error && error.message) return error.message;
  if (typeof error === 'string' && error.trim()) return error;
  if (isRecord(error) && typeof error.error === 'string' && error.error.trim()) return error.error;
  return 'The local application bridge could not complete that project action.';
}
</script>

<style scoped>
.projects-surface {
  min-height: 100%;
  padding: clamp(18px, 3vw, 30px);
  background: var(--surface);
  color: var(--text);
  color-scheme: light;
}

.surface-header,
.panel-heading,
.form-actions,
.record-title-line {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 14px;
}

.surface-header {
  align-items: center;
  margin-bottom: 20px;
}

.header-copy,
.record-main {
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
  background: var(--text);
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
  font-weight: 700;
}

.surface-notice p {
  margin: 0;
}

.notice-warning {
  border-left-color: var(--text-dim);
}

.notice-error {
  border-left-color: var(--danger);
}

.notice-success {
  border-left-color: var(--success);
}

.projects-layout {
  display: grid;
  grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.65fr);
  gap: 14px;
  align-items: start;
}

.surface-panel {
  min-width: 0;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
}

.records-panel,
.entry-panel,
.local-panel {
  padding: clamp(16px, 2.2vw, 22px);
}

.local-panel {
  margin-top: 14px;
}

.panel-heading {
  margin-bottom: 18px;
}

.panel-heading h3 {
  margin: 0;
  color: var(--text);
  font-size: 17px;
  font-weight: 720;
  letter-spacing: -0.02em;
}

.record-badge,
.local-badge,
.destination-badge {
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
  background: var(--surface);
  color: var(--text-dim);
}

.record-title-line {
  align-items: center;
  justify-content: flex-start;
  gap: 8px;
}

.record-title-line h4 {
  min-width: 0;
  margin: 0;
  overflow: hidden;
  color: var(--text);
  font-size: 13px;
  font-weight: 700;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.record-badge-local {
  border-color: var(--border-light);
}

.record-list {
  display: grid;
  gap: 8px;
  margin: 0;
  padding: 0;
  list-style: none;
}

.record-item {
  display: flex;
  min-width: 0;
  align-items: flex-start;
  justify-content: space-between;
  gap: 12px;
  padding: 11px 12px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
}

.record-path {
  margin: 6px 0 0;
  overflow: hidden;
  color: var(--text-dim);
  font: 11px/1.45 var(--font-mono);
  text-overflow: ellipsis;
  white-space: nowrap;
}

.record-muted {
  color: var(--text-muted);
  font-family: var(--font);
}

.record-meta {
  margin: 5px 0 0;
  color: var(--text-muted);
  font-size: 10px;
}

.record-actions {
  display: flex;
  flex: 0 0 auto;
  align-items: center;
  gap: 5px;
}

.icon-button,
.remove-button,
.quiet-button,
.primary-button,
.secondary-button {
  border: 1px solid var(--border);
  border-radius: 6px;
  cursor: pointer;
  font: 700 11px/1 var(--font);
}

.icon-button,
.remove-button,
.quiet-button {
  min-height: 30px;
  background: var(--surface);
  color: var(--text-muted);
}

.icon-button {
  width: 30px;
  padding: 0;
  font-size: 15px;
}

.remove-button {
  padding: 0 8px;
}

.icon-button:hover:not(:disabled),
.remove-button:hover:not(:disabled),
.quiet-button:hover:not(:disabled),
.secondary-button:hover:not(:disabled) {
  border-color: var(--border-light);
  background: var(--surface);
  color: var(--text);
}

.quiet-button {
  padding: 0 10px;
}

.primary-button,
.secondary-button {
  min-height: 36px;
  padding: 0 12px;
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

.secondary-button {
  background: var(--surface);
  color: var(--text-dim);
}

.icon-button:disabled,
.remove-button:disabled,
.quiet-button:disabled,
.primary-button:disabled,
.secondary-button:disabled {
  cursor: not-allowed;
  opacity: 0.45;
}

.panel-message {
  display: flex;
  min-height: 76px;
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
  animation: project-spin 800ms linear infinite;
}

@keyframes project-spin {
  to { transform: rotate(360deg); }
}

.entry-form {
  display: grid;
  gap: 14px;
}

.field {
  min-width: 0;
}

.field label {
  display: block;
  margin-bottom: 6px;
  color: var(--text-dim);
  font-size: 12px;
  font-weight: 700;
}

.optional-label {
  color: var(--text-muted);
  font-size: 10px;
  font-weight: 500;
}

.field input {
  display: block;
  width: 100%;
  min-height: 38px;
  padding: 0 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  outline: 0;
  background: var(--surface);
  color: var(--text);
  font: 12px/1.4 var(--font);
}

.field input:focus {
  border-color: var(--primary);
  box-shadow: 0 0 0 2px var(--accent-soft);
}

.field input:focus-visible {
  outline: 2px solid var(--primary);
  outline-offset: 2px;
}

.destination-note {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  padding: 9px 10px;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: var(--surface-2);
  color: var(--text-muted);
  font-size: 11px;
  line-height: 1.45;
}

.destination-badge {
  flex: 0 0 auto;
}

.form-message {
  margin: 0;
  font-size: 11px;
  line-height: 1.45;
}

.form-message-error {
  color: var(--danger);
}

.form-actions {
  align-items: center;
  flex-wrap: wrap;
}

.local-list {
  min-width: 0;
}

@media (max-width: 840px) {
  .projects-layout {
    grid-template-columns: minmax(0, 1fr);
  }
}

@media (max-width: 580px) {
  .surface-header,
  .panel-heading {
    align-items: flex-start;
    flex-direction: column;
  }

  .surface-status {
    align-self: flex-start;
  }

  .record-item {
    flex-direction: column;
  }

  .record-actions {
    align-self: flex-end;
  }
}
</style>

<template>
  <section class="workspace" aria-labelledby="workspace-title">
    <header class="workspace-header">
      <div class="heading">
        <span class="eyebrow">MECH / MECHANISTIC INTERPRETABILITY</span>
        <h1 id="workspace-title">Neural Network Workspace</h1>
        <p>Explore transformer layers, attention heads, MLP neurons, and measured model activations.</p>
      </div>
      <div class="header-actions">
        <span class="status" :class="'status--' + runtimeState"><i></i>{{ runtimeLabel }}</span>
        <button class="button button--quiet" :disabled="architectureLoading" @click="loadArchitecture">{{ architectureLoading ? 'Connecting…' : 'Refresh model' }}</button>
      </div>
    </header>

    <div class="command-bar">
      <label class="prompt-field"><span>INPUT PROMPT</span><input v-model="prompt" :disabled="capturing" placeholder="The capital of France is" @keydown.enter.prevent="captureActivations" /></label>
      <button class="button button--primary" :disabled="capturing || !prompt.trim()" @click="captureActivations">{{ capturing ? 'Capturing activations…' : 'Capture activations' }}</button>
      <span class="command-note">Runs a real forward pass on the loaded GPT-2 model.</span>
    </div>

    <div v-if="errorMessage" class="notice notice--error" role="alert">{{ errorMessage }}</div>
    <div v-if="captureMessage" class="notice" role="status">{{ captureMessage }}</div>

    <div class="metrics">
      <div><span>MODEL</span><strong>{{ modelName }}</strong><small>{{ architecture ? 'Runtime architecture' : 'Awaiting model metadata' }}</small></div>
      <div><span>TRANSFORMER LAYERS</span><strong>{{ layerCount || '—' }}</strong><small>Returned by backend</small></div>
      <div><span>SELECTED ELEMENT</span><strong>{{ selected?.label || 'None' }}</strong><small>{{ selected?.kind || 'Select a node in the workspace' }}</small></div>
      <div><span>ACTIVATION</span><strong>{{ selectedActivation === null ? '—' : formatNumber(selectedActivation) }}</strong><small>{{ activationSource }}</small></div>
    </div>

    <div class="workspace-grid">
      <aside class="side-panel layers-panel">
        <div class="panel-title"><div><span class="eyebrow">MODEL TREE</span><h2>Architecture</h2></div><span class="count">{{ layerCount || 0 }}</span></div>
        <div v-if="architectureLoading && !architecture" class="empty-state">Reading model architecture…</div>
        <div v-else-if="!architecture" class="empty-state">No architecture data returned. Connect the MECH backend and load GPT-2 to populate the workspace.</div>
        <template v-else>
          <button class="tree-row root-row" @click="expandedAll = !expandedAll"><span class="chevron">{{ expandedAll ? '▾' : '▸' }}</span><span class="tree-icon">M</span><span class="tree-label"><b>{{ modelName }}</b><small>Decoder-only transformer</small></span></button>
          <div v-for="layer in layers" :key="layer.index" class="tree-group">
            <button class="tree-row" :class="{ 'tree-row--active': selected?.layer === layer.index && selected?.kind === 'Layer' }" @click="toggleLayer(layer.index)">
              <span class="chevron">{{ isExpanded(layer.index) ? '▾' : '▸' }}</span><span class="tree-icon tree-icon--layer">{{ layer.index + 1 }}</span><span class="tree-label"><b>Layer {{ layer.index }}</b><small>{{ layer.heads === null ? 'Head count not supplied' : layer.heads + ' attention heads' }}</small></span>
            </button>
            <div v-if="isExpanded(layer.index) && expandedAll" class="tree-children">
              <button v-for="head in headIndices(layer)" :key="'h'+head" class="tree-row tree-row--child" :class="{ 'tree-row--selected': selected?.layer === layer.index && selected?.kind === 'Attention head' && selected?.index === head }" @click="selectHead(layer.index, head)">
                <span class="tree-icon tree-icon--head">A</span><span class="tree-label"><b>Attention head {{ head }}</b><small>Layer {{ layer.index }} · head {{ head }}</small></span>
              </button>
              <button class="tree-row tree-row--child" :class="{ 'tree-row--selected': selected?.layer === layer.index && selected?.kind === 'MLP' }" @click="selectMlp(layer.index)">
                <span class="tree-icon tree-icon--mlp">ƒ</span><span class="tree-label"><b>MLP neurons</b><small>{{ layer.mlpWidth === null ? 'Width not supplied' : layer.mlpWidth.toLocaleString() + ' neurons' }}</small></span>
              </button>
            </div>
          </div>
        </template>
        <div class="tree-legend"><span><i class="dot dot--blue"></i>Layer</span><span><i class="dot dot--purple"></i>Attention</span><span><i class="dot dot--green"></i>MLP</span></div>
      </aside>

      <main class="graph-panel">
        <div class="graph-toolbar">
          <div><span class="eyebrow">LIVE GRAPH</span><h2>Information flow</h2></div>
          <div class="graph-actions">
            <label class="toggle"><input v-model="showConnections" type="checkbox" /> Connections</label>
            <button class="icon-button" title="Zoom out" aria-label="Zoom out" @click="zoom = Math.max(.45, zoom - .15)">−</button>
            <span class="zoom-label">{{ Math.round(zoom * 100) }}%</span>
            <button class="icon-button" title="Zoom in" aria-label="Zoom in" @click="zoom = Math.min(2.5, zoom + .15)">+</button>
            <button class="button button--quiet button--small" @click="resetView">Fit view</button>
          </div>
        </div>
        <div ref="stage" class="graph-stage" @wheel.prevent="onWheel" @pointerdown="onStagePointerDown" @pointermove="onPointerMove" @pointerup="onPointerUp" @pointercancel="onPointerUp" @pointerleave="onPointerUp">
          <div class="canvas-hint"><span>DRAG</span> pan canvas <span>SCROLL</span> zoom <span>CLICK</span> select</div>
          <svg v-if="layers.length" class="network-svg" :viewBox="`0 0 ${canvasWidth} ${canvasHeight}`" role="img" aria-label="Interactive transformer layer graph">
            <defs><pattern id="workspace-grid" width="28" height="28" patternUnits="userSpaceOnUse"><path d="M 28 0 L 0 0 0 28" fill="none" stroke="#cbd5e1" stroke-opacity=".36" stroke-width=".7" /></pattern><filter id="node-shadow" x="-40%" y="-40%" width="180%" height="180%"><feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#64748b" flood-opacity=".18" /></filter></defs>
            <rect :width="canvasWidth" :height="canvasHeight" fill="url(#workspace-grid)" />
            <g :transform="`translate(${pan.x} ${pan.y}) scale(${zoom})`">
              <g v-if="showConnections">
                <path v-for="edge in graphEdges" :key="edge.id" :d="edge.d" class="connection" :class="{ 'connection--highlight': edge.highlight, 'connection--dim': selected && !edge.highlight }" />
              </g>
              <g v-for="node in graphNodes" :key="node.id" class="graph-node" :class="{ 'graph-node--selected': selected?.id === node.id, 'graph-node--related': relatedIds.has(node.id) }" :transform="`translate(${node.x} ${node.y})`" tabindex="0" role="button" :aria-label="'Select ' + node.label" @pointerdown.stop="onNodePointerDown($event,node)" @click.stop="selectNode(node)" @keydown.enter.prevent="selectNode(node)" @keydown.space.prevent="selectNode(node)">
                <rect class="node-card" :width="node.width" :height="node.height" rx="9" filter="url(#node-shadow)" />
                <rect class="node-accent" width="4" :height="node.height" rx="2" />
                <circle class="node-indicator" cx="19" cy="19" r="4" />
                <text x="31" y="18" class="node-kicker">{{ node.kind.toUpperCase() }}</text>
                <text x="14" y="39" class="node-name">{{ node.label }}</text>
                <text x="14" y="56" class="node-meta">{{ node.meta }}</text>
                <text v-if="node.activation !== null" :x="node.width-10" y="18" text-anchor="end" class="node-value">{{ formatNumber(node.activation) }}</text>
                <g v-if="node.kind === 'Layer'">
                  <text x="14" y="node.height-10" class="node-expand">Select to inspect · expand in tree</text>
                </g>
              </g>
              <g v-if="!graphNodes.length"><text x="500" y="170" text-anchor="middle" class="empty-svg">Connect a model to render its architecture</text></g>
            </g>
          </svg>
          <div v-if="!layers.length && !architectureLoading" class="graph-empty"><div class="empty-glyph">⌘</div><h3>Waiting for model architecture</h3><p>The graph is populated from the backend response; no synthetic layer topology is drawn.</p><button class="button button--primary" @click="loadArchitecture">Retry connection</button></div>
        </div>
        <div class="graph-footer"><span><i class="dot dot--blue"></i> Model topology</span><span><i class="dot dot--green"></i> Selected pathway</span><span class="footer-spacer"></span><span>Pan: drag empty canvas · Zoom: wheel or controls</span></div>
      </main>

      <aside class="side-panel inspector-panel">
        <div class="panel-title"><div><span class="eyebrow">INSPECTOR</span><h2>{{ selected?.label || 'Selection details' }}</h2></div><span class="count">{{ selected?.kind || '—' }}</span></div>
        <div v-if="!selected" class="empty-state">Select a layer, attention head, or MLP node to inspect its available data.</div>
        <template v-else>
          <div class="selection-card"><span class="selection-icon">{{ selected.kind === 'Attention head' ? 'A' : selected.kind.startsWith('MLP') ? 'ƒ' : 'L' }}</span><div><b>{{ selected.label }}</b><small>{{ selected.kind }} · layer {{ selected.layer }}</small></div></div>
          <div class="detail-tabs"><button :class="{active: detailTab==='overview'}" @click="detailTab='overview'">Overview</button><button :class="{active: detailTab==='path'}" @click="detailTab='path'">Pathway</button></div>
          <template v-if="detailTab==='overview'">
            <div class="detail-row"><span>Activation</span><b>{{ selectedActivation === null ? 'Not captured' : formatNumber(selectedActivation) }}</b></div>
            <div v-if="selectedActivation !== null" class="activation-meter"><span :style="{width: activationPercent + '%'}"></span></div>
            <div class="detail-row"><span>Data source</span><b>{{ activationSource }}</b></div>
            <div class="detail-row"><span>Model layer index</span><b>{{ selected.layer }}</b></div>
            <div v-if="selected.kind==='Attention head'" class="detail-row"><span>Head index</span><b>{{ selected.index }}</b></div>\n            <div v-if="selected.kind==='MLP neuron'" class="detail-row"><span>Neuron index</span><b>{{ selected.index }}</b></div>
            <div v-if="selected.kind==='MLP' || selected.kind==='MLP neuron'" class="detail-row"><span>MLP width</span><b>{{ selected.mlpWidth === null ? 'Not returned' : selected.mlpWidth.toLocaleString() }}</b></div>
            <div class="inspector-note">Unavailable values remain unavailable; this panel does not estimate or fabricate measurements.</div>
          </template>
          <template v-else>
            <div class="pathway-actions"><button class="button button--quiet" @click="highlightDirection('upstream')">Highlight upstream</button><button class="button button--quiet" @click="highlightDirection('downstream')">Highlight downstream</button><button class="button button--quiet" @click="highlightDirection('both')">Show both</button><button class="button button--quiet" @click="clearPathway">Clear highlight</button></div>
            <div class="inspector-note">Pathway highlighting follows the displayed layer graph. It does not claim causal attribution without a supported intervention result.</div>
          </template>
        </template>
        <div class="inspector-bottom"><span class="eyebrow">ACTIVATION CAPTURE</span><p>{{ capture ? 'Measured values from the latest backend response.' : 'No activation capture yet.' }}</p><button class="button button--primary button--full" :disabled="capturing || !prompt.trim()" @click="captureActivations">{{ capturing ? 'Capturing…' : 'Run forward pass' }}</button><small v-if="captureProvenance">Provenance: {{ captureProvenance }}</small></div>
      </aside>
    </div>
    <footer class="workspace-footer"><span><i class="status-dot" :class="'status-dot--' + runtimeState"></i>{{ runtimeLabel }}</span><span>{{ modelName }} · {{ capture ? 'Measured activations available' : 'Activation data not captured' }}</span><span class="footer-spacer"></span><span>MECH / Neural Network Workspace</span></footer>
  </section>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue';
import {
  asArray, asNumber, asString, describeError, fieldProvenanceSummary,
  gpt2Activations, gpt2Architecture, gpt2LayerActivations, isRecord,
  numberMatrix, provenanceOf, statusOf, tokenLabels, type JsonRecord,
} from '../services/interpretabilityClient';

type LayerInfo = { index: number; heads: number | null; mlpWidth: number | null; label: string };
type GraphNode = { id: string; label: string; kind: string; layer: number; index: number; x: number; y: number; width: number; height: number; meta: string; activation: number | null; mlpWidth: number | null };
type GraphEdge = { id: string; d: string; from: string; to: string; highlight: boolean };
type Selection = { id: string; label: string; kind: string; layer: number; index: number; mlpWidth: number | null };
const prompt = ref('The capital of France is');
const architecture = ref<JsonRecord | null>(null);
const architectureLoading = ref(false);
const capturing = ref(false);
const errorMessage = ref('');
const captureMessage = ref('');
const capture = ref<{ prompt: string; provenance: string; tokens: string[]; layerRecord: JsonRecord; matrices: unknown[] } | null>(null);
const layers = ref<LayerInfo[]>([]);
const selected = ref<Selection | null>(null);
const expanded = ref(new Set<number>());
const expandedAll = ref(true);\nconst expandedMlp = ref(new Set<number>());
const showConnections = ref(true);
const detailTab = ref<'overview' | 'path'>('overview');
const highlightMode = ref<'none' | 'upstream' | 'downstream' | 'both'>('none');
const zoom = ref(1);
const pan = ref({ x: 24, y: 28 });
const stage = ref<HTMLElement | null>(null);
const nodePositions = ref<Record<string, {x:number;y:number}>>({});
const drag = ref<{kind:'pan'|'node'; pointerId:number; startX:number; startY:number; originX:number; originY:number; nodeId?:string} | null>(null);
const layerCount = computed(() => layers.value.length);
const modelName = computed(() => asString(architecture.value?.model_name) || asString(architecture.value?.model) || 'GPT-2');
const runtimeState = computed(() => architectureLoading.value ? 'loading' : architecture.value ? 'online' : 'offline');
const runtimeLabel = computed(() => architectureLoading.value ? 'Connecting' : architecture.value ? 'Backend connected' : 'Backend unavailable');
const activationSource = computed(() => capture.value ? (capture.value.provenance || 'Backend response; provenance not supplied') : 'Not captured');
const selectedActivation = computed(() => {
  if (!capture.value || !selected.value) return null;
  const layer = selected.value.layer;
  const record = layer === 0 ? capture.value.layerRecord : capture.value.matrices[layer];
  if (!isRecord(record)) return null;
  const keys = selected.value.kind === 'Attention head' ? ['attention_activations','attn_activations','head_activations'] : (selected.value.kind === 'MLP' || selected.value.kind === 'MLP neuron') ? ['mlp_activations','post_mlp','mlp_post','activations'] : ['resid_post','residual_activations','activations'];
  for (const key of keys) {
    const matrix = numberMatrix(record[key]);
    if (matrix.length) {
      const row = matrix[Math.min(selectedTokenIndex.value, matrix.length - 1)];
      if (!row?.length) continue;
      const index = selected.value.kind === 'Attention head' || selected.value.kind === 'MLP neuron' ? selected.value.index : 0;
      const value = row[Math.min(index, row.length - 1)];
      if (typeof value === 'number') return value;
      const numbers = row.filter((v): v is number => typeof v === 'number');
      if (numbers.length) return numbers.reduce((sum, v) => sum + v, 0) / numbers.length;
    }
  }
  return null;
});
const selectedTokenIndex = ref(0);
const activationPercent = computed(() => selectedActivation.value === null ? 0 : Math.min(100, Math.abs(selectedActivation.value) / (1 + Math.abs(selectedActivation.value)) * 100));
const canvasWidth = 1280;
const canvasHeight = 650;
const expandedSet = computed(() => expanded.value);
const graphNodes = computed<GraphNode[]>(() => {
  const result: GraphNode[] = [];
  const positions = nodePositions.value;
  const total = Math.max(layers.value.length, 1);
  const startX = 72;
  const spacing = Math.max(155, Math.min(220, (canvasWidth - 180) / Math.max(total - 1, 1)));
  layers.value.forEach((layer, index) => {
    const id = 'layer-' + layer.index;
    const baseX = startX + index * spacing;
    const baseY = 170;
    const pos = positions[id] || {x:baseX,y:baseY};
    result.push({id,label:'Layer '+layer.index,kind:'Layer',layer:layer.index,index:layer.index,x:pos.x,y:pos.y,width:152,height:78,meta:(layer.heads === null ? 'Heads —' : layer.heads+' heads')+' · MLP '+(layer.mlpWidth === null ? '—' : layer.mlpWidth),activation:layerActivation(layer.index),mlpWidth:layer.mlpWidth});
    if (selected.value?.layer === layer.index && selected.value.kind === 'Attention head') {
      const headX = pos.x;
      const headY = pos.y + 125;
      const hPos = positions['head-'+layer.index+'-'+selected.value.index] || {x:headX,y:headY};
      result.push({id:'head-'+layer.index+'-'+selected.value.index,label:'Head '+selected.value.index,kind:'Attention head',layer:layer.index,index:selected.value.index,x:hPos.x,y:hPos.y,width:142,height:70,meta:'Attention head',activation:null,mlpWidth:null});
    }
    if (selected.value?.layer === layer.index && (selected.value.kind === 'MLP' || selected.value.kind === 'MLP neuron')) {
      const mPos = positions['mlp-'+layer.index] || {x:pos.x,y:pos.y+125};
      const isNeuron = selected.value.kind === 'MLP neuron';
      result.push({id:isNeuron?'neuron-'+layer.index+'-'+selected.value.index:'mlp-'+layer.index,label:isNeuron?'Neuron '+selected.value.index:'MLP neurons',kind:isNeuron?'MLP neuron':'MLP',layer:layer.index,index:isNeuron?selected.value.index:0,x:mPos.x,y:mPos.y,width:142,height:70,meta:isNeuron?'MLP activation dimension':layer.mlpWidth === null ? 'Width unavailable' : layer.mlpWidth+' neurons',activation:isNeuron?mlpNeuronActivation(layer.index,selected.value.index):layerActivation(layer.index,'MLP'),mlpWidth:layer.mlpWidth});
    }
  });
  return result;
});
const graphEdges = computed<GraphEdge[]>(() => {
  const result: GraphEdge[] = [];
  const byLayer = [...graphNodes.value].filter(n => n.kind === 'Layer').sort((a,b)=>a.layer-b.layer);
  for (let i=0;i<byLayer.length-1;i++) {
    const a=byLayer[i], b=byLayer[i+1];
    const dx=b.x-a.x, dy=b.y-a.y;
    const id='edge-'+a.id+'-'+b.id;
    const related = relatedIds.value.has(a.id) && relatedIds.value.has(b.id);
    result.push({id,from:a.id,to:b.id,highlight:related,d:`M ${a.x+a.width} ${a.y+a.height/2} C ${a.x+a.width+dx*.38} ${a.y+a.height/2}, ${b.x-dx*.38} ${b.y+b.height/2}, ${b.x} ${b.y+b.height/2}`});
  }
  for (const node of graphNodes.value.filter(n=>n.kind!=='Layer')) {
    const parent=byLayer.find(n=>n.layer===node.layer);
    if (!parent) continue;
    result.push({id:'edge-'+parent.id+'-'+node.id,from:parent.id,to:node.id,highlight:relatedIds.value.has(parent.id)&&relatedIds.value.has(node.id),d:`M ${parent.x+parent.width/2} ${parent.y+parent.height} C ${parent.x+parent.width/2} ${parent.y+parent.height+38}, ${node.x+node.width/2} ${node.y-38}, ${node.x+node.width/2} ${node.y}`});
  }
  return result;
});
const relatedIds = computed(() => {
  const ids = new Set<string>();
  if (!selected.value || highlightMode.value === 'none') return ids;
  for (const node of graphNodes.value) {
    if (highlightMode.value === 'both' || (highlightMode.value === 'upstream' && node.layer <= selected.value.layer) || (highlightMode.value === 'downstream' && node.layer >= selected.value.layer)) ids.add(node.id);
  }
  return ids;
});

function isExpanded(index:number):boolean { return expandedSet.value.has(index); }
function headIndices(layer:LayerInfo):number[] { return Array.from({length:Math.min(layer.heads ?? 0, 16)},(_,i)=>i); }\nfunction neuronIndices(layer:LayerInfo):number[] { return Array.from({length:Math.min(layer.mlpWidth ?? 0, 16)},(_,i)=>i); }\nfunction toggleMlp(layer:number) { const next=new Set(expandedMlp.value); if(next.has(layer)) next.delete(layer); else next.add(layer); expandedMlp.value=next; selectMlp(layer); }\nfunction selectNeuron(layer:number,index:number) { expandedAll.value=true; expanded.value=new Set([...expanded.value,layer]); expandedMlp.value=new Set([...expandedMlp.value,layer]); selected.value={id:'neuron-'+layer+'-'+index,label:'Neuron '+index,kind:'MLP neuron',layer,index,mlpWidth:layers.value.find(l=>l.index===layer)?.mlpWidth ?? null}; highlightMode.value='both'; }
function toggleLayer(index:number) {
  const next=new Set(expanded.value);
  if(next.has(index)) next.delete(index); else next.add(index);
  expanded.value=next;
  selected.value={id:'layer-'+index,label:'Layer '+index,kind:'Layer',layer:index,index,mlpWidth:layers.value.find(l=>l.index===index)?.mlpWidth ?? null};
}
function selectNode(node:GraphNode) {
  selected.value={id:node.id,label:node.label,kind:node.kind,layer:node.layer,index:node.index,mlpWidth:node.mlpWidth};
  detailTab.value='overview';
  highlightMode.value='both';
}
function selectHead(layer:number,head:number) {
  expandedAll.value=true;
  expanded.value=new Set([...expanded.value,layer]);
  selected.value={id:'head-'+layer+'-'+head,label:'Attention head '+head,kind:'Attention head',layer,index:head,mlpWidth:null};
  highlightMode.value='both';
}
function selectMlp(layer:number) {
  expandedAll.value=true;
  expanded.value=new Set([...expanded.value,layer]);
  selected.value={id:'mlp-'+layer,label:'MLP neurons',kind:'MLP',layer,index:0,mlpWidth:layers.value.find(l=>l.index===layer)?.mlpWidth ?? null};
  highlightMode.value='both';
}
function highlightDirection(mode:'upstream'|'downstream'|'both') { highlightMode.value=mode; }
function clearPathway() { highlightMode.value='none'; }
function resetView() { zoom.value=1; pan.value={x:24,y:28}; nodePositions.value={}; }
function onWheel(event:WheelEvent) { zoom.value=Math.max(.45,Math.min(2.5,zoom.value+(event.deltaY<0?.08:-.08))); }
function onStagePointerDown(event:PointerEvent) {
  if ((event.target as HTMLElement).closest('.graph-node')) return;
  drag.value={kind:'pan',pointerId:event.pointerId,startX:event.clientX,startY:event.clientY,originX:pan.value.x,originY:pan.value.y};
  (event.currentTarget as HTMLElement).setPointerCapture?.(event.pointerId);
}
function onNodePointerDown(event:PointerEvent,node:GraphNode) {
  drag.value={kind:'node',pointerId:event.pointerId,startX:event.clientX,startY:event.clientY,originX:node.x,originY:node.y,nodeId:node.id};
  (event.currentTarget as HTMLElement).setPointerCapture?.(event.pointerId);
}
function onPointerMove(event:PointerEvent) {
  if(!drag.value || drag.value.pointerId!==event.pointerId) return;
  const dx=(event.clientX-drag.value.startX)/zoom.value, dy=(event.clientY-drag.value.startY)/zoom.value;
  if(drag.value.kind==='pan') pan.value={x:drag.value.originX+(event.clientX-drag.value.startX),y:drag.value.originY+(event.clientY-drag.value.startY)};
  else if(drag.value.nodeId) nodePositions.value={...nodePositions.value,[drag.value.nodeId]:{x:drag.value.originX+dx,y:drag.value.originY+dy}};
}
function onPointerUp(event:PointerEvent) { if(drag.value?.pointerId===event.pointerId) drag.value=null; }

function architectureLayerCount(record:JsonRecord):number {
  const rows=asArray(record.layers);
  return rows.length || asNumber(record.n_layers) || asNumber(record.num_layers) || 0;
}
function readLayers(record:JsonRecord):LayerInfo[] {
  return asArray(record.layers).map((item,index)=>{
    const row=isRecord(item)?item:{};
    const heads=asNumber(row.num_attention_heads) ?? asNumber(row.n_heads) ?? asNumber(row.num_heads) ?? asNumber(record.num_attention_heads) ?? asNumber(record.n_heads);
    const mlp=asNumber(row.num_mlp_neurons) ?? asNumber(row.mlp_width) ?? asNumber(row.intermediate_size) ?? asNumber(row.mlp_dim) ?? asNumber(record.mlp_width);
    return {index:asNumber(row.layer_index) ?? asNumber(row.index) ?? index,heads,mlpWidth:mlp,label:asString(row.name)||'Layer '+index};
  });
}
async function loadArchitecture() {
  architectureLoading.value=true; errorMessage.value=''; captureMessage.value='';
  try {
    const result=await gpt2Architecture();
    if(statusOf(result)!=='ok') throw new Error(asString(result.error)||asString(result.reason)||'The backend did not return a usable architecture (status: '+(statusOf(result)||'not supplied')+').');
    const parsed=readLayers(result);
    if(!parsed.length) throw new Error('The backend responded but supplied no transformer layer list.');
    architecture.value=result; layers.value=parsed; expanded.value=new Set(parsed.slice(0,3).map(l=>l.index));
    if(!selected.value) selected.value={id:'layer-0',label:'Layer 0',kind:'Layer',layer:parsed[0].index,index:parsed[0].index,mlpWidth:parsed[0].mlpWidth};
    captureMessage.value='Architecture loaded from backend · '+parsed.length+' layers.';
  } catch(error) {
    architecture.value=null; layers.value=[];
    errorMessage.value=describeError(error);
  } finally { architectureLoading.value=false; }
}
async function captureActivations() {
  if(!prompt.value.trim()||capturing.value)return;
  capturing.value=true; errorMessage.value=''; captureMessage.value='Running forward pass and requesting layer activations…';
  try {
    const probe=await gpt2Activations({prompt:prompt.value.trim(),layer:0});
    if(statusOf(probe)!=='ok') throw new Error(asString(probe.error)||asString(probe.reason)||'Activation probe was not successful (status: '+(statusOf(probe)||'not supplied')+').');
    const records:unknown[]=[];
    for(const layer of layers.value) {
      const result=await gpt2LayerActivations({prompt:prompt.value.trim(),layer:layer.index});
      records[layer.index]=statusOf(result)==='ok'?result:null;
    }
    const first=records.find(isRecord) as JsonRecord|undefined;
    capture.value={prompt:prompt.value.trim(),provenance:provenanceOf(first||probe)||fieldProvenanceSummary(first||probe)||'Backend response; provenance not supplied',tokens:tokenLabels(probe),layerRecord:probe,matrices:records};
    captureMessage.value='Forward pass completed. Activation values will appear only where the backend returned numeric tensors.';
  } catch(error) {
    capture.value=null; captureMessage.value='';
    errorMessage.value=describeError(error);
  } finally { capturing.value=false; }
}
function tensorMean(record: unknown, keys: string[], dimension = 0): number | null {\n  if (!isRecord(record)) return null;\n  for (const key of keys) {\n    const matrix=numberMatrix(record[key]);\n    const row=matrix[Math.min(selectedTokenIndex.value,Math.max(0,matrix.length-1))];\n    if (!row?.length) continue;\n    const value=row[dimension];\n    if (typeof value==='number') return value;\n    const nums=row.filter((v):v is number=>typeof v==='number');\n    if(nums.length) return nums.reduce((sum,v)=>sum+v,0)/nums.length;\n  }\n  return null;\n}\nfunction layerActivation(index:number, kind='Layer'):number|null {\n  if(!capture.value) return null;\n  const record=index===0?capture.value.layerRecord:capture.value.matrices[index];\n  const keys=kind==='MLP'?['mlp_activations','post_mlp','mlp_post','activations']:['resid_post','residual_activations','activations'];\n  return tensorMean(record,keys);\n}\nfunction mlpNeuronActivation(layer:number,index:number):number|null {\n  if(!capture.value) return null;\n  return tensorMean(capture.value.matrices[layer],['mlp_activations','post_mlp','mlp_post','activations'],index);\n}\nfunction formatNumber(value:number):string { return Number.isFinite(value)?value.toFixed(4):'—'; }
onMounted(()=>{ void loadArchitecture(); });
onBeforeUnmount(()=>{ drag.value=null; });
</script>

<style scoped>
.workspace{display:flex;flex-direction:column;gap:14px;min-width:0;min-height:100%;padding:clamp(14px,2vw,24px);background:var(--bg);color:var(--text);font-family:var(--font);font-size:12px;color-scheme:light}
.workspace *{box-sizing:border-box}.workspace button,.workspace input{font:inherit}.workspace button{cursor:pointer}.workspace button:focus-visible,.workspace input:focus-visible,.graph-node:focus-visible{outline:2px solid var(--primary);outline-offset:2px}
.workspace-header,.header-actions,.command-bar,.metrics,.panel-title,.graph-toolbar,.graph-actions,.graph-footer,.workspace-footer,.selection-card,.detail-row,.tree-row{display:flex;align-items:center}
.workspace-header{justify-content:space-between;gap:18px}.heading{min-width:0}.eyebrow{display:block;color:var(--text-muted);font:700 9px/1.4 var(--font-mono);letter-spacing:.09em;text-transform:uppercase}.heading h1{margin:4px 0 3px;font-size:clamp(22px,2.4vw,30px);letter-spacing:-.04em;line-height:1.1}.heading p{margin:0;color:var(--text-muted);font-size:12px;line-height:1.5}.header-actions{gap:8px;flex-wrap:wrap}
.status{display:inline-flex;align-items:center;gap:6px;padding:6px 9px;border:1px solid var(--border);border-radius:99px;background:var(--surface);color:var(--text-muted);font:700 10px var(--font-mono)}.status i,.status-dot{width:6px;height:6px;border-radius:50%;background:currentColor}.status--online{color:var(--success);border-color:#b9dfd3;background:var(--success-soft)}.status--offline{color:var(--danger);background:var(--danger-soft)}.status--loading{color:var(--warning)}
.button{display:inline-flex;align-items:center;justify-content:center;min-height:32px;padding:0 11px;border:1px solid var(--border);border-radius:6px;background:var(--surface);color:var(--text);font-size:11px;font-weight:650;white-space:nowrap}.button:hover:not(:disabled){background:var(--surface-2);border-color:var(--border-light)}.button:disabled{opacity:.5;cursor:not-allowed}.button--primary{background:var(--primary);border-color:var(--primary);color:white}.button--primary:hover:not(:disabled){background:var(--primary-focus);border-color:var(--primary-focus)}.button--small{min-height:28px;padding:0 8px}
.command-bar{gap:9px;flex-wrap:wrap;padding:10px;border:1px solid var(--border);border-radius:var(--radius-lg);background:var(--surface)}.prompt-field{display:grid;flex:1 1 280px;gap:5px;min-width:180px}.prompt-field span{color:var(--text-muted);font:700 9px var(--font-mono);letter-spacing:.06em}.prompt-field input{width:100%;min-height:32px;padding:0 9px;border:1px solid var(--border);border-radius:5px;background:var(--surface);color:var(--text)}.command-note{color:var(--text-muted);font-size:10px}
.notice{padding:9px 11px;border:1px solid var(--border);border-radius:6px;background:var(--surface);color:var(--text-muted);line-height:1.5}.notice--error{border-color:#efc2c7;background:var(--danger-soft);color:var(--danger)}
.metrics{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}.metrics>div{display:flex;flex-direction:column;align-items:flex-start;gap:3px;min-width:0;padding:10px 12px;border:1px solid var(--border);border-radius:var(--radius);background:var(--surface)}.metrics span{color:var(--text-muted);font:700 9px var(--font-mono);letter-spacing:.05em}.metrics strong{max-width:100%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:15px}.metrics small{color:var(--text-muted);font-size:10px}
.workspace-grid{display:grid;grid-template-columns:minmax(190px,235px) minmax(380px,1fr) minmax(220px,270px);gap:10px;min-height:600px;flex:1}.side-panel,.graph-panel{min-width:0;border:1px solid var(--border);border-radius:var(--radius-lg);background:var(--surface);overflow:hidden}.side-panel{display:flex;flex-direction:column}.panel-title{justify-content:space-between;gap:8px;padding:13px 12px;border-bottom:1px solid var(--border)}.panel-title h2,.graph-toolbar h2{margin:3px 0 0;font-size:14px;letter-spacing:-.02em}.count{max-width:100px;padding:4px 6px;border:1px solid var(--border);border-radius:5px;color:var(--text-muted);font:700 9px var(--font-mono);overflow-wrap:anywhere}
.tree-row{width:100%;gap:7px;min-height:42px;padding:6px 10px;border:0;border-bottom:1px solid #eef1f5;background:transparent;color:var(--text);text-align:left}.tree-row:hover,.tree-row--active{background:var(--bg-hover)}.tree-row--selected{background:var(--accent-soft)}.root-row{padding:10px}.chevron{width:10px;color:var(--text-muted);font-size:12px}.tree-icon{display:grid;place-items:center;flex:0 0 24px;width:24px;height:24px;border:1px solid #c7d2fe;border-radius:6px;background:#eef2ff;color:#4338ca;font:700 10px var(--font-mono)}.tree-icon--layer{background:#eaf1ff;color:#1d4ed8}.tree-icon--head{border-color:#ddd6fe;background:#f5f3ff;color:#7c3aed}.tree-icon--mlp{border-color:#bbf7d0;background:#f0fdf4;color:#15803d}.tree-label{display:flex;flex-direction:column;gap:2px;min-width:0}.tree-label b{font-size:11px;font-weight:650}.tree-label small{color:var(--text-muted);font-size:9px}.tree-children{padding-left:15px;border-left:1px solid var(--border);margin-left:18px}.tree-row--child{min-height:36px;padding-left:6px}.tree-legend{display:flex;gap:8px;flex-wrap:wrap;margin-top:auto;padding:12px;border-top:1px solid var(--border);color:var(--text-muted);font-size:9px}.tree-legend span,.graph-footer span{display:flex;align-items:center;gap:5px}.dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--primary)}.dot--purple{background:#8b5cf6}.dot--green{background:#16a34a}
.graph-panel{display:flex;flex-direction:column}.graph-toolbar{justify-content:space-between;gap:10px;padding:12px;border-bottom:1px solid var(--border);flex-wrap:wrap}.graph-actions{gap:6px;flex-wrap:wrap}.toggle{display:flex;align-items:center;gap:5px;color:var(--text-muted);font-size:10px}.toggle input{accent-color:var(--primary)}.icon-button{display:grid;place-items:center;width:27px;height:27px;border:1px solid var(--border);border-radius:5px;background:var(--surface);color:var(--text);font-size:16px}.zoom-label{min-width:34px;text-align:center;color:var(--text-muted);font:10px var(--font-mono)}
.graph-stage{position:relative;flex:1;min-height:480px;overflow:hidden;background:#f8fafc;touch-action:none;cursor:grab}.graph-stage:active{cursor:grabbing}.network-svg{display:block;width:100%;height:100%;min-height:480px;user-select:none;overflow:visible}.canvas-hint{position:absolute;z-index:2;top:10px;left:12px;display:flex;gap:6px;align-items:center;color:#64748b;font-size:9px;pointer-events:none}.canvas-hint span{padding:3px 4px;border:1px solid #dbe2ec;border-radius:3px;background:#fff;color:#475569;font:700 8px var(--font-mono)}.connection{fill:none;stroke:#94a3b8;stroke-width:1.2;stroke-opacity:.42;pointer-events:none}.connection--highlight{stroke:#16a34a;stroke-width:2.2;stroke-opacity:.9}.connection--dim{stroke-opacity:.08}.graph-node{cursor:pointer;outline:none}.node-card{fill:#fff;stroke:#cbd5e1;stroke-width:1}.graph-node--selected .node-card{stroke:#2563eb;stroke-width:2;fill:#eff6ff}.graph-node--related .node-card{stroke:#16a34a}.node-accent{fill:#2563eb}.graph-node--selected .node-accent{fill:#16a34a}.node-indicator{fill:#3b82f6}.graph-node--selected .node-indicator{fill:#16a34a}.node-kicker{fill:#64748b;font:700 8px var(--font-mono);letter-spacing:.06em}.node-name{fill:#172033;font:650 11px var(--font)}.node-meta{fill:#64748b;font:9px var(--font)}.node-value{fill:#047857;font:700 9px var(--font-mono)}.node-expand{fill:#64748b;font:8px var(--font)}.empty-svg{fill:#64748b;font:13px var(--font)}.graph-empty{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:24px;text-align:center;pointer-events:none}.graph-empty>*{pointer-events:auto}.empty-glyph{display:grid;place-items:center;width:42px;height:42px;border:1px solid var(--border);border-radius:12px;background:#fff;color:var(--primary);font-size:23px}.graph-empty h3{margin:12px 0 4px;font-size:15px}.graph-empty p{max-width:280px;margin:0 0 12px;color:var(--text-muted);font-size:11px;line-height:1.5}
.graph-footer{gap:12px;flex-wrap:wrap;padding:9px 12px;border-top:1px solid var(--border);color:var(--text-muted);font-size:9px}.footer-spacer{flex:1}
.inspector-panel{padding-bottom:12px}.selection-card{gap:9px;margin:12px;padding:10px;border:1px solid var(--border);border-radius:7px;background:var(--surface-2)}.selection-icon{display:grid;place-items:center;width:30px;height:30px;border:1px solid #c7d2fe;border-radius:7px;background:#eef2ff;color:#4338ca;font:700 12px var(--font-mono)}.selection-card div{display:flex;flex-direction:column;gap:3px;min-width:0}.selection-card b{font-size:11px}.selection-card small{color:var(--text-muted);font-size:9px}.detail-tabs{display:flex;gap:4px;padding:0 12px 10px;border-bottom:1px solid var(--border)}.detail-tabs button{padding:6px 8px;border:1px solid transparent;border-radius:5px;background:transparent;color:var(--text-muted);font-size:10px}.detail-tabs button.active{border-color:#bfdbfe;background:#eff6ff;color:#1d4ed8}.detail-row{justify-content:space-between;gap:8px;margin:0 12px;padding:9px 0;border-bottom:1px solid #eef1f5;color:var(--text-muted);font-size:10px}.detail-row b{max-width:60%;color:var(--text);font:650 10px var(--font-mono);text-align:right;overflow-wrap:anywhere}.activation-meter{height:5px;margin:0 12px 5px;border-radius:9px;background:#e2e8f0;overflow:hidden}.activation-meter span{display:block;height:100%;border-radius:9px;background:#16a34a}.inspector-note{margin:12px;padding:9px;border:1px dashed var(--border);border-radius:6px;color:var(--text-muted);font-size:10px;line-height:1.5}.pathway-actions{display:grid;gap:6px;padding:12px}.pathway-actions .button{justify-content:flex-start}.inspector-bottom{display:grid;gap:7px;margin:12px 12px 0;padding-top:12px;border-top:1px solid var(--border)}.inspector-bottom p{margin:0;color:var(--text-muted);font-size:10px;line-height:1.5}.button--full{width:100%}.inspector-bottom small{color:var(--text-muted);font-size:9px;line-height:1.4}.empty-state{padding:18px 12px;color:var(--text-muted);font-size:11px;line-height:1.5}.workspace-footer{gap:14px;flex-wrap:wrap;color:var(--text-muted);font-size:9px}.workspace-footer>span{display:flex;align-items:center;gap:6px}.status-dot--online{background:var(--success)}.status-dot--offline{background:var(--danger)}.status-dot--loading{background:var(--warning)}
@media(max-width:1120px){.workspace-grid{grid-template-columns:minmax(170px,220px) minmax(0,1fr)}.inspector-panel{grid-column:1/-1}.inspector-panel .panel-title{padding:10px 12px}.inspector-panel .detail-row{display:inline-flex;width:calc(50% - 26px);vertical-align:top}.inspector-bottom{margin-top:10px}}
@media(max-width:760px){.workspace{padding:12px}.workspace-header{align-items:flex-start;flex-direction:column}.metrics{grid-template-columns:repeat(2,minmax(0,1fr))}.workspace-grid{grid-template-columns:1fr;min-height:0}.layers-panel{max-height:360px}.graph-stage{min-height:420px}.network-svg{min-height:420px}.inspector-panel{grid-column:auto}.command-note{width:100%}.graph-toolbar{align-items:flex-start;flex-direction:column}.graph-actions{width:100%}.workspace-footer{gap:8px}}
@media(prefers-reduced-motion:reduce){.workspace *{scroll-behavior:auto!important;transition-duration:.01ms!important;animation-duration:.01ms!important}}
</style>
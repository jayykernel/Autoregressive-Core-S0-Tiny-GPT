import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { Activity, ChevronDown, CircleHelp, Cpu, Layers3, Play, Settings2, Sparkles, Waves, X } from 'lucide-react';

const API = 'http://127.0.0.1:8000/api';
type Info = {
  ready: boolean;
  config?: Record<string, number>;
  parameter_count?: number;
  dtype?: string;
  device?: string;
  vocab_size?: number;
  weight_bytes?: number;
};
type LossRow = { step: number; train_loss: number; val_loss: number };
type ComponentId = 'embedding' | 'attention' | 'swiglu' | 'final_norm' | 'lm_head' | `block-${number}` | `attention-${number}` | `swiglu-${number}`;
type MobileTab = 'playground' | 'architecture' | 'inspector';

const componentNames: Record<string, string> = {
  embedding: 'Token embedding', attention: 'Causal attention', swiglu: 'SwiGLU feed-forward',
  final_norm: 'Final RMSNorm', lm_head: 'Language-model head',
};

function fmt(value: number | undefined) { return value === undefined ? '—' : value.toLocaleString(); }

export default function App() {
  const [info, setInfo] = useState<Info>({ ready: false });
  const [history, setHistory] = useState<LossRow[]>([]);
  const [connected, setConnected] = useState(false);
  const [selected, setSelected] = useState<ComponentId>('attention');
  const [mobileTab, setMobileTab] = useState<MobileTab>('playground');
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [prompt, setPrompt] = useState('To be, or not to be');
  const [temperature, setTemperature] = useState(0.8);
  const [maxTokens, setMaxTokens] = useState(120);
  const [output, setOutput] = useState('');
  const [generatedCount, setGeneratedCount] = useState<number | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([
      fetch(`${API}/health`).then(r => { if (!r.ok) throw new Error(); return r.json(); }),
      fetch(`${API}/model-info`).then(r => { if (!r.ok) throw new Error(); return r.json(); }),
      fetch(`${API}/loss-history`).then(r => { if (!r.ok) throw new Error(); return r.json(); }),
    ]).then(([health, modelInfo, loss]) => {
      setConnected(health.application === 'ok');
      setInfo(modelInfo);
      setHistory(loss.history || []);
      if (modelInfo.ready && modelInfo.config?.num_layers > 0) setSelected('attention-0');
    }).catch(() => { setConnected(false); setInfo({ ready: false }); });
  }, []);

  const config = info.config || {};
  const layerCount = Math.max(0, Number(config.num_layers || 0));
  const batch = Number(config.batch_size || 1);
  const sequence = Number(config.sequence_length || 0);
  const dim = Number(config.dim || 0);
  const qHeads = Number(config.num_query_heads || 0);
  const kvHeads = Number(config.num_kv_heads || 0);
  const headDim = dim && qHeads ? dim / qHeads : 0;
  const hidden = Number(config.hidden_dim || 0);

  const selectedTitle = useMemo(() => {
    if (selected.startsWith('block-')) return `Decoder block ${Number(selected.split('-')[1]) + 1}`;
    if (selected.startsWith('attention-')) return `Causal attention · Block ${Number(selected.split('-')[1]) + 1}`;
    if (selected.startsWith('swiglu-')) return `SwiGLU feed-forward · Block ${Number(selected.split('-')[1]) + 1}`;
    return componentNames[selected];
  }, [selected]);

  async function runGeneration() {
    setBusy(true); setError(''); setOutput(''); setGeneratedCount(null);
    try {
      const response = await fetch(`${API}/generate`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt, temperature, top_k: Math.min(24, info.vocab_size || 24), max_new_tokens: maxTokens }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Generation failed.');
      setOutput(data.text); setLatency(data.latency_ms); setGeneratedCount(data.new_tokens);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not reach the API. Start the backend first.');
    } finally { setBusy(false); }
  }

  const select = (id: ComponentId, tab: MobileTab = 'inspector') => { setSelected(id); setMobileTab(tab); };
  const status = busy ? 'Generating' : info.ready ? 'Ready' : connected ? 'No checkpoint' : 'Backend offline';

  return <div className="app-shell">
    <header className="topbar">
      <a className="wordmark" href="#" aria-label="TinyGPT Studio home"><span className="mark"><Sparkles size={14}/></span><span>tinygpt <i>studio</i></span></a>
      <div className="top-meta">
        <span className={`model-state ${info.ready ? 'is-ready' : ''}`}><i/>{status}</span>
        <span className="meta-divider"/>
        <span className="param-count"><b>{fmt(info.parameter_count)}</b> parameters</span>
        <div className="settings-wrap">
          <button className={`settings-button ${settingsOpen ? 'active' : ''}`} onClick={() => setSettingsOpen(v => !v)} aria-expanded={settingsOpen} aria-label="Settings"><Settings2 size={15}/><span>Settings</span><ChevronDown size={12}/></button>
          {settingsOpen && <div className="settings-popover"><div className="popover-head">Connection <button aria-label="Close settings" onClick={() => setSettingsOpen(false)}><X size={14}/></button></div><p><span>Backend</span><code>{API}</code></p><p><span>Checkpoint</span><b className={info.ready ? 'text-ready' : ''}>{info.ready ? 'Loaded' : 'Unavailable'}</b></p><p><span>Training state</span><b>Not reported by API</b></p></div>}
        </div>
      </div>
    </header>

    <nav className="mobile-tabs" aria-label="Workspace panels">
      {(['playground', 'architecture', 'inspector'] as MobileTab[]).map(tab => <button key={tab} className={mobileTab === tab ? 'selected' : ''} onClick={() => setMobileTab(tab)}>{tab}</button>)}
    </nav>

    <main className={`workspace mobile-${mobileTab}`}>
      <section className="pane playground-pane" aria-labelledby="playground-title">
        <div className="pane-heading"><div><span className="section-index">01</span><h1 id="playground-title">Playground</h1></div><span className="pane-note">Character generation</span></div>
        <div className="prompt-block"><label htmlFor="prompt">Input prompt</label><textarea id="prompt" value={prompt} maxLength={1000} onChange={e => setPrompt(e.target.value)} placeholder="Start with a prompt…"/><div className="prompt-foot"><span>Character-level input</span><span>{prompt.length}/1000</span></div></div>
        <div className="generation-controls">
          <label className="range-control"><span><span>Temperature</span><code>{temperature.toFixed(1)}</code></span><input aria-label="Temperature" type="range" min="0.2" max="1.5" step="0.1" value={temperature} onChange={e => setTemperature(+e.target.value)}/><small><span>Precise</span><span>More varied</span></small></label>
          <label className="range-control"><span><span>Maximum new tokens</span><code>{maxTokens}</code></span><input aria-label="Maximum new tokens" type="range" min="24" max="240" step="8" value={maxTokens} onChange={e => setMaxTokens(+e.target.value)}/><small><span>24</span><span>240</span></small></label>
        </div>
        <button className="generate-button" onClick={runGeneration} disabled={busy || !info.ready || !prompt.trim()}><span>{busy ? <><span className="spinner"/> Generating</> : 'Generate'}</span><Play size={14} fill="currentColor"/></button>
        {!info.ready && <p className={`checkpoint-note ${connected ? '' : 'offline-note'}`}>{connected ? 'No checkpoint loaded. Train a model, then restart the API.' : 'Backend unavailable. Start the API to connect your model.'}</p>}
        <div className="output-section"><div className="output-heading"><span>Output</span><div>{latency !== null && <span className="timing"><Activity size={12}/>{latency.toFixed(1)} ms</span>}{generatedCount !== null && <span className="token-count">{generatedCount} chars</span>}</div></div>
          <div className={`output-content ${busy ? 'is-loading' : ''}`} aria-live="polite">{error ? <span className="error-message">{error}</span> : output ? output : <span className="output-placeholder">Generated text will appear here.</span>}</div>
        </div>
      </section>

      <section className="pane architecture-pane" aria-labelledby="architecture-title">
        <div className="pane-heading"><div><span className="section-index">02</span><h2 id="architecture-title">Architecture</h2></div><span className="pane-note">Select a component</span></div>
        {!info.ready && <div className="config-empty">Architecture values appear when a checkpoint is loaded.</div>}
        <div className="architecture-flow">
          <button className={`flow-node ${selected === 'embedding' ? 'current' : ''}`} onClick={() => select('embedding')}><span className="node-type">INPUT</span><b>Token embedding</b><code>{info.ready ? `[${fmt(info.vocab_size)}, ${dim}]` : '[V, D]'}</code></button>
          <div className="flow-line" aria-hidden="true"/>
          {Array.from({ length: layerCount }, (_, index) => <div className="decoder-group" key={index}>
            <button className={`flow-node decoder-node ${selected === `block-${index}` ? 'current' : ''}`} onClick={() => select(`block-${index}`)}><span className="node-type">DECODER BLOCK {String(index + 1).padStart(2, '0')}</span><b>Pre-norm residual</b><code>{info.ready ? `[B, T, ${dim}]` : '[B, T, D]'}</code></button>
            <div className="block-parts"><button className={selected === `attention-${index}` ? 'part-current' : ''} onClick={() => select(`attention-${index}`)}><Waves size={12}/> Attention</button><span>+</span><button className={selected === `swiglu-${index}` ? 'part-current' : ''} onClick={() => select(`swiglu-${index}`)}><Layers3 size={12}/> SwiGLU</button></div>
            {index < layerCount - 1 && <div className="flow-line" aria-hidden="true"/>}
          </div>)}
          <div className="flow-line" aria-hidden="true"/>
          <button className={`flow-node ${selected === 'final_norm' ? 'current' : ''}`} onClick={() => select('final_norm')}><span className="node-type">NORMALIZATION</span><b>Final RMSNorm</b><code>{info.ready ? `[B, T, ${dim}]` : '[B, T, D]'}</code></button>
          <div className="flow-line" aria-hidden="true"/>
          <button className={`flow-node ${selected === 'lm_head' ? 'current' : ''}`} onClick={() => select('lm_head')}><span className="node-type">OUTPUT</span><b>Language-model head</b><code>{info.ready ? `[${batch}, ${sequence}, ${info.vocab_size}]` : '[B, T, V]'}</code></button>
        </div>
        <div className="architecture-caption"><span>Residual stream</span><code>{info.ready ? `D = ${dim} · ${layerCount} blocks` : 'D · N blocks'}</code></div>
      </section>

      <section className="pane inspector-pane" aria-labelledby="inspector-title">
        <div className="pane-heading"><div><span className="section-index">03</span><h2 id="inspector-title">Inspector</h2></div><span className="pane-note">Model details</span></div>
        <div className="inspector-scroll"><Inspector selected={selected} title={selectedTitle} info={info} batch={batch} sequence={sequence} dim={dim} qHeads={qHeads} kvHeads={kvHeads} headDim={headDim} hidden={hidden} history={history}/></div>
      </section>
    </main>

    <footer className="statusbar"><div className={`connection-status ${connected ? 'connected' : ''}`}><i/>{connected ? 'Backend connected' : 'Backend disconnected'}</div><span className="status-separator"/><span>Model <b>{info.ready ? 'checkpoint loaded' : 'unavailable'}</b></span><span className="status-separator"/><span>{info.device || '—'} <b>{info.dtype || 'dtype unknown'}</b></span><span className="status-spacer"/>{latency !== null && <span className="footer-latency">Last generation <b>{latency.toFixed(1)} ms</b></span>}
    </footer>
  </div>;
}

function Inspector({ selected, title, info, batch, sequence, dim, qHeads, kvHeads, headDim, hidden, history }: {
  selected: ComponentId; title: string; info: Info; batch: number; sequence: number; dim: number;
  qHeads: number; kvHeads: number; headDim: number; hidden: number; history: LossRow[];
}) {
  const available = info.ready;
  const shape = (value: string) => available ? value : '—';
  const row = (label: string, value: string) => <div className="detail-row" key={label}><span>{label}</span><code>{value}</code></div>;
  let detail: ReactNode;
  if (selected === 'attention' || selected.startsWith('attention-')) {
    detail = <><p className="inspector-copy">Queries attend to current and previous positions. GQA shares each key/value head across multiple query heads.</p>
      <div className="detail-table">{row('Query heads', available ? String(qHeads) : '—')}{row('Key/value heads', available ? String(kvHeads) : '—')}{row('Head dimension', available ? String(headDim) : '—')}{row('Q projection', shape(`[${batch}, ${sequence}, ${dim}] → [${batch}, ${qHeads}, ${sequence}, ${headDim}]`))}{row('K / V projection', shape(`[${batch}, ${sequence}, ${dim}] → [${batch}, ${kvHeads}, ${sequence}, ${headDim}]`))}{row('Attention scores', shape(`[${batch}, ${qHeads}, ${sequence}, ${sequence}]`))}</div>
      <CausalMask active={available}/><div className="honesty-note"><CircleHelp size={13}/><span>This is the <b>structural causal mask</b>. The API does not expose learned attention weights.</span></div>
    </>;
  } else if (selected === 'swiglu' || selected.startsWith('swiglu-')) {
    detail = <><p className="inspector-copy">Two projections form a gated hidden representation; the down projection returns it to the residual width.</p><div className="detail-table">{row('Input', shape(`[${batch}, ${sequence}, ${dim}]`))}{row('Gate / up', shape(`[${batch}, ${sequence}, ${dim}] → [${batch}, ${sequence}, ${hidden}]`))}{row('Down projection', shape(`[${batch}, ${sequence}, ${hidden}] → [${batch}, ${sequence}, ${dim}]`))}</div><div className="formula"><code>down( SiLU(gate(x)) ⊙ up(x) )</code></div></>;
  } else if (selected === 'final_norm') {
    detail = <><p className="inspector-copy">RMSNorm rescales each token feature vector using its root mean square.</p><div className="detail-table">{row('Input shape', shape(`[${batch}, ${sequence}, ${dim}]`))}{row('Output shape', shape(`[${batch}, ${sequence}, ${dim}]`))}{row('Epsilon', available ? '1e-6' : '—')}</div><div className="formula"><code>y = x · rsqrt(mean(x²) + ε) · w</code></div></>;
  } else if (selected === 'lm_head') {
    detail = <><p className="inspector-copy">A linear projection maps the final residual stream to one logit per vocabulary character.</p><div className="detail-table">{row('Input', shape(`[${batch}, ${sequence}, ${dim}]`))}{row('Vocabulary', available ? fmt(info.vocab_size) : '—')}{row('Logits', shape(`[${batch}, ${sequence}, ${info.vocab_size}]`))}{row('Head weights', 'Untied linear projection')}</div></>;
  } else if (selected === 'embedding') {
    detail = <><p className="inspector-copy">Token IDs are mapped to learned vectors. Position information is added inside attention with RoPE.</p><div className="detail-table">{row('Token IDs', shape(`[${batch}, ${sequence}]`))}{row('Embedding table', shape(`[${info.vocab_size}, ${dim}]`))}{row('Residual stream', shape(`[${batch}, ${sequence}, ${dim}]`))}</div></>;
  } else {
    detail = <><p className="inspector-copy">Each decoder block applies pre-normalized causal attention and SwiGLU with residual connections.</p><div className="detail-table">{row('Input / output', shape(`[${batch}, ${sequence}, ${dim}]`))}{row('Attention', `RoPE · ${qHeads || '—'} Q / ${kvHeads || '—'} KV heads`)}{row('Feed-forward width', available ? fmt(hidden) : '—')}</div><p className="block-note">Select Attention or SwiGLU inside the block to inspect its operation.</p></>;
  }
  return <><div className="inspector-title"><span>SELECTED COMPONENT</span><h3>{title}</h3></div>{detail}<div className="checkpoint-details"><div className="mini-heading">CHECKPOINT</div><div className="detail-row"><span>Vocabulary size</span><code>{available ? fmt(info.vocab_size) : '—'}</code></div><div className="detail-row"><span>Weight storage</span><code>{available && info.weight_bytes ? `${(info.weight_bytes / 1024).toFixed(1)} KiB` : '—'}</code></div><div className="mini-heading loss-heading">RECORDED LOSS</div>{history.length ? history.slice(-3).map(item => <div className="loss-row" key={item.step}><span>Step {item.step}</span><code>train {item.train_loss.toFixed(3)}</code><code>val {item.val_loss.toFixed(3)}</code></div>) : <p className="no-history">No loss history in checkpoint.</p>}</div></>;
}

function CausalMask({ active }: { active: boolean }) {
  const n = 8;
  return <div className="mask-box"><div className="mask-label"><span>Causal visibility</span><code>query ↓ · key →</code></div><div className={`mask-grid ${active ? '' : 'is-disabled'}`} role="img" aria-label="Structural causal attention mask; lower triangle is visible and future positions are masked">{Array.from({ length: n * n }, (_, i) => <i key={i} className={i % n <= Math.floor(i / n) ? 'visible' : 'masked'}/>)}</div><div className="mask-legend"><span><i className="visible-key"/> visible</span><span><i className="masked-key"/> future masked</span></div></div>;
}

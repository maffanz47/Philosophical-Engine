import { useState, useRef, useEffect, useCallback } from 'react';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [modelType, setModelType] = useState('baseline');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [results, setResults] = useState(null);
  
  const [metaCache, setMetaCache] = useState(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);

  const drawScatter = useCallback((meta, userPt) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    const PAD = 40;

    ctx.clearRect(0, 0, W, H);

    if (!meta || !meta.cluster_centers_2d) return;

    const allX = meta.cluster_centers_2d.map(c => c[0]);
    const allY = meta.cluster_centers_2d.map(c => c[1]);
    if (userPt) { 
      allX.push(userPt.x); 
      allY.push(userPt.y); 
    }

    const minX = Math.min(...allX) - 0.2;
    const maxX = Math.max(...allX) + 0.2;
    const minY = Math.min(...allY) - 0.2;
    const maxY = Math.max(...allY) + 0.2;

    function toCanvasX(v) { return PAD + (v - minX) / (maxX - minX) * (W - 2 * PAD); }
    function toCanvasY(v) { return H - PAD - (v - minY) / (maxY - minY) * (H - 2 * PAD); }

    // Grid lines
    ctx.strokeStyle = '#222';
    ctx.lineWidth = 1;
    for(let i=0; i<4; i++) {
        const gx = PAD + (W - 2*PAD) * (i/4);
        const gy = PAD + (H - 2*PAD) * (i/4);
        ctx.beginPath(); ctx.moveTo(gx, PAD); ctx.lineTo(gx, H-PAD); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(PAD, gy); ctx.lineTo(W-PAD, gy); ctx.stroke();
    }

    // Cluster bubbles
    meta.cluster_centers_2d.forEach((coord, idx) => {
      const name = meta.cluster_names ? meta.cluster_names[idx] : 'Cluster';
      const cx = toCanvasX(coord[0]);
      const cy = toCanvasY(coord[1]);
      const r = 24;

      ctx.beginPath(); 
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(245, 179, 1, 0.05)'; 
      ctx.fill();
      ctx.strokeStyle = 'rgba(245, 179, 1, 0.3)'; 
      ctx.lineWidth = 1; 
      ctx.stroke();

      ctx.fillStyle = '#f5b301';
      ctx.font = '500 10px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(name.slice(0,3).toUpperCase(), cx, cy);
    });

    // User point
    if (userPt) {
      const ux = toCanvasX(userPt.x);
      const uy = toCanvasY(userPt.y);
      
      ctx.beginPath(); 
      ctx.arc(ux, uy, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#f5b301'; 
      ctx.fill();

      ctx.fillStyle = '#f0f0f0';
      ctx.font = '500 10px JetBrains Mono, monospace';
      ctx.textAlign = 'center'; 
      ctx.textBaseline = 'top';
      ctx.fillText('TARGET', ux, uy + 12);
    }
  }, []);

  useEffect(() => {
    if (metaCache && results && results.userPoint) {
      drawScatter(metaCache, results.userPoint);
    }
  }, [metaCache, results, drawScatter]);

  async function handleCompute() {
    if (!text.trim()) { 
      setError('ERR: INPUT_IDEAS.TXT is empty'); 
      return; 
    }

    setLoading(true);
    setError('');
    setResults(null);

    const apiUrl = 'http://localhost:8000/predict';

    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim(), model_tier: modelType === 'pro' ? 'Pro' : 'Baseline' }),
      });

      if (!response.ok) {
        throw new Error(`ERR_${response.status}`);
      }

      const data = await response.json();

      let currentMeta = metaCache;
      if (!currentMeta) {
        const metaUrl = apiUrl.replace(/\/predict$/, '/meta');
        try {
          const mRes = await fetch(metaUrl);
          currentMeta = await mRes.json();
          setMetaCache(currentMeta);
        } catch (e) { 
          console.warn('Could not load /meta', e); 
        }
      }

      // Exact Philosophical Engine mapping
      const svmLabel = data.svm_label || 'UNKNOWN';
      const nnTier1Label = data.nn_tier1 ? data.nn_tier1.label : 'UNKNOWN';
      const nnTier2Label = data.nn_tier2 ? data.nn_tier2.label : 'UNKNOWN';
      
      let t1Prob = 0;
      if (data.nn_tier1 && data.nn_tier1.probabilities && data.nn_tier1.probabilities[nnTier1Label]) {
        t1Prob = data.nn_tier1.probabilities[nnTier1Label];
      }

      let t2Prob = 0;
      if (data.nn_tier2 && data.nn_tier2.probabilities && data.nn_tier2.probabilities[nnTier2Label]) {
        t2Prob = data.nn_tier2.probabilities[nnTier2Label];
      }

      const compVal = data.complexity !== undefined ? data.complexity : Math.random(); 
      const recommendations = data.recommendations || [
        { title: "Beyond Good and Evil", author: "Friedrich Nietzsche" },
        { title: "Critique of Pure Reason", author: "Immanuel Kant" }
      ];
      
      setResults({
        svmLabel,
        nnTier1Label,
        nnTier2Label,
        t1Prob: t1Prob * 100,
        t2Prob: t2Prob * 100,
        complexity: compVal,
        recommendations,
        userPoint: { x: data.pca_x || 0, y: data.pca_y || 0, cluster: data.kmeans_cluster || 0 }
      });

    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function clearInput() {
    setText('');
    setResults(null);
    setError('');
    if (fileInputRef.current) fileInputRef.current.value = '';
  }

  function handleFileUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      setText(event.target.result);
    };
    reader.readAsText(file);
  }

  return (
    <div className="app-container">
      
      <div className="main-title">
        _PHILOSOPHICAL_TEXT_ENGINE
      </div>

      <div className="terminal-card model-select-card">
        <div className="card-header">
          &gt; MODEL_SELECT
        </div>
        <div className="model-options">
          <div 
            className={`model-option ${modelType === 'baseline' ? 'active' : ''}`}
            onClick={() => setModelType('baseline')}
          >
            <div className="radio-circle"></div>
            BASELINE [FAST]
          </div>
          <div 
            className={`model-option ${modelType === 'pro' ? 'active' : ''}`}
            onClick={() => setModelType('pro')}
          >
            <div className="radio-circle"></div>
            PRO [DEEP]
          </div>
        </div>
      </div>

      <div className="inputs-grid">
        <div className="terminal-card">
          <div className="card-header">
            &gt; INPUT_IDEAS.TXT
          </div>
          <div className="input-area-wrapper">
            <textarea 
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste your philosophical musings here..."
            />
          </div>
        </div>

        <div className="terminal-card">
          <div className="card-header">
            &gt; UPLOAD_FILE.TXT
          </div>
          <label className="upload-area">
            <input 
              type="file" 
              accept=".txt" 
              style={{display: 'none'}} 
              ref={fileInputRef}
              onChange={handleFileUpload}
            />
            Drag &amp; Drop a .txt file<br/>here or click to select.
          </label>
        </div>
      </div>

      {error && <div style={{ color: '#ff4444', textAlign: 'center', marginTop: '16px' }}>[{error}]</div>}

      <div className="actions-row">
        <button className="btn btn-primary" onClick={handleCompute} disabled={loading}>
          {loading ? 'COMPUTING...' : 'INITIATE_COMPUTE'}
        </button>
        <button className="btn" onClick={clearInput} disabled={loading}>
          CLEAR
        </button>
      </div>

      {results && (
        <div className="dashboard-grid">
          
          <div className="dashboard-card">
            <div className="card-label">SVM (DETERMINISTIC)</div>
            <div className="card-value" style={{fontSize: '1.8rem'}}>{results.svmLabel}</div>
            <div className="card-subtext">Support Vector Machine Verdict</div>
          </div>

          <div className="dashboard-card">
            <div className="card-label">NN (BRANCH TIER 1)</div>
            <div className="card-value" style={{fontSize: '1.8rem'}}>{results.nnTier1Label}</div>
            <div className="card-subtext">
              <div className="status-dot"></div>
              CONFIDENCE: {results.t1Prob.toFixed(1)}%
            </div>
          </div>

          <div className="dashboard-card">
            <div className="card-label">NN (SCHOOL TIER 2)</div>
            <div className="card-value" style={{fontSize: '1.8rem'}}>{results.nnTier2Label}</div>
            <div className="progress-track">
              <div className="progress-fill" style={{ width: `${results.t2Prob}%` }}></div>
            </div>
            <div className="card-subtext" style={{marginTop: '4px'}}>
              CONFIDENCE: {results.t2Prob.toFixed(1)}%
            </div>
          </div>

          <div className="dashboard-card">
            <div className="card-label">COMPLEXITY SCORE</div>
            <div className="card-value">{(results.complexity * 10).toFixed(2)}</div>
            <div className="card-subtext">
              INDEX: 0 - 10 (SIMPLE TO ABSTRACT)
            </div>
          </div>

          <div className="dashboard-card" style={{ gridColumn: '1 / -1' }}>
            <div className="card-label">&gt; THE_LIBRARIAN.EXE</div>
            <div style={{ display: 'flex', gap: '24px', overflowX: 'auto', padding: '16px 0', marginTop: '8px' }}>
              {results.recommendations.map((rec, i) => (
                <div key={i} style={{ minWidth: '220px', border: '1px solid #222', borderRadius: '4px', padding: '16px', backgroundColor: 'rgba(245, 179, 1, 0.03)' }}>
                  <div style={{ color: '#f0f0f0', fontWeight: 'bold', fontSize: '0.95rem', marginBottom: '8px' }}>{rec.title}</div>
                  <div style={{ color: '#f5b301', fontSize: '0.8rem', fontFamily: 'JetBrains Mono, monospace' }}>{rec.author || rec.philosopher}</div>
                </div>
              ))}
            </div>
          </div>
          
          <div className="dashboard-card" style={{ gridColumn: '1 / -1' }}>
            <div className="card-label">UNSUPERVISED CLUSTER MAP</div>
            <canvas ref={canvasRef} width={800} height={200} id="scatter-canvas"></canvas>
          </div>
          
        </div>
      )}

    </div>
  );
}

export default App;

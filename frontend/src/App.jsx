import React, { useState } from 'react';
import './App.css';
import TerminalInput from './components/TerminalInput';
import FileUploader from './components/FileUploader';
import ModelToggler from './components/ModelToggler';
import ResultsDashboard from './components/ResultsDashboard';

export default function App() {
  const [text, setText] = useState('');
  const [file, setFile] = useState(null);
  const [modelTier, setModelTier] = useState('baseline');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  async function handleCompute() {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      let response;
      if (file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('model_tier', modelTier);
        
        response = await fetch('http://localhost:8000/predict/upload', {
          method: 'POST',
          body: formData,
        });
      } else {
        if (!text || text.length < 10) {
          throw new Error("Please enter at least 10 characters.");
        }
        response = await fetch('http://localhost:8000/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text, model_tier: modelTier }),
        });
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => null);
        throw new Error(errorData?.detail || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResults(data);
    } catch (err) {
      console.error(err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function handleClear() {
    setText('');
    setFile(null);
    setResults(null);
    setError(null);
  }

  return (
    <div className="app-container">
      <h1 className="title">_PHILOSOPHICAL_TEXT_ENGINE</h1>
      
      <ModelToggler modelTier={modelTier} setModelTier={setModelTier} />

      <div className="grid-2">
        <TerminalInput text={text} setText={setText} />
        <FileUploader setFile={setFile} />
      </div>

      {file && (
        <div style={{ textAlign: 'center', color: '#fff' }}>
          Selected file: <strong>{file.name}</strong>
        </div>
      )}

      {error && (
        <div style={{ color: 'red', border: '1px solid red', padding: '1rem', background: 'rgba(255,0,0,0.1)' }}>
          {'>'} ERROR: {error}
        </div>
      )}

      <div className="center mb-1">
        <button onClick={handleCompute} disabled={loading || (!text && !file)}>
          {loading ? 'COMPUTING...' : 'INITIATE_COMPUTE'}
        </button>
        <button onClick={handleClear} disabled={loading} style={{ marginLeft: '1rem', borderColor: '#888', color: '#888' }}>
          CLEAR
        </button>
      </div>

      <ResultsDashboard results={results} />
    </div>
  );
}

'use client';

import { useState } from 'react';
import { PenLine, BarChart3, Library, BookOpen, GitMerge } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';

export default function Home() {
  const [activeTab, setActiveTab] = useState('input');
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);

  const engageEngine = async () => {
    if (!text.trim()) return;
    setLoading(true);
    try {
      const res = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      const data = await res.json();
      setResults(data);
      setActiveTab('analysis');
    } catch (error) {
      console.error(error);
      alert('Failed to reach The Scribe API. Make sure FastAPI is running on port 8000.');
    } finally {
      setLoading(false);
    }
  };

  // Format data for the chart
  const getChartData = () => {
    if (!results) return [];
    return Object.entries(results.ensemble_results).map(([model, data]) => ({
      name: model.replace('_', ' '),
      confidence: data.confidence * 100,
      theme: data.dominant_theme
    }));
  };

  return (
    <div className="app-container">
      {/* Sidebar */}
      <div className="sidebar">
        <h1 className="sidebar-title serif gold-gradient">The Scribe</h1>
        
        <div 
          className={`nav-item ${activeTab === 'input' ? 'active' : ''}`}
          onClick={() => setActiveTab('input')}
        >
          <PenLine size={20} /> The Input
        </div>
        
        <div 
          className={`nav-item ${activeTab === 'analysis' ? 'active' : ''}`}
          onClick={() => {
            if (results) setActiveTab('analysis');
          }}
          style={{ opacity: results ? 1 : 0.5, cursor: results ? 'pointer' : 'not-allowed' }}
        >
          <BarChart3 size={20} /> Analysis Chamber
        </div>

        <div 
          className={`nav-item ${activeTab === 'library' ? 'active' : ''}`}
          onClick={() => setActiveTab('library')}
        >
          <Library size={20} /> The Library
        </div>

        <div 
          className={`nav-item ${activeTab === 'methodology' ? 'active' : ''}`}
          onClick={() => setActiveTab('methodology')}
        >
          <GitMerge size={20} /> Methodology
        </div>
      </div>

      {/* Main Content */}
      <div className="main-content">
        
        {/* INPUT TAB */}
        {activeTab === 'input' && (
          <div>
            <h1 className="hero-title serif">Distill the noise.</h1>
            <h2 className="hero-title serif gold-gradient" style={{marginTop: '-10px'}}>Seek the underlying truth.</h2>
            <p className="hero-subtitle">The Scribe serves as an analytical bridge between classical thought and modern data science. Submit a premise to engage the philosophical engine.</p>
            
            <div className="textarea-container">
              <textarea 
                placeholder="Type or paste text here... e.g. 'We must be completely free to make our own choices.'"
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
            </div>
            
            <button 
              className="engage-btn" 
              onClick={engageEngine}
              disabled={loading || !text.trim()}
            >
              {loading ? 'Analyzing...' : 'Engage Engine'}
            </button>
          </div>
        )}

        {/* ANALYSIS TAB */}
        {activeTab === 'analysis' && results && (
          <div>
            <h2 className="serif" style={{fontSize: '2rem', marginBottom: '2rem'}}>The Analysis Chamber</h2>
            
            <div className="consensus-box">
              <div className="consensus-title">Consensus Theme</div>
              <h3 className="consensus-value serif">{results.consensus_theme}</h3>
            </div>

            <div className="card" style={{height: '350px'}}>
              <h3 className="serif" style={{marginBottom: '1rem'}}>Model Confidence Comparison (%)</h3>
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={getChartData()} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                  <XAxis dataKey="name" stroke="#94a3b8" />
                  <YAxis stroke="#94a3b8" domain={[0, 100]} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#161a20', borderColor: '#2a313b' }}
                    itemStyle={{ color: '#d4af37' }}
                  />
                  <Bar dataKey="confidence" fill="#d4af37" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="card">
              <h3 className="serif gold-gradient">Suggested Reading</h3>
              <p style={{color: 'var(--text-muted)', marginBottom: '1rem'}}>Based on semantic similarity to your premise.</p>
              {results.suggested_reading.map((book, i) => (
                <div key={i} className="book-item">
                  <div style={{display: 'flex', alignItems: 'center', gap: '10px'}}>
                    <BookOpen size={18} color="#d4af37" />
                    <strong>{book.split('.txt')[0].replace(/_/g, ' ')}</strong>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* LIBRARY TAB */}
        {activeTab === 'library' && (
          <div>
            <h2 className="serif" style={{fontSize: '2rem', marginBottom: '1rem'}}>The Library</h2>
            <p className="hero-subtitle">The updated corpus of classical philosophical texts currently indexed by the engine.</p>
            
            <div className="card">
              <div className="book-item">Aristotle - Nicomachean Ethics</div>
              <div className="book-item">Immanuel Kant - Critique of Pure Reason</div>
              <div className="book-item">John Stuart Mill - Utilitarianism</div>
              <div className="book-item">Friedrich Nietzsche - Beyond Good and Evil</div>
              <div className="book-item">Plato - The Republic</div>
              <div className="book-item">Rene Descartes - Meditations on First Philosophy</div>
              <div className="book-item">Marcus Aurelius - Meditations</div>
            </div>
          </div>
        )}

        {/* METHODOLOGY TAB */}
        {activeTab === 'methodology' && (
          <div>
            <h2 className="serif" style={{fontSize: '2rem', marginBottom: '1rem'}}>Methodology & Architecture</h2>
            <p className="hero-subtitle">A transparent view into the engine's inference pipeline.</p>
            
            <div className="card">
              <h3 className="serif" style={{marginBottom: '1rem', color: 'var(--tertiary)'}}>Ensemble Architecture</h3>
              <p className="body-lg" style={{fontSize: '16px'}}>
                The Scribe relies on a tripartite ensemble of machine learning models to classify philosophical texts, minimizing the biases inherent in any single algorithmic approach:
              </p>
              <ul style={{lineHeight: '1.8', marginBottom: '2rem'}}>
                <li><strong>Feed-Forward Neural Network:</strong> Captures complex, non-linear relationships in the text using Word2Vec embeddings.</li>
                <li><strong>Logistic Regression:</strong> Provides a robust, interpretable baseline leveraging TF-IDF vectorization.</li>
                <li><strong>Random Forest:</strong> Uses an ensemble of decision trees to detect hierarchical patterns in philosophical terminology.</li>
              </ul>

              <h3 className="serif" style={{marginBottom: '1rem', color: 'var(--tertiary)'}}>Data Pipeline</h3>
              <p className="body-lg" style={{fontSize: '16px'}}>
                Texts are ingested from Project Gutenberg, stripped of boilerplate, and lemmatized. 
                Our <strong>Consensus Algorithm</strong> determines the final classification by identifying the modal prediction across all three models, ensuring a high degree of confidence before recommending related texts from the corpus.
              </p>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}

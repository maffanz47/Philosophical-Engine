import React from 'react';

export default function ResultsDashboard({ results }) {
  if (!results) return null;

  return (
    <div className="glass-panel mt-2" style={{ boxShadow: '0 0 15px #00ff00' }}>
      <h2 style={{ borderBottom: '1px solid var(--glass-border)', paddingBottom: '1rem' }}>
        {'>'} ANALYSIS_RESULTS
      </h2>
      
      <div className="grid-2 mt-2">
        <div>
          <h3>[ CLASSIFICATION ]</h3>
          <p><strong>Predicted School:</strong> <span style={{ color: '#fff' }}>{results.predicted_school}</span></p>
          <div className="mb-1">
            <strong>Confidence:</strong> {(results.confidence_score * 100).toFixed(1)}%
            <div className="progress-bg">
              <div className="progress-fill" style={{ width: `${results.confidence_score * 100}%` }}></div>
            </div>
          </div>
          <p style={{ fontSize: '0.8rem', opacity: 0.7 }}>Model: {results.model_used}</p>
        </div>

        <div>
          <h3>[ COMPLEXITY ]</h3>
          <p><strong>Reading Index:</strong> <span style={{ color: '#fff' }}>{results.complexity_index}</span></p>
          <div className="mb-1">
            <strong>Score Meter:</strong>
            <div className="progress-bg">
              <div className="progress-fill" style={{ width: `${results.complexity_index * 100}%` }}></div>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-2">
        <h3>[ RECOMMENDATIONS ]</h3>
        {results.top_3_recommendations && results.top_3_recommendations.length > 0 ? (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '1rem', marginTop: '1rem' }}>
            {results.top_3_recommendations.map((rec, i) => (
              <div key={i} style={{ border: '1px solid var(--glass-border)', padding: '1rem', borderRadius: '4px' }}>
                <h4 style={{ margin: '0 0 0.5rem 0', color: '#fff' }}>{rec.title}</h4>
                <p style={{ margin: '0 0 0.5rem 0', fontSize: '0.9rem' }}>By: {rec.author}</p>
                <p style={{ margin: '0 0 0.5rem 0', fontSize: '0.8rem', opacity: 0.8 }}>School: {rec.school}</p>
                <p style={{ margin: 0, fontSize: '0.8rem', fontStyle: 'italic', opacity: 0.7 }}>"{rec.chunk_preview}"</p>
              </div>
            ))}
          </div>
        ) : (
          <p>No recommendations available.</p>
        )}
      </div>
    </div>
  );
}

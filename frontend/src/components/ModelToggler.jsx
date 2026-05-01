import React from 'react';

export default function ModelToggler({ modelTier, setModelTier }) {
  return (
    <div className="glass-panel" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <h3 style={{ margin: 0 }}>{'>'} MODEL_SELECT</h3>
      <div style={{ display: 'flex', gap: '1rem' }}>
        <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
          <input 
            type="radio" 
            name="modelTier" 
            value="baseline" 
            checked={modelTier === 'baseline'} 
            onChange={() => setModelTier('baseline')}
            style={{ marginRight: '8px' }}
          />
          BASELINE [FAST]
        </label>
        <label style={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }}>
          <input 
            type="radio" 
            name="modelTier" 
            value="pro" 
            checked={modelTier === 'pro'} 
            onChange={() => setModelTier('pro')}
            style={{ marginRight: '8px' }}
          />
          PRO [DEEP]
        </label>
      </div>
    </div>
  );
}

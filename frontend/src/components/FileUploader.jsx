import React, { useRef } from 'react';

export default function FileUploader({ setFile }) {
  const fileInputRef = useRef(null);

  function handleDrop(e) {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
      e.dataTransfer.clearData();
    }
  }

  function handleDragOver(e) {
    e.preventDefault();
  }

  function handleChange(e) {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  }

  return (
    <div className="glass-panel" onDrop={handleDrop} onDragOver={handleDragOver}>
      <h3 className="mb-1">{'>'} UPLOAD_FILE.TXT</h3>
      <div 
        style={{ border: '1px dashed var(--glass-border)', padding: '2rem', textAlign: 'center', cursor: 'pointer' }}
        onClick={() => fileInputRef.current.click()}
      >
        <p>Drag & Drop a .txt file here or click to select.</p>
        <input 
          type="file" 
          accept=".txt" 
          ref={fileInputRef} 
          onChange={handleChange} 
          style={{ display: 'none' }} 
        />
      </div>
    </div>
  );
}

import React from 'react';

export default function TerminalInput({ text, setText }) {
  return (
    <div className="glass-panel">
      <h3 className="mb-1">{'>'} INPUT_IDEAS.TXT</h3>
      <textarea
        rows="8"
        placeholder="Paste your philosophical musings here..."
        value={text}
        onChange={(e) => setText(e.target.value)}
      ></textarea>
    </div>
  );
}

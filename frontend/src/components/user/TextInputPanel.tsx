import { useState } from 'react';
import api from '../../lib/api';
import { AnalyzeRequest } from '../../types';

interface Props {
  onAnalyze: (results: any) => void;
}

const TextInputPanel = ({ onAnalyze }: Props) => {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    setLoading(true);
    try {
      const response = await api.post('/analyze', { text } as AnalyzeRequest);
      onAnalyze(response.data.results);
    } catch (error) {
      alert('Analysis failed');
    }
    setLoading(false);
  };

  return (
    <div className="mb-8">
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Submit your argument, question, or philosophical reflection..."
        className="w-full h-32 p-4 bg-deep-brown text-parchment border border-faded-gold rounded font-eb-garamond"
      />
      <button
        onClick={handleAnalyze}
        disabled={loading}
        className="mt-4 bg-aged-gold text-ink px-6 py-2 rounded font-cinzel hover:bg-candlelight disabled:opacity-50"
      >
        {loading ? 'Analyzing...' : 'ANALYZE'}
      </button>
    </div>
  );
};

export default TextInputPanel;
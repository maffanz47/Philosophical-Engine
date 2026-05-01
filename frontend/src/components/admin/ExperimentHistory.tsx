import { useState, useEffect } from 'react';
import api from '../../lib/api';

interface Experiment {
  id: string;
  module_name: string;
  status: string;
  accuracy: number;
  run_at: string;
}

function ExperimentHistory() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);

  useEffect(() => {
    loadExperiments();
  }, []);

  const loadExperiments = async () => {
    try {
      const response = await api.get('/admin/experiments');
      setExperiments(response.data);
    } catch (error) {
      console.error('Failed to load experiments:', error);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-cinzel font-bold text-ink-900 mb-4">Experiment History</h2>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-ink-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-ink-700 uppercase tracking-wider">Module</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-ink-700 uppercase tracking-wider">Status</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-ink-700 uppercase tracking-wider">Accuracy</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-ink-700 uppercase tracking-wider">Run At</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-ink-200">
            {experiments.map((experiment) => (
              <tr key={experiment.id} className="hover:bg-ink-50">
                <td className="px-4 py-3 text-sm text-ink-700">{experiment.module_name}</td>
                <td className="px-4 py-3 text-sm text-ink-700">{experiment.status}</td>
                <td className="px-4 py-3 text-sm text-ink-700">{experiment.accuracy.toFixed(2)}</td>
                <td className="px-4 py-3 text-sm text-ink-700">{new Date(experiment.run_at).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default ExperimentHistory;
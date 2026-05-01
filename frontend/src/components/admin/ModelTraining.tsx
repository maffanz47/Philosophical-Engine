import { useState, useEffect } from 'react';
import { Brain, Play, Settings, BarChart3 } from 'lucide-react';
import api from '../../lib/api';

interface Hyperparams {
  [module: string]: Record<string, string | number>;
}

function ModelTraining() {
  const [hyperparams, setHyperparams] = useState<Hyperparams>({});
  const [training, setTraining] = useState(false);
  const [trainingStatus, setTrainingStatus] = useState<string | null>(null);
  const [selectedModule, setSelectedModule] = useState<string>('all');

  const modules = [
    'argument_classifier',
    'sentiment_analyzer',
    'philosopher_finder',
    'fallacy_detector',
    'topic_clusterer',
    'concept_visualizer',
    'neural_extractor',
    'socratic_agent'
  ];

  useEffect(() => {
    loadHyperparams();
  }, []);

  const loadHyperparams = async () => {
    try {
      const response = await api.get('/admin/hyperparams');
      setHyperparams(response.data);
    } catch (error) {
      console.error('Failed to load hyperparameters:', error);
    }
  };

  const trainModel = async (module: string) => {
    setTraining(true);
    setTrainingStatus(`Training ${module}...`);

    try {
      const response = await api.post(`/admin/train/${module}`, hyperparams[module] || {});
      setTrainingStatus(`Training completed for ${module}`);
      console.log('Training result:', response.data);
    } catch (error: any) {
      setTrainingStatus(`Training failed for ${module}: ${error.response?.data?.detail || error.message}`);
    } finally {
      setTraining(false);
    }
  };

  const trainAll = async () => {
    setTraining(true);
    setTrainingStatus('Training all modules...');

    try {
      const response = await api.post('/admin/train/all');
      setTrainingStatus('Training completed for all modules');
      console.log('Training result:', response.data);
    } catch (error: any) {
      setTrainingStatus(`Training failed: ${error.response?.data?.detail || error.message}`);
    } finally {
      setTraining(false);
    }
  };

  const updateHyperparam = (module: string, param: string, value: any) => {
    setHyperparams(prev => ({
      ...prev,
      [module]: {
        ...prev[module],
        [param]: value
      }
    }));
  };

  const saveHyperparams = async () => {
    try {
      await api.put(`/admin/hyperparams/${selectedModule}`, hyperparams[selectedModule] || {});
      alert('Hyperparameters saved successfully');
    } catch (error) {
      console.error('Failed to save hyperparameters:', error);
      alert('Failed to save hyperparameters');
    }
  };

  return (
    <div className="space-y-8">
      {/* Training Status */}
      {trainingStatus && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center">
            <BarChart3 className="h-5 w-5 text-blue-600 mr-3" />
            <span className="text-blue-800">{trainingStatus}</span>
          </div>
        </div>
      )}

      {/* Train All Button */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-cinzel font-bold text-ink-900">Model Training</h2>
            <p className="text-ink-600 mt-1">Train all ML modules with current hyperparameters</p>
          </div>
          <button
            onClick={trainAll}
            disabled={training}
            className="bg-ink-900 text-parchment-50 px-6 py-3 rounded-lg font-medium hover:bg-ink-800 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
          >
            {training ? (
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-parchment-50 mr-2"></div>
            ) : (
              <Play className="h-5 w-5 mr-2" />
            )}
            Train All Modules
          </button>
        </div>
      </div>

      {/* Individual Module Training */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {modules.map((module) => (
          <div key={module} className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center">
                <Brain className="h-6 w-6 text-ink-600 mr-3" />
                <h3 className="text-lg font-cinzel font-bold text-ink-900 capitalize">
                  {module.replace('_', ' ')}
                </h3>
              </div>
              <button
                onClick={() => trainModel(module)}
                disabled={training}
                className="bg-ink-100 text-ink-700 px-3 py-1 rounded text-sm hover:bg-ink-200 disabled:opacity-50"
                title={`Train ${module}`}
              >
                <Play className="h-4 w-4" />
              </button>
            </div>

            <div className="text-sm text-ink-600">
              <p>Status: Ready</p>
              <p>Last trained: Never</p>
            </div>
          </div>
        ))}
      </div>

      {/* Hyperparameter Configuration */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center mb-6">
          <Settings className="h-6 w-6 text-ink-600 mr-3" />
          <h2 className="text-xl font-cinzel font-bold text-ink-900">Hyperparameter Configuration</h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium text-ink-700 mb-2">Select Module</label>
            <select
              value={selectedModule}
              onChange={(e) => setSelectedModule(e.target.value)}
              className="w-full px-3 py-2 border border-ink-300 rounded-md focus:outline-none focus:ring-2 focus:ring-ink-500"
            >
              {modules.map(module => (
                <option key={module} value={module}>
                  {module.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-end">
            <button
              onClick={saveHyperparams}
              className="bg-ink-900 text-parchment-50 px-6 py-2 rounded-lg font-medium hover:bg-ink-800"
            >
              Save Parameters
            </button>
          </div>
        </div>

        {selectedModule && hyperparams[selectedModule] && (
          <div className="mt-6">
            <h3 className="text-lg font-medium text-ink-900 mb-4 capitalize">
              {selectedModule.replace('_', ' ')} Parameters
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(hyperparams[selectedModule]).map(([key, value]) => (
                <div key={key}>
                  <label className="block text-sm font-medium text-ink-700 mb-1 capitalize">
                    {key.replace('_', ' ')}
                  </label>
                  <input
                    type={typeof value === 'number' ? 'number' : 'text'}
                    value={value as string | number}
                    onChange={(e) => updateHyperparam(selectedModule, key,
                      typeof value === 'number' ? parseFloat(e.target.value) : e.target.value
                    )}
                    className="w-full px-3 py-2 border border-ink-300 rounded-md focus:outline-none focus:ring-2 focus:ring-ink-500"
                  />
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ModelTraining;
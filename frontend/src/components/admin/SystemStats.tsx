import { useState, useEffect } from 'react';
import api from '../../lib/api';
import { Activity, BookOpen, Layers } from 'lucide-react';

interface SystemStatsState {
  total_chunks: number;
  total_books: number;
  chunks_per_philosopher: Record<string, number>;
}

function SystemStats() {
  const [stats, setStats] = useState<SystemStatsState>({
    total_chunks: 0,
    total_books: 0,
    chunks_per_philosopher: {}
  });

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    try {
      const response = await api.get('/admin/stats');
      setStats(response.data);
    } catch (error) {
      console.error('Failed to load stats:', error);
    }
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center mb-4">
          <BookOpen className="h-6 w-6 text-ink-600 mr-3" />
          <h3 className="text-lg font-cinzel font-bold text-ink-900">Total Books</h3>
        </div>
        <p className="text-3xl font-bold text-ink-900">{stats.total_books}</p>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center mb-4">
          <Layers className="h-6 w-6 text-ink-600 mr-3" />
          <h3 className="text-lg font-cinzel font-bold text-ink-900">Total Chunks</h3>
        </div>
        <p className="text-3xl font-bold text-ink-900">{stats.total_chunks}</p>
      </div>

      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center mb-4">
          <Activity className="h-6 w-6 text-ink-600 mr-3" />
          <h3 className="text-lg font-cinzel font-bold text-ink-900">Philosopher Breakdown</h3>
        </div>
        <div className="space-y-2 text-sm text-ink-700">
          {Object.entries(stats.chunks_per_philosopher).length === 0 ? (
            <p>No data available yet.</p>
          ) : (
            Object.entries(stats.chunks_per_philosopher).map(([philosopher, count]) => (
              <div key={philosopher} className="flex justify-between">
                <span>{philosopher}</span>
                <span>{count}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

export default SystemStats;
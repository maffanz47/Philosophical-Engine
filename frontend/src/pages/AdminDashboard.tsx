import { useState } from 'react';
import BookUploader from '../components/admin/BookUploader';
import BookLibrary from '../components/admin/BookLibrary';
import ModelTraining from '../components/admin/ModelTraining';
import ExperimentHistory from '../components/admin/ExperimentHistory';
import UserManagement from '../components/admin/UserManagement';
import SystemStats from '../components/admin/SystemStats';

function AdminDashboard() {
  const [activeTab, setActiveTab] = useState('books');

  const tabs = [
    { id: 'books', label: 'Book Management', component: BookLibrary },
    { id: 'upload', label: 'Upload Book', component: BookUploader },
    { id: 'training', label: 'Model Training', component: ModelTraining },
    { id: 'experiments', label: 'Experiments', component: ExperimentHistory },
    { id: 'users', label: 'User Management', component: UserManagement },
    { id: 'stats', label: 'System Stats', component: SystemStats },
  ];

  const ActiveComponent = tabs.find(tab => tab.id === activeTab)?.component || BookLibrary;

  return (
    <div className="min-h-screen bg-parchment-50">
      <header className="bg-ink-900 text-parchment-50 p-6 shadow-lg">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-cinzel font-bold">Philosophical Engine - Admin Dashboard</h1>
          <p className="text-parchment-200 mt-2">Manage books, train models, and oversee the system</p>
        </div>
      </header>

      <nav className="bg-ink-800 text-parchment-100">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex space-x-8 py-4">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  activeTab === tab.id
                    ? 'bg-parchment-200 text-ink-900'
                    : 'hover:bg-ink-700'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto p-6">
        <ActiveComponent />
      </main>
    </div>
  );
}

export default AdminDashboard;
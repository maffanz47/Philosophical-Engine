import { useState, useEffect } from 'react';
import api from '../../lib/api';

interface User {
  id: string;
  email: string;
  role: string;
  created_at: string;
}

function UserManagement() {
  const [users, setUsers] = useState<User[]>([]);

  useEffect(() => {
    loadUsers();
  }, []);

  const loadUsers = async () => {
    try {
      const response = await api.get('/admin/users');
      setUsers(response.data);
    } catch (error) {
      console.error('Failed to load users:', error);
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-cinzel font-bold text-ink-900 mb-4">User Management</h2>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-ink-50">
            <tr>
              <th className="px-4 py-3 text-left text-xs font-medium text-ink-700 uppercase tracking-wider">Email</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-ink-700 uppercase tracking-wider">Role</th>
              <th className="px-4 py-3 text-left text-xs font-medium text-ink-700 uppercase tracking-wider">Created</th>
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-ink-200">
            {users.map((user) => (
              <tr key={user.id} className="hover:bg-ink-50">
                <td className="px-4 py-3 text-sm text-ink-700">{user.email}</td>
                <td className="px-4 py-3 text-sm text-ink-700 capitalize">{user.role}</td>
                <td className="px-4 py-3 text-sm text-ink-700">{new Date(user.created_at).toLocaleDateString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default UserManagement;
import { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import api from '../lib/api';
import { RegisterRequest } from '../types';

const RegisterPage = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await api.post('/auth/register', { email, password } as RegisterRequest);
      navigate('/login');
    } catch (error) {
      alert('Registration failed');
    }
  };

  return (
    <div className="min-h-screen bg-ink flex items-center justify-center">
      <div className="bg-deep-brown p-8 rounded-lg shadow-lg w-full max-w-md">
        <h1 className="text-2xl font-cinzel text-aged-gold text-center mb-6">Register</h1>
        <form onSubmit={handleSubmit}>
          <div className="mb-4">
            <label className="block text-parchment font-eb-garamond mb-2">Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full p-2 bg-ink text-parchment border border-faded-gold rounded"
              required
            />
          </div>
          <div className="mb-6">
            <label className="block text-parchment font-eb-garamond mb-2">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full p-2 bg-ink text-parchment border border-faded-gold rounded"
              required
            />
          </div>
          <button type="submit" className="w-full bg-aged-gold text-ink py-2 rounded font-cinzel hover:bg-candlelight">
            Register
          </button>
        </form>
        <p className="text-parchment text-center mt-4">
          Already have an account? <Link to="/login" className="text-aged-gold hover:text-candlelight">Login</Link>
        </p>
      </div>
    </div>
  );
};

export default RegisterPage;
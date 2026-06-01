import React, { createContext, useContext, useState, useEffect } from 'react';

// ─── Auth Context ──────────────────────────────────────────────────────────────
export const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser]   = useState(null);
  const [token, setToken] = useState(localStorage.getItem('fs_token'));

  useEffect(() => {
    if (token) {
      const stored = localStorage.getItem('fs_user');
      if (stored) setUser(JSON.parse(stored));
    }
  }, []);

  const login = async (email, password) => {
    const res = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });
    if (!res.ok) throw new Error('Invalid credentials');
    const data = await res.json();
    setToken(data.token);
    setUser(data.user);
    localStorage.setItem('fs_token', data.token);
    localStorage.setItem('fs_user', JSON.stringify(data.user));
    return data;
  };

  const logout = () => {
    setToken(null); setUser(null);
    localStorage.removeItem('fs_token');
    localStorage.removeItem('fs_user');
  };

  return (
    <AuthContext.Provider value={{ user, token, login, logout, isAuthenticated: !!token }}>
      {children}
    </AuthContext.Provider>
  );
}

export const useAuth = () => useContext(AuthContext);

// ─── API Helper ────────────────────────────────────────────────────────────────
export function useApi() {
  const { token } = useAuth();
  const call = async (url, options = {}) => {
    const res = await fetch(url, {
      ...options,
      headers: {
        'Authorization': `Bearer ${token}`,
        ...(options.headers || {})
      }
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(err.detail || 'Request failed');
    }
    return res.json();
  };
  return call;
}
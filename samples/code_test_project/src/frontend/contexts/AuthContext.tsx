import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';

interface User {
  id: number;
  name: string;
  email: string;
  permissions: string[];
}

interface AuthContextType {
  user: User | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<boolean>;
  logout: () => void;
  loading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check for existing session
    const checkSession = async () => {
      try {
        const token = localStorage.getItem('auth_token');
        if (token) {
          const userData = await validateToken(token);
          if (userData) {
            setUser(userData);
          }
        }
      } catch (error) {
        console.error('Session validation failed:', error);
        localStorage.removeItem('auth_token');
      } finally {
        setLoading(false);
      }
    };

    checkSession();
  }, []);

  const login = async (email: string, password: string): Promise<boolean> => {
    try {
      setLoading(true);
      
      // Mock authentication - replace with real API call
      const response = await mockLogin(email, password);
      
      if (response.success) {
        setUser(response.user);
        localStorage.setItem('auth_token', response.token);
        return true;
      }
      
      return false;
    } catch (error) {
      console.error('Login failed:', error);
      return false;
    } finally {
      setLoading(false);
    }
  };

  const logout = () => {
    setUser(null);
    localStorage.removeItem('auth_token');
  };

  const value: AuthContextType = {
    user,
    isAuthenticated: !!user,
    login,
    logout,
    loading
  };

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

// Mock authentication functions
const mockLogin = async (email: string, password: string) => {
  // Simulate API delay
  await new Promise(resolve => setTimeout(resolve, 1000));
  
  // Mock users
  const users = [
    {
      id: 1,
      name: 'Admin User',
      email: 'admin@example.com',
      password: 'admin123',
      permissions: ['read', 'write', 'admin']
    },
    {
      id: 2,
      name: 'Regular User',
      email: 'user@example.com', 
      password: 'user123',
      permissions: ['read']
    }
  ];
  
  const user = users.find(u => u.email === email && u.password === password);
  
  if (user) {
    const { password: _, ...userWithoutPassword } = user;
    return {
      success: true,
      user: userWithoutPassword,
      token: 'mock_token_' + Date.now()
    };
  }
  
  return { success: false };
};

const validateToken = async (token: string): Promise<User | null> => {
  // Mock token validation
  if (token.startsWith('mock_token_')) {
    return {
      id: 1,
      name: 'Mock User',
      email: 'user@example.com',
      permissions: ['read']
    };
  }
  return null;
};

import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';
import Header from './components/Header';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import FileExplorer from './pages/FileExplorer';
import SearchResults from './pages/SearchResults';
import Settings from './pages/Settings';
import Login from './pages/Login';
import './App.css';

interface AppProps {
  title?: string;
}

const App: React.FC<AppProps> = ({ title = "CodeRaptor" }) => {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Simulate app initialization
    const timer = setTimeout(() => {
      setLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="loading-spinner"></div>
        <p>Loading {title}...</p>
      </div>
    );
  }

  return (
    <ThemeProvider>
      <AuthProvider>
        <Router>
          <div className="app">
            <AppContent 
              title={title}
              sidebarOpen={sidebarOpen}
              setSidebarOpen={setSidebarOpen}
            />
          </div>
        </Router>
      </AuthProvider>
    </ThemeProvider>
  );
};

interface AppContentProps {
  title: string;
  sidebarOpen: boolean;
  setSidebarOpen: (open: boolean) => void;
}

const AppContent: React.FC<AppContentProps> = ({ title, sidebarOpen, setSidebarOpen }) => {
  const { user, isAuthenticated } = useAuth();

  if (!isAuthenticated) {
    return <Login />;
  }

  return (
    <>
      <Header 
        title={title}
        user={user}
        onToggleSidebar={() => setSidebarOpen(!sidebarOpen)}
      />
      
      <div className="app-body">
        <Sidebar 
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
        />
        
        <main className={`main-content ${sidebarOpen ? 'with-sidebar' : 'full-width'}`}>
          <Routes>
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/files" element={<FileExplorer />} />
            <Route path="/search" element={<SearchResults />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="*" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </main>
      </div>
    </>
  );
};

export default App;

// examples/samples/sample.js
/**
 * Sample JavaScript File
 * ======================
 * 
 * Demonstrates modern JavaScript features for semantic chunking:
 * - ES6+ syntax (classes, arrow functions, destructuring)
 * - Async/await patterns
 * - React JSX components
 * - Module imports/exports
 * - Higher-order functions
 */

import React, { useState, useEffect, useCallback } from 'react';
import { EventEmitter } from 'events';
import axios from 'axios';
import _ from 'lodash';

// Constants and configuration
const DEFAULT_CONFIG = {
  apiUrl: 'https://api.example.com',
  timeout: 5000,
  retries: 3,
  batchSize: 100
};

const API_ENDPOINTS = {
  users: '/users',
  posts: '/posts',
  comments: '/comments'
};

/**
 * Utility class for API operations
 */
class ApiClient extends EventEmitter {
  constructor(config = {}) {
    super();
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.requestCache = new Map();
    this.retryQueue = [];
    
    // Bind methods to preserve context
    this.request = this.request.bind(this);
    this.handleError = this.handleError.bind(this);
  }

  /**
   * Make an authenticated API request
   */
  async request(endpoint, options = {}) {
    const url = `${this.config.apiUrl}${endpoint}`;
    const requestKey = `${options.method || 'GET'}:${url}`;
    
    // Check cache first
    if (this.requestCache.has(requestKey)) {
      this.emit('cache-hit', { endpoint, cached: true });
      return this.requestCache.get(requestKey);
    }

    try {
      const response = await this.makeRequestWithRetry(url, options);
      
      // Cache successful responses
      if (response.status === 200) {
        this.requestCache.set(requestKey, response.data);
      }
      
      this.emit('request-success', { endpoint, status: response.status });
      return response.data;
      
    } catch (error) {
      this.handleError(error, endpoint);
      throw error;
    }
  }

  /**
   * Make request with automatic retry logic
   */
  async makeRequestWithRetry(url, options, attempt = 1) {
    try {
      return await axios({
        url,
        timeout: this.config.timeout,
        ...options
      });
      
    } catch (error) {
      if (attempt < this.config.retries && this.shouldRetry(error)) {
        const delay = Math.pow(2, attempt) * 1000; // Exponential backoff
        await this.delay(delay);
        return this.makeRequestWithRetry(url, options, attempt + 1);
      }
      throw error;
    }
  }

  shouldRetry(error) {
    return error.code === 'ECONNRESET' || 
           error.code === 'ETIMEDOUT' ||
           (error.response && error.response.status >= 500);
  }

  handleError(error, endpoint) {
    this.emit('request-error', { 
      endpoint, 
      error: error.message,
      status: error.response?.status 
    });
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  clearCache() {
    this.requestCache.clear();
    this.emit('cache-cleared');
  }
}

/**
 * Data processing utilities
 */
const DataProcessor = {
  /**
   * Process data in batches with async operations
   */
  async processBatch(data, processingFn, batchSize = DEFAULT_CONFIG.batchSize) {
    const results = [];
    
    for (let i = 0; i < data.length; i += batchSize) {
      const batch = data.slice(i, i + batchSize);
      
      // Process batch items concurrently
      const batchPromises = batch.map(async (item, index) => {
        try {
          return await processingFn(item, i + index);
        } catch (error) {
          return { error: error.message, item };
        }
      });
      
      const batchResults = await Promise.allSettled(batchPromises);
      results.push(...batchResults.map(result => 
        result.status === 'fulfilled' ? result.value : result.reason
      ));
    }
    
    return results;
  },

  /**
   * Transform and filter data using functional programming
   */
  transformData: (data) => {
    return data
      .filter(item => item && typeof item === 'object')
      .map(({ id, name, email, ...rest }) => ({
        id: parseInt(id) || 0,
        name: name?.trim() || 'Unknown',
        email: email?.toLowerCase() || '',
        metadata: rest
      }))
      .filter(item => item.id > 0 && item.email.includes('@'));
  },

  /**
   * Group data by specified key with statistics
   */
  groupWithStats: (data, groupKey) => {
    const grouped = _.groupBy(data, groupKey);
    
    return Object.entries(grouped).reduce((acc, [key, items]) => {
      acc[key] = {
        items,
        count: items.length,
        stats: {
          avgId: _.meanBy(items, 'id'),
          minId: _.minBy(items, 'id')?.id,
          maxId: _.maxBy(items, 'id')?.id
        }
      };
      return acc;
    }, {});
  }
};

/**
 * React Hook for data fetching
 */
function useApiData(endpoint, dependencies = []) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const apiClient = useMemo(() => new ApiClient(), []);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const result = await apiClient.request(endpoint);
      setData(result);
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [apiClient, endpoint]);

  useEffect(() => {
    fetchData();
  }, [fetchData, ...dependencies]);

  const refetch = useCallback(() => {
    apiClient.clearCache();
    fetchData();
  }, [apiClient, fetchData]);

  return { data, loading, error, refetch };
}

/**
 * React component for displaying user data
 */
const UserList = ({ filter = '', sortBy = 'name' }) => {
  const { data: users, loading, error, refetch } = useApiData(API_ENDPOINTS.users);
  const [selectedUsers, setSelectedUsers] = useState(new Set());

  // Memoized filtered and sorted users
  const processedUsers = useMemo(() => {
    if (!users) return [];
    
    return users
      .filter(user => 
        user.name.toLowerCase().includes(filter.toLowerCase()) ||
        user.email.toLowerCase().includes(filter.toLowerCase())
      )
      .sort((a, b) => {
        if (sortBy === 'name') return a.name.localeCompare(b.name);
        if (sortBy === 'email') return a.email.localeCompare(b.email);
        return a.id - b.id;
      });
  }, [users, filter, sortBy]);

  const handleUserSelect = useCallback((userId) => {
    setSelectedUsers(prev => {
      const newSelected = new Set(prev);
      if (newSelected.has(userId)) {
        newSelected.delete(userId);
      } else {
        newSelected.add(userId);
      }
      return newSelected;
    });
  }, []);

  const handleBulkAction = async (action) => {
    const selectedUserList = processedUsers.filter(user => 
      selectedUsers.has(user.id)
    );

    try {
      await DataProcessor.processBatch(
        selectedUserList,
        async (user) => {
          console.log(`Performing ${action} on user ${user.id}`);
          // Simulate API call
          await new Promise(resolve => setTimeout(resolve, 100));
          return { success: true, userId: user.id };
        }
      );
      
      setSelectedUsers(new Set());
      refetch();
      
    } catch (error) {
      console.error(`Bulk ${action} failed:`, error);
    }
  };

  if (loading) {
    return <LoadingSpinner message="Loading users..." />;
  }

  if (error) {
    return (
      <ErrorDisplay 
        message={error} 
        onRetry={refetch}
      />
    );
  }

  return (
    <div className="user-list">
      <div className="user-list__header">
        <h2>Users ({processedUsers.length})</h2>
        {selectedUsers.size > 0 && (
          <div className="bulk-actions">
            <button onClick={() => handleBulkAction('delete')}>
              Delete Selected ({selectedUsers.size})
            </button>
            <button onClick={() => handleBulkAction('export')}>
              Export Selected
            </button>
          </div>
        )}
      </div>

      <div className="user-list__content">
        {processedUsers.map(user => (
          <UserCard
            key={user.id}
            user={user}
            selected={selectedUsers.has(user.id)}
            onSelect={() => handleUserSelect(user.id)}
          />
        ))}
      </div>
    </div>
  );
};

/**
 * Individual user card component
 */
const UserCard = React.memo(({ user, selected, onSelect }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div 
      className={`user-card ${selected ? 'user-card--selected' : ''}`}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="user-card__header">
        <input
          type="checkbox"
          checked={selected}
          onChange={onSelect}
          onClick={(e) => e.stopPropagation()}
        />
        <h3>{user.name}</h3>
        <span className="user-card__email">{user.email}</span>
      </div>
      
      {expanded && (
        <div className="user-card__details">
          <p>ID: {user.id}</p>
          {user.metadata && (
            <pre>{JSON.stringify(user.metadata, null, 2)}</pre>
          )}
        </div>
      )}
    </div>
  );
});

/**
 * Reusable loading component
 */
const LoadingSpinner = ({ message = 'Loading...' }) => (
  <div className="loading-spinner">
    <div className="spinner" />
    <p>{message}</p>
  </div>
);

/**
 * Error display component
 */
const ErrorDisplay = ({ message, onRetry }) => (
  <div className="error-display">
    <h3>⚠️ Error</h3>
    <p>{message}</p>
    {onRetry && (
      <button onClick={onRetry}>Try Again</button>
    )}
  </div>
);

// Utility functions
export const utils = {
  /**
   * Debounced search function
   */
  createDebouncedSearch: (searchFn, delay = 300) => {
    return _.debounce(searchFn, delay);
  },

  /**
   * Format date for display
   */
  formatDate: (date, options = {}) => {
    return new Intl.DateTimeFormat('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      ...options
    }).format(new Date(date));
  },

  /**
   * Validate email format
   */
  isValidEmail: (email) => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  }
};

// Default exports
export default UserList;
export { ApiClient, DataProcessor, useApiData };

// Main execution for standalone running
if (typeof window === 'undefined' && typeof module !== 'undefined') {
  // Node.js environment - run demo
  (async () => {
    console.log('🚀 Running JavaScript demo...');
    
    const client = new ApiClient();
    const testData = [
      { id: 1, name: 'John Doe', email: 'john@example.com' },
      { id: 2, name: 'Jane Smith', email: 'jane@example.com' },
      { id: 3, name: 'Bob Johnson', email: 'bob@example.com' }
    ];
    
    console.log('📊 Processing test data...');
    const processed = DataProcessor.transformData(testData);
    const grouped = DataProcessor.groupWithStats(processed, 'name');
    
    console.log('✅ Processed data:', processed);
    console.log('📈 Grouped stats:', grouped);
    console.log('🎉 Demo completed!');
  })();
}
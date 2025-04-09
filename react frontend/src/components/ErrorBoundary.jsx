import React from 'react';
import '../home.css';

export default class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { 
      hasError: false, 
      error: null,
      isNetworkError: false
    };
  }

  static getDerivedStateFromError(error) {
    return { 
      hasError: true, 
      error,
      isNetworkError: error.message.includes('Network Error') 
    };
  }

  componentDidCatch(error, errorInfo) {
    console.error("Error caught by boundary:", error, errorInfo);
    if (error.message.includes('Network Error')) {
      this.setState({ isNetworkError: true });
    }
  }

  handleReset = () => {
    if (this.state.isNetworkError) {
      // Clear cache and hard reload for network errors
      if (caches) {
        caches.keys().then(names => {
          names.forEach(name => caches.delete(name));
        });
      }
      window.location.reload(true);
    } else {
      this.setState({ hasError: false, error: null, isNetworkError: false });
    }
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="error-boundary">
          <h2>Something went wrong</h2>
          <p>{this.state.error.message}</p>
          <div className="error-actions">
            <button onClick={this.handleReset}>
              {this.state.isNetworkError ? 'Clear Cache & Reload' : 'Retry'}
            </button>
            {!this.state.isNetworkError && (
              <button onClick={() => window.location.reload()}>Reload Page</button>
            )}
          </div>
          {this.state.isNetworkError && (
            <div className="network-tips">
              <p>Network connection tips:</p>
              <ul>
                <li>Check your internet connection</li>
                <li>Verify the API server is running</li>
                <li>Try again in a few moments</li>
              </ul>
            </div>
          )}
        </div>
      );
    }
    return this.props.children;
  }
}
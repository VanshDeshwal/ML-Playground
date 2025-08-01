/* Configuration for different environments */
const CONFIG = {
    // For development - local backend
    development: {
        API_BASE_URL: 'http://localhost:8000',
        ENVIRONMENT: 'development'
    },
    
    // For production - deployed backend
    production: {
        API_BASE_URL: 'https://api.playground.vanshdeshwal.dev',
        ENVIRONMENT: 'production'
    }
};

// Automatically detect environment
// Custom domain and GitHub Pages will be production, localhost will be development
const isLocalhost = window.location.hostname === 'localhost' || 
                   window.location.hostname === '127.0.0.1' || 
                   window.location.hostname === '';

// Use appropriate configuration based on environment
const CURRENT_CONFIG = isLocalhost ? CONFIG.development : CONFIG.production;

// Debug logging
console.log('Environment Detection:', {
    hostname: window.location.hostname,
    isLocalhost: isLocalhost,
    selectedConfig: CURRENT_CONFIG.ENVIRONMENT,
    apiBaseUrl: CURRENT_CONFIG.API_BASE_URL
});

// Export for use in other files
window.APP_CONFIG = CURRENT_CONFIG;

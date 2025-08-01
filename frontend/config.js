/* Configuration for different environments */
const CONFIG = {
    // For development - local backend
    development: {
        API_BASE_URL: 'http://localhost:8000',
        ENVIRONMENT: 'development'
    },
    
    // For production - Azure backend
    production: {
        API_BASE_URL: 'https://api.playground.vanshdeshwal.dev', // Replace with your Azure URL
        ENVIRONMENT: 'production'
    }
};

// Automatically detect environment
// Custom domain and GitHub Pages will be production, localhost will be development
const isLocalhost = window.location.hostname === 'localhost' || 
                   window.location.hostname === '127.0.0.1' || 
                   window.location.hostname === '';

// Temporarily force development mode for testing
const CURRENT_CONFIG = CONFIG.development;  // Always use localhost backend

// Export for use in other files
window.APP_CONFIG = CURRENT_CONFIG;

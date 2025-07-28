/* Configuration for different environments */
const CONFIG = {
    // For development - local backend
    development: {
        API_BASE_URL: 'http://localhost:8000',
        ENVIRONMENT: 'development'
    },
    
    // For production - Azure backend
    production: {
        API_BASE_URL: 'https://your-azure-app.azurewebsites.net', // Replace with your Azure URL
        ENVIRONMENT: 'production'
    }
};

// Automatically detect environment
// GitHub Pages will be production, localhost will be development
const isLocalhost = window.location.hostname === 'localhost' || 
                   window.location.hostname === '127.0.0.1' || 
                   window.location.hostname === '';

const CURRENT_CONFIG = isLocalhost ? CONFIG.development : CONFIG.production;

// Export for use in other files
window.APP_CONFIG = CURRENT_CONFIG;

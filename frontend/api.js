// API service for backend communication
class APIService {
    constructor() {
        this.baseURL = window.APP_CONFIG.API_BASE_URL;
        this.cache = new Map();
    }

    async checkBackendStatus() {
        try {
            const response = await fetch(`${this.baseURL}/`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
            });
            return response.ok;
        } catch (error) {
            console.error('Backend status check failed:', error);
            return false;
        }
    }

    async getAlgorithms() {
        const cacheKey = 'algorithms';
        
        // Only return cached data if we previously had a successful response
        if (this.cache.has(cacheKey)) {
            const cachedData = this.cache.get(cacheKey);
            // Always return an array - handle both old and new cache formats
            if (Array.isArray(cachedData)) {
                return cachedData;  // Already an array
            } else if (cachedData && Array.isArray(cachedData.algorithms)) {
                return cachedData.algorithms;  // Extract algorithms array
            } else {
                console.warn('Invalid cached data, clearing cache');
                this.cache.delete(cacheKey);
            }
        }

        try {
            const response = await fetch(`${this.baseURL}/algorithms/`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
                // Add timeout to prevent hanging
                signal: AbortSignal.timeout(10000) // 10 second timeout
            });

            // Check if response is ok (status 200-299)
            if (!response.ok) {
                const errorText = await response.text();
                console.error(`Backend error ${response.status}:`, errorText);
                throw new Error(`Backend returned ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            
            // Validate that we got the expected data structure
            if (!data || !data.algorithms || !Array.isArray(data.algorithms)) {
                console.error('Invalid data structure received:', data);
                throw new Error('Invalid response format from backend');
            }

            // Only cache successful responses with valid data
            this.cache.set(cacheKey, data.algorithms);  // Cache just the algorithms array
            return data.algorithms;  // Return just the algorithms array
            
        } catch (error) {
            console.error('Failed to fetch algorithms:', error);
            
            // Clear any stale cache on error
            if (this.cache.has(cacheKey)) {
                this.cache.delete(cacheKey);
            }
            
            // Re-throw with more specific error message
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                throw new Error('Cannot connect to backend server. Please check if the backend is running.');
            } else if (error.name === 'AbortError') {
                throw new Error('Request timed out. Backend server may be slow or unavailable.');
            } else {
                throw error;
            }
        }
    }

    async trainAlgorithm(algorithmName, hyperparameters = {}, datasetConfig = {}) {
        try {
            const requestBody = {
                algorithm_id: algorithmName,
                hyperparameters: hyperparameters,
                dataset_config: datasetConfig,
                dataset_source: "generated",
                compare_sklearn: true
            };

            const response = await fetch(`${this.baseURL}/training/train`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody),
                // Add timeout for training requests
                signal: AbortSignal.timeout(30000) // 30 second timeout for training
            });

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Training API error response:', errorText);
                
                // Provide more specific error messages
                if (response.status === 404) {
                    throw new Error(`Algorithm "${algorithmName}" not found on backend`);
                } else if (response.status === 500) {
                    throw new Error(`Backend training error: ${errorText}`);
                } else {
                    throw new Error(`Training failed (${response.status}): ${errorText}`);
                }
            }

            const result = await response.json();
            
            // Validate training result structure
            if (!result || typeof result.success !== 'boolean') {
                console.error('Invalid training result:', result);
                throw new Error('Invalid training response format');
            }

            return result;
            
        } catch (error) {
            console.error(`Failed to train ${algorithmName}:`, error);
            
            // Provide user-friendly error messages
            if (error.name === 'TypeError' && error.message.includes('fetch')) {
                throw new Error('Cannot connect to backend server for training. Please check if the backend is running.');
            } else if (error.name === 'AbortError') {
                throw new Error('Training request timed out. The algorithm may be taking too long to train.');
            } else {
                throw error;
            }
        }
    }

    // Clear cache when needed
    clearCache() {
        this.cache.clear();
    }

    // Force refresh - clears cache and fetches fresh data
    async forceRefreshAlgorithms() {
        console.log('DEBUG: Force refreshing algorithms - clearing cache first');
        this.clearCache();
        return await this.getAlgorithms();
    }

    // Check if backend is reachable
    async checkBackendHealth() {
        try {
            console.log('DEBUG: Checking backend health');
            const response = await fetch(`${this.baseURL}/health`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
                signal: AbortSignal.timeout(5000) // 5 second timeout for health check
            });

            if (!response.ok) {
                return false;
            }

            const health = await response.json();
            console.log('DEBUG: Backend health check result:', health);
            return health.status === 'healthy';
            
        } catch (error) {
            console.warn('Backend health check failed:', error);
            return false;
        }
    }

    // Clear cache if backend is not healthy
    async ensureBackendConnectivity() {
        const isHealthy = await this.checkBackendHealth();
        if (!isHealthy) {
            console.log('DEBUG: Backend not healthy, clearing cache');
            this.clearCache();
            throw new Error('Backend server is not available. Please check if the backend is running and try again.');
        }
        return true;
    }
}

// Create global API service instance
window.apiService = new APIService();

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
        
        if (this.cache.has(cacheKey)) {
            return this.cache.get(cacheKey);
        }

        try {
            const response = await fetch(`${this.baseURL}/algorithms`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.cache.set(cacheKey, data);
            return data;
        } catch (error) {
            console.error('Failed to fetch algorithms:', error);
            throw error;
        }
    }

    async trainAlgorithm(algorithmName) {
        try {
            const response = await fetch(`${this.baseURL}/train/${algorithmName}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            return await response.json();
        } catch (error) {
            console.error(`Failed to train ${algorithmName}:`, error);
            throw error;
        }
    }

    // Clear cache when needed
    clearCache() {
        this.cache.clear();
    }
}

// Create global API service instance
window.apiService = new APIService();

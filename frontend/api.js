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

    async trainAlgorithm(algorithmName, hyperparameters = {}, datasetConfig = {}) {
        try {
            const requestBody = {
                algorithm_id: algorithmName,
                hyperparameters: hyperparameters,
                dataset_config: datasetConfig,
                dataset_source: "generated",
                compare_sklearn: true
            };

            console.log('Sending training request:', requestBody);

            const response = await fetch(`${this.baseURL}/train`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody)
            });

            console.log('Response status:', response.status);

            if (!response.ok) {
                const errorText = await response.text();
                console.error('Training API error response:', errorText);
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }

            const result = await response.json();
            console.log('Training API success response:', result);
            return result;
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

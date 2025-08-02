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

    async trainAlgorithm(algorithmName, hyperparameters = {}, dataset = null) {
        try {
            const requestBody = hyperparameters;
            
            // Build URL with dataset parameter only if specified
            let url = `${this.baseURL}/training/${algorithmName}`;
            if (dataset) {
                url += `?dataset=${dataset}`;
            }

            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestBody),
                // Add timeout for training requests
                signal: AbortSignal.timeout(60000) // 60 second timeout for training
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
            
            // Initialize data contract validator if not exists
            if (!this.dataValidator) {
                this.dataValidator = new DataContractValidator();
            }

            // Validate and normalize the result using data contract
            let processedResult;
            try {
                processedResult = this.dataValidator.processTrainingResult(result);
                console.log('✅ API data contract validation passed');
            } catch (contractError) {
                console.warn('⚠️ Data contract validation failed, using raw data:', contractError);
                processedResult = result; // Use raw data as fallback
            }
            
            // Validate response structure
            if (!processedResult.success) {
                throw new Error(processedResult.error || 'Training failed');
            }

            console.log('Training completed successfully:', processedResult);
            return processedResult;

        } catch (error) {
            console.error('Training failed:', error);
            
            // Return a structured error response
            return {
                success: false,
                error: error.message,
                algorithm_id: algorithmName,
                timestamp: new Date().toISOString()
            };
        }
    }

    async getAvailableDatasets() {
        try {
            const response = await fetch(`${this.baseURL}/training/datasets`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (!response.ok) {
                throw new Error(`Failed to fetch datasets: ${response.status}`);
            }

            const result = await response.json();
            return result.datasets || [];

        } catch (error) {
            console.error('Failed to fetch datasets:', error);
            // Return default datasets if API fails
            return [
                {
                    id: "diabetes",
                    name: "Diabetes Dataset",
                    type: "regression",
                    description: "Diabetes progression prediction",
                    samples: 442,
                    features: 10
                }
            ];
        }
    }

    async getMetricsInfo(algorithmType) {
        try {
            const response = await fetch(`${this.baseURL}/training/metrics/${algorithmType}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
            });

            if (!response.ok) {
                throw new Error(`Failed to fetch metrics info: ${response.status}`);
            }

            return await response.json();

        } catch (error) {
            console.error('Failed to fetch metrics info:', error);
            return null;
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

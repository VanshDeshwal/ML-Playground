// Main application controller for ML Playground homepage
class MLPlaygroundApp {
    constructor() {
        this.algorithms = [];
        this.init();
    }

    async init() {
        await this.checkBackendConnection();
        await this.loadAlgorithms();
    }

    async init() {
        await this.checkBackendConnection();
        await this.loadAlgorithms();
    }

    async checkBackendConnection() {
        const statusIndicator = document.getElementById('status-indicator');
        const statusDot = statusIndicator.querySelector('.status-dot');
        const statusText = statusIndicator.querySelector('.status-text');
        
        try {
            // Use the new health check method
            const isHealthy = await window.apiService.checkBackendHealth();
            
            if (isHealthy) {
                statusDot.className = 'status-dot connected';
                statusText.textContent = 'Connected';
            } else {
                statusDot.className = 'status-dot disconnected';
                statusText.textContent = 'Backend Unhealthy';
                // Clear any stale cache when backend is unhealthy
                window.apiService.clearCache();
            }
        } catch (error) {
            statusDot.className = 'status-dot status-error';
            statusText.textContent = 'Connection Failed';
            // Clear cache on connection error
            window.apiService.clearCache();
        }
    }

    async loadAlgorithms() {
        const gridElement = document.getElementById('algorithms-grid');
        
        try {
            console.log('DEBUG: Loading algorithms...');
            const algorithms = await window.apiService.getAlgorithms();
            console.log('DEBUG: Received algorithms:', algorithms);
            this.algorithms = algorithms;
            this.renderAlgorithms(algorithms);
        } catch (error) {
            console.error('DEBUG: Failed to load algorithms:', error);
            gridElement.innerHTML = `
                <div class="loading">
                    <i class="fas fa-exclamation-triangle" style="color: var(--error-color);"></i>
                    <div>
                        <p>Failed to load algorithms</p>
                        <button class="btn btn-primary" onclick="app.loadAlgorithms()" style="margin-top: 1rem;">
                            <i class="fas fa-refresh"></i> Retry
                        </button>
                    </div>
                </div>
            `;
        }
    }

    renderAlgorithms(algorithms) {
        const gridElement = document.getElementById('algorithms-grid');
        
        if (!algorithms || algorithms.length === 0) {
            gridElement.innerHTML = `
                <div class="loading">
                    <i class="fas fa-info-circle"></i>
                    <p>No algorithms available</p>
                </div>
            `;
            return;
        }

        const algorithmsHTML = algorithms.map(algorithm => {
            const icon = this.getAlgorithmIcon(algorithm.type);
            return `
                <div class="algorithm-card" onclick="window.location.href='algorithm.html?id=${algorithm.id}'">
                    <h3>
                        <i class="${icon}"></i>
                        ${algorithm.name}
                    </h3>
                    <p>${algorithm.description}</p>
                    <span class="algorithm-type">${algorithm.type}</span>
                    <div class="card-action">
                        <i class="fas fa-arrow-right"></i>
                        <span>Train Algorithm</span>
                    </div>
                </div>
            `;
        }).join('');

        gridElement.innerHTML = algorithmsHTML;
    }

    getAlgorithmIcon(type) {
        const iconMap = {
            'regression': 'fas fa-chart-line',
            'classification': 'fas fa-layer-group',
            'clustering': 'fas fa-project-diagram',
            'neural network': 'fas fa-brain',
            'ensemble': 'fas fa-sitemap',
            'default': 'fas fa-cog'
        };
        return iconMap[type] || iconMap.default;
    }

    showError(message) {
        // Simple error display - could be enhanced with a toast system
        alert(message);
    }

    // Utility method to refresh the app
    async refresh() {
        window.apiService.clearCache();
        await this.checkBackendConnection();
        await this.loadAlgorithms();
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new MLPlaygroundApp();

    // Fetch and display version
    fetch('version.txt')
        .then(response => response.text())
        .then(version => {
            document.getElementById('version-info').textContent = `Version: ${version.trim()}`;
        })
        .catch(() => {
            document.getElementById('version-info').textContent = 'Version: unknown';
        });
});

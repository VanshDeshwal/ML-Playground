// Main application logic
class MLPlaygroundApp {
    constructor() {
        this.currentAlgorithm = null;
        this.isTraining = false;
        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.checkBackendConnection();
        await this.loadAlgorithms();
    }

    setupEventListeners() {
        // Modal controls
        const modal = document.getElementById('algorithm-modal');
        const closeBtn = document.getElementById('close-modal');
        
        closeBtn.addEventListener('click', () => this.closeModal());
        modal.addEventListener('click', (e) => {
            if (e.target === modal) this.closeModal();
        });

        // Train button
        const trainBtn = document.getElementById('train-btn');
        trainBtn.addEventListener('click', () => this.trainCurrentAlgorithm());

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') this.closeModal();
        });

        // Refresh backend status every 30 seconds
        setInterval(() => this.checkBackendConnection(), 30000);
    }

    async checkBackendConnection() {
        const statusElement = document.getElementById('backend-status');
        const statusIcon = statusElement.querySelector('i');
        
        // Show connecting state
        statusElement.className = 'status connecting';
        statusElement.innerHTML = '<i class="fas fa-circle"></i> Connecting...';

        try {
            const isOnline = await window.apiService.checkBackendStatus();
            
            if (isOnline) {
                statusElement.className = 'status online';
                statusElement.innerHTML = '<i class="fas fa-circle"></i> Backend Online';
            } else {
                throw new Error('Backend offline');
            }
        } catch (error) {
            statusElement.className = 'status offline';
            statusElement.innerHTML = '<i class="fas fa-circle"></i> Backend Offline';
            console.error('Backend connection failed:', error);
        }
    }

    async loadAlgorithms() {
        const gridElement = document.getElementById('algorithms-grid');
        
        try {
            const algorithms = await window.apiService.getAlgorithms();
            this.renderAlgorithms(algorithms);
        } catch (error) {
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
                <div class="algorithm-card" onclick="app.openAlgorithmModal('${algorithm.name}')">
                    <h3>
                        <i class="${icon}"></i>
                        ${algorithm.name}
                    </h3>
                    <p>${algorithm.description}</p>
                    <span class="algorithm-type">${algorithm.type}</span>
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
        
        return iconMap[type.toLowerCase()] || iconMap['default'];
    }

    async openAlgorithmModal(algorithmName) {
        try {
            const algorithms = await window.apiService.getAlgorithms();
            const algorithm = algorithms.find(a => a.name === algorithmName);
            
            if (!algorithm) {
                throw new Error('Algorithm not found');
            }

            this.currentAlgorithm = algorithm;
            
            // Update modal content
            document.getElementById('modal-title').textContent = algorithm.name;
            document.getElementById('algorithm-description').textContent = algorithm.description;
            
            // Reset modal state
            document.getElementById('training-status').classList.add('hidden');
            document.getElementById('results-section').classList.add('hidden');
            document.getElementById('train-btn').disabled = false;
            
            // Show modal
            document.getElementById('algorithm-modal').classList.add('active');
            document.body.style.overflow = 'hidden';
            
        } catch (error) {
            console.error('Failed to open algorithm modal:', error);
            this.showError('Failed to load algorithm details');
        }
    }

    closeModal() {
        document.getElementById('algorithm-modal').classList.remove('active');
        document.body.style.overflow = '';
        this.currentAlgorithm = null;
    }

    async trainCurrentAlgorithm() {
        if (!this.currentAlgorithm || this.isTraining) return;

        this.isTraining = true;
        const trainBtn = document.getElementById('train-btn');
        const trainingStatus = document.getElementById('training-status');
        const resultsSection = document.getElementById('results-section');

        // Update UI for training state
        trainBtn.disabled = true;
        trainBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training...';
        trainingStatus.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        try {
            const results = await window.apiService.trainAlgorithm(this.currentAlgorithm.name);
            this.displayResults(results);
        } catch (error) {
            console.error('Training failed:', error);
            this.showError('Training failed. Please check the backend connection.');
        } finally {
            // Reset UI state
            this.isTraining = false;
            trainBtn.disabled = false;
            trainBtn.innerHTML = '<i class="fas fa-play"></i> Start Training';
            trainingStatus.classList.add('hidden');
        }
    }

    displayResults(results) {
        const resultsSection = document.getElementById('results-section');
        const resultsContent = document.getElementById('results-content');
        
        // Format results for display
        let formattedResults = '';
        
        if (typeof results === 'object') {
            formattedResults = JSON.stringify(results, null, 2);
        } else {
            formattedResults = results.toString();
        }
        
        resultsContent.textContent = formattedResults;
        resultsSection.classList.remove('hidden');
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
});

// Add some helpful global functions for debugging
window.refreshApp = () => window.app?.refresh();
window.checkBackend = () => window.app?.checkBackendConnection();

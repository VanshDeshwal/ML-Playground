// Optimized Algorithm Page Controller
class AlgorithmPageApp {
    constructor() {
        this.currentAlgorithm = null;
        this.algorithmId = null;
        
        // Component managers
        this.statusManager = null;
        this.hyperparameterManager = null;
        this.codeModalManager = null;
        this.trainingManager = null;
        this.resultsDisplay = null;
        
        this.init();
    }

    async init() {
        // Get algorithm ID from URL
        if (!this.parseAlgorithmId()) return;

        // Initialize core components
        this.initializeManagers();
        
        // Check backend and load algorithm
        await this.statusManager.checkConnection();
        await this.loadAlgorithm();
        
        // Setup view code button
        this.setupViewCodeButton();
        
        // Initialize results display (async but not awaited for faster page load)
        this.initializeResultsDisplay().catch(console.error);
    }

    parseAlgorithmId() {
        const urlParams = new URLSearchParams(window.location.search);
        this.algorithmId = urlParams.get('id') || urlParams.get('algorithm');
        
        if (!this.algorithmId) {
            this.showError('No algorithm ID provided. Redirecting to homepage...');
            setTimeout(() => window.location.href = 'index.html', 2000);
            return false;
        }
        return true;
    }

    initializeManagers() {
        try {
            this.statusManager = new StatusManager();
            this.hyperparameterManager = new HyperparameterManager('hyperparameter-controls');
            this.codeModalManager = new CodeModalManager();
        } catch (error) {
            console.error('Failed to initialize managers:', error);
        }
    }

    async loadAlgorithm() {
        try {
            console.log('Loading algorithm for ID:', this.algorithmId);
            const algorithms = await window.apiService.getAlgorithms();
            
            const algorithm = algorithms.find(a => a.id === this.algorithmId);
            if (!algorithm) {
                throw new Error(`Algorithm with ID '${this.algorithmId}' not found`);
            }

            this.currentAlgorithm = algorithm;
            this.displayAlgorithmInfo(algorithm);
            this.hyperparameterManager.generateControls(algorithm.hyperparameters || {});
            
            // Initialize training manager after hyperparameters are set up
            this.trainingManager = new TrainingManager(
                this.algorithmId, 
                this.hyperparameterManager, 
                this.resultsDisplay
            );
            
        } catch (error) {
            console.error('Failed to load algorithm:', error);
            this.showError(`Failed to load algorithm details: ${error.message}`);
        }
    }

    displayAlgorithmInfo(algorithm) {
        // Update basic info
        document.getElementById('algorithm-name').textContent = algorithm.name;
        document.getElementById('algorithm-description').textContent = algorithm.description;
        
        // Set algorithm icon
        const iconMap = {
            'regression': 'fas fa-chart-line',
            'classification': 'fas fa-layer-group',
            'clustering': 'fas fa-project-diagram',
            'neural network': 'fas fa-brain',
            'ensemble': 'fas fa-sitemap',
            'default': 'fas fa-cog'
        };
        const icon = iconMap[algorithm.type] || iconMap.default;
        document.getElementById('algorithm-icon').className = icon;
        
        // Update page title
        document.title = `${algorithm.name} - ML Playground`;
    }

    setupViewCodeButton() {
        const viewCodeBtn = document.getElementById('view-code-btn');
        if (viewCodeBtn) {
            viewCodeBtn.addEventListener('click', () => {
                this.codeModalManager.show(this.algorithmId);
            });
        } else {
            console.error('View Code button not found!');
        }
    }

    async initializeResultsDisplay() {
        try {
            console.log('🔧 Initializing results display...');
            
            const resultsContainer = document.getElementById('results-display');
            if (!resultsContainer) {
                console.warn('Results display container not found');
                return;
            }
            
            // Initialize enhanced results display
            this.resultsDisplay = new ResultsDisplay(resultsContainer);
            
            // Update training manager with results display
            if (this.trainingManager) {
                this.trainingManager.resultsDisplay = this.resultsDisplay;
            }
            
            console.log('Results display initialized successfully');
        } catch (error) {
            console.error('Failed to initialize results display:', error);
        }
    }

    showError(message) {
        // Simple error display - could be enhanced with a toast system
        alert(message);
    }
}

// Initialize the algorithm page app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.algorithmApp = new AlgorithmPageApp();
});

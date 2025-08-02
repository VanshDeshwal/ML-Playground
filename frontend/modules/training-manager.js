// Training orchestrator - handles the training process and UI updates
class TrainingManager {
    constructor(algorithmId, hyperparameterManager, resultsDisplay) {
        this.algorithmId = algorithmId;
        this.hyperparameterManager = hyperparameterManager;
        this.resultsDisplay = resultsDisplay;
        this.isTraining = false;
        
        this.trainBtn = document.getElementById('train-btn');
        this.trainingStatus = document.getElementById('training-status');
        this.resultsSection = document.getElementById('results-section');
        
        this.setupEventListeners();
    }

    setupEventListeners() {
        if (this.trainBtn) {
            this.trainBtn.addEventListener('click', () => this.startTraining());
        }
    }

    async startTraining() {
        if (this.isTraining) return;
        
        this.isTraining = true;
        this.updateTrainingUI(true);

        try {
            const hyperparameters = this.hyperparameterManager.getValues();
            
            // Use training API for rich results
            const results = await window.apiService.trainAlgorithm(this.algorithmId, hyperparameters);
            console.log('Training results received:', results);
            
            // Display results
            if (this.resultsDisplay && results.success) {
                console.log('Using enhanced results display');
                this.resultsDisplay.displayResults(results);
                this.resultsSection.classList.remove('hidden');
            } else {
                console.log('Using fallback display');
                this.displayFallbackResults(results);
            }
            
        } catch (error) {
            console.error('Training failed:', error);
            this.handleTrainingError(error);
        } finally {
            this.updateTrainingUI(false);
            this.isTraining = false;
        }
    }

    updateTrainingUI(isTraining) {
        if (!this.trainBtn) return;

        this.trainBtn.disabled = isTraining;
        
        if (isTraining) {
            this.trainBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training...';
            this.trainingStatus?.classList.remove('hidden');
            this.resultsSection?.classList.add('hidden');
        } else {
            this.trainBtn.innerHTML = '<i class="fas fa-play"></i> Start Training';
            this.trainingStatus?.classList.add('hidden');
        }
    }

    handleTrainingError(error) {
        if (this.resultsDisplay) {
            this.resultsDisplay.displayError(error.message || 'Training failed');
            this.resultsSection?.classList.remove('hidden');
        } else {
            let errorMessage = 'Training failed. ';
            if (error.message && error.message.includes('HTTP error!')) {
                errorMessage += error.message;
            } else {
                errorMessage += `Error: ${error.message || 'Unknown error'}`;
            }
            this.showError(errorMessage);
        }
    }

    displayFallbackResults(results) {
        const resultsContent = document.getElementById('enhanced-results-display') || 
                               document.getElementById('results-content');
        
        if (!resultsContent) {
            console.warn('No results container found');
            return;
        }

        const safeNumber = (value, decimals = 4) => {
            return (typeof value === 'number') ? value.toFixed(decimals) : 'N/A';
        };

        const basicHTML = `
            <div class="basic-results">
                <h3>Training Results</h3>
                <p><strong>Success:</strong> ${results.success ? 'Yes' : 'No'}</p>
                ${results.error ? `<p class="error"><strong>Error:</strong> ${results.error}</p>` : ''}
                ${results.your_implementation && results.your_implementation.metrics ? `
                    <div class="metrics">
                        <h4>Performance Metrics</h4>
                        <pre>${JSON.stringify(results.your_implementation.metrics, null, 2)}</pre>
                    </div>
                ` : ''}
            </div>
        `;
        
        resultsContent.innerHTML = basicHTML;
        this.resultsSection?.classList.remove('hidden');
    }

    showError(message) {
        // Simple error display - could be enhanced with a toast system
        alert(message);
    }
}

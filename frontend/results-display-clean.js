/**
 * Clean, modular Results Display for ML Playground
 * Main coordinator for results visualization
 */
class ResultsDisplay {
    constructor(container) {
        // Handle both string ID and DOM element
        if (typeof container === 'string') {
            const element = document.getElementById(container);
            if (!element) {
                throw new Error(`No element found with ID: ${container}`);
            }
            this.container = element;
        } else if (container && container.nodeType === 1) {
            this.container = container;
        } else {
            throw new Error('Container must be a string ID or DOM element');
        }
        
        // Initialize modules
        this.chartManager = new ChartManager();
        this.metricsDisplay = new MetricsDisplay();
        this.trainingDetails = new TrainingDetails();
        
        this.initializeContainer();
        console.log('✅ ResultsDisplay initialized');
    }

    /**
     * Initialize the main container structure
     */
    initializeContainer() {
        this.container.innerHTML = `
            <div class="results-container">
                <div class="results-sections">
                    <section class="sklearn-comparison-section">
                        <h3>📊 Performance Comparison</h3>
                        <div id="comparison-content">
                            <div class="loading">Calculating metrics...</div>
                        </div>
                    </section>
                    
                    <section class="visualizations-section">
                        <h3>📈 Visualizations</h3>
                        <div id="charts-grid" class="charts-grid">
                            <div class="loading">Generating charts...</div>
                        </div>
                    </section>
                    
                    <section class="training-details-section">
                        <h3>🔧 Training Details</h3>
                        <div id="training-details">
                            <div class="loading">Loading details...</div>
                        </div>
                    </section>
                </div>
            </div>
        `;
    }

    /**
     * Main method to display training results
     */
    async displayResults(result) {
        console.log('📊 Displaying results:', result);
        
        try {
            // Process result through data contract validation
            const dataContract = new DataContractValidator();
            const processedResult = dataContract.processTrainingResult(result);
            console.log('✅ Data contract processing completed');

            // Check if training was successful
            if (!processedResult.success) {
                console.error('❌ Training failed:', processedResult.error);
                this.displayError(processedResult.error || 'Training failed');
                return;
            }

            console.log('✅ Result successful, displaying results');
            
            // Display components using modules
            this.metricsDisplay.displayComparison(processedResult);
            await this.chartManager.renderCharts(processedResult);
            this.trainingDetails.displayTrainingDetails(processedResult);
            
            console.log('✅ All results displayed successfully');
            
        } catch (error) {
            console.error('❌ Error displaying results:', error);
            this.displayError(`Error displaying results: ${error.message}`);
        }
    }

    /**
     * Display error message
     */
    displayError(error) {
        this.container.innerHTML = `
            <div class="error-container">
                <div class="error-icon">❌</div>
                <h3>Training Error</h3>
                <p>${error}</p>
            </div>
        `;
    }

    /**
     * Clear all displayed results
     */
    clearResults() {
        this.metricsDisplay.clearMetrics();
        this.trainingDetails.clearDetails();
        this.initializeContainer();
    }

    /**
     * Export current results data
     */
    exportResults() {
        return {
            metrics: this.metricsDisplay.metrics,
            details: this.trainingDetails.exportDetails(),
            timestamp: new Date().toISOString()
        };
    }
}

// Export for use in other files
window.ResultsDisplay = ResultsDisplay;

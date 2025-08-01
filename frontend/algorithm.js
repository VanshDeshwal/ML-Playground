// Algorithm page controller
class AlgorithmPageApp {
    constructor() {
        this.currentAlgorithm = null;
        this.isTraining = false;
        this.algorithmId = null;
        
        this.init();
    }

    async init() {
        // Get algorithm ID from URL parameters (support both 'id' and 'algorithm')
        const urlParams = new URLSearchParams(window.location.search);
        this.algorithmId = urlParams.get('id') || urlParams.get('algorithm');
        
        if (!this.algorithmId) {
            this.showError('No algorithm ID provided. Redirecting to homepage...');
            setTimeout(() => {
                window.location.href = 'index.html';
            }, 2000);
            return;
        }

        // Check backend connection
        await this.checkBackendConnection();
        
        // Load algorithm details
        await this.loadAlgorithm();
        
        // Setup event listeners
        this.setupEventListeners();
    }

    async checkBackendConnection() {
        const statusIndicator = document.getElementById('status-indicator');
        const statusDot = statusIndicator.querySelector('.status-dot');
        const statusText = statusIndicator.querySelector('.status-text');
        
        try {
            const isConnected = await window.apiService.checkBackendStatus();
            
            if (isConnected) {
                statusDot.className = 'status-dot status-connected';
                statusText.textContent = 'Backend Connected';
            } else {
                statusDot.className = 'status-dot status-error';
                statusText.textContent = 'Backend Disconnected';
            }
        } catch (error) {
            statusDot.className = 'status-dot status-error';
            statusText.textContent = 'Connection Failed';
        }
    }

    async loadAlgorithm() {
        try {
            const algorithms = await window.apiService.getAlgorithms();
            const algorithm = algorithms.find(a => a.id === this.algorithmId);
            
            if (!algorithm) {
                throw new Error('Algorithm not found');
            }

            this.currentAlgorithm = algorithm;
            this.displayAlgorithmInfo(algorithm);
            this.generateHyperparameterControls(algorithm.hyperparameters || {});
            
        } catch (error) {
            console.error('Failed to load algorithm:', error);
            this.showError('Failed to load algorithm details. Please try again.');
        }
    }

    displayAlgorithmInfo(algorithm) {
        document.getElementById('algorithm-name').textContent = algorithm.name;
        document.getElementById('algorithm-description').textContent = algorithm.description;
        document.getElementById('algorithm-type').textContent = algorithm.type;
        document.getElementById('algorithm-id').textContent = algorithm.id;
        
        // Set algorithm icon
        const icon = this.getAlgorithmIcon(algorithm.type);
        document.getElementById('algorithm-icon').className = icon;
        
        // Update page title
        document.title = `${algorithm.name} - ML Playground`;
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

    generateHyperparameterControls(hyperparameters) {
        const container = document.getElementById('hyperparameter-controls');
        
        if (Object.keys(hyperparameters).length === 0) {
            container.innerHTML = `
                <div class="no-hyperparameters">
                    <i class="fas fa-info-circle"></i>
                    <p>This algorithm uses default parameters with no adjustable hyperparameters.</p>
                </div>
            `;
            return;
        }

        const controlsHTML = Object.entries(hyperparameters).map(([key, param]) => {
            return this.createHyperparameterControl(key, param);
        }).join('');

        container.innerHTML = controlsHTML;
    }

    createHyperparameterControl(key, param) {
        const { type, default: defaultValue, min, max, description } = param;
        
        if (type === 'float' || type === 'int') {
            const step = type === 'float' ? '0.001' : '1';
            return `
                <div class="control-group">
                    <label for="${key}" class="control-label">
                        ${key}
                        <span class="control-description">${description}</span>
                    </label>
                    <div class="control-input">
                        <input 
                            type="number" 
                            id="${key}" 
                            name="${key}" 
                            value="${defaultValue}" 
                            min="${min}" 
                            max="${max}" 
                            step="${step}"
                            class="number-input"
                        >
                        <span class="control-range">Range: ${min} - ${max}</span>
                    </div>
                </div>
            `;
        } else if (type === 'select') {
            const options = param.options.map(option => 
                `<option value="${option}" ${option === defaultValue ? 'selected' : ''}>${option}</option>`
            ).join('');
            
            return `
                <div class="control-group">
                    <label for="${key}" class="control-label">
                        ${key}
                        <span class="control-description">${description}</span>
                    </label>
                    <select id="${key}" name="${key}" class="select-input">
                        ${options}
                    </select>
                </div>
            `;
        }
        
        return '';
    }

    setupEventListeners() {
        const trainBtn = document.getElementById('train-btn');
        trainBtn.addEventListener('click', () => this.startTraining());
        
        const viewCodeBtn = document.getElementById('view-code-btn');
        if (viewCodeBtn) {
            viewCodeBtn.addEventListener('click', () => {
                console.log('View Code button clicked!');
                this.showCodeModal();
            });
            console.log('View Code button event listener attached successfully');
        } else {
            console.error('View Code button not found!');
        }
        
        // Modal close events
        const codeModal = document.getElementById('code-modal');
        const codeModalClose = document.getElementById('code-modal-close');
        const codeModalBackdrop = document.getElementById('code-modal-backdrop');
        const copyCodeBtn = document.getElementById('copy-code-btn');
        
        if (codeModalClose) {
            codeModalClose.addEventListener('click', () => this.hideCodeModal());
        }
        if (codeModalBackdrop) {
            codeModalBackdrop.addEventListener('click', () => this.hideCodeModal());
        }
        if (copyCodeBtn) {
            copyCodeBtn.addEventListener('click', () => this.copyCode());
        }
        
        // Close modal on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && codeModal && !codeModal.classList.contains('hidden')) {
                this.hideCodeModal();
            }
        });
    }

    getHyperparameters() {
        const hyperparameters = {};
        const controls = document.querySelectorAll('#hyperparameter-controls input, #hyperparameter-controls select');
        
        controls.forEach(control => {
            const value = control.type === 'number' ? 
                (control.step === '1' ? parseInt(control.value) : parseFloat(control.value)) : 
                control.value;
            hyperparameters[control.name] = value;
        });
        
        return hyperparameters;
    }

    async startTraining() {
        if (this.isTraining) return;
        
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
            const hyperparameters = this.getHyperparameters();
            console.log('Training with hyperparameters:', hyperparameters);
            console.log('Algorithm ID:', this.algorithmId);
            const results = await window.apiService.trainAlgorithm(this.algorithmId, hyperparameters);
            console.log('Training results:', results);
            this.displayResults(results);
        } catch (error) {
            console.error('Training failed:', error);
            // Show detailed error message from the API response
            let errorMessage = 'Training failed. ';
            if (error.message && error.message.includes('HTTP error!')) {
                errorMessage += error.message;
            } else {
                errorMessage += `Error: ${error.message || 'Unknown error'}`;
            }
            this.showError(errorMessage);
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
        
        // Safely get values with fallbacks
        const safeNumber = (value, decimals = 4) => {
            return (typeof value === 'number') ? value.toFixed(decimals) : 'N/A';
        };
        
        const safePercentage = (value, decimals = 4) => {
            return (typeof value === 'number') ? `${(value * 100).toFixed(decimals)}%` : 'N/A';
        };
        
        // Create formatted HTML display
        const formattedHTML = `
            <div class="results-container">
                <div class="results-header">
                    <h3><i class="fas fa-chart-line"></i> Training Results</h3>
                    <div class="algorithm-badge">${results.algorithm_id}</div>
                </div>
                
                <div class="results-grid">
                    <div class="result-card">
                        <h4><i class="fas fa-trophy"></i> Performance</h4>
                        <div class="metric">
                            <span class="metric-label">Algorithm Score:</span>
                            <span class="metric-value">${safePercentage(results.custom_score)}</span>
                        </div>
                        ${results.sklearn_score ? `
                            <div class="metric">
                                <span class="metric-label">Sklearn Score:</span>
                                <span class="metric-value">${safePercentage(results.sklearn_score)}</span>
                            </div>
                        ` : ''}
                        <div class="metric">
                            <span class="metric-label">Training Time:</span>
                            <span class="metric-value">${safeNumber(results.training_time, 3)}s</span>
                        </div>
                    </div>
                    
                    <div class="result-card">
                        <h4><i class="fas fa-calculator"></i> Metrics</h4>
                        ${this.renderMetrics(results.metrics)}
                    </div>
                    
                    <div class="result-card">
                        <h4><i class="fas fa-cogs"></i> Training Details</h4>
                        ${this.renderTrainingDetails(results.metadata)}
                    </div>
                    
                    <div class="result-card">
                        <h4><i class="fas fa-database"></i> Dataset Info</h4>
                        ${this.renderDatasetInfo(results.metadata)}
                    </div>
                </div>
                
                ${this.renderVisualization(results.metadata)}
            </div>
        `;
        
        resultsContent.innerHTML = formattedHTML;
        resultsSection.classList.remove('hidden');
    }
    
    renderMetrics(metrics) {
        if (!metrics) return '<p>No metrics available</p>';
        
        let html = '';
        for (const [key, value] of Object.entries(metrics)) {
            const label = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            if (typeof value === 'number') {
                if (key.includes('score') || key.includes('accuracy')) {
                    html += `<div class="metric">
                        <span class="metric-label">${label}:</span>
                        <span class="metric-value">${(value * 100).toFixed(4)}%</span>
                    </div>`;
                } else {
                    html += `<div class="metric">
                        <span class="metric-label">${label}:</span>
                        <span class="metric-value">${value.toFixed(6)}</span>
                    </div>`;
                }
            }
        }
        return html || '<p>No numeric metrics</p>';
    }
    
    renderTrainingDetails(metadata) {
        if (!metadata) return '<p>No training details available</p>';
        
        let html = '';
        
        // Common training details
        if (metadata.n_iterations !== undefined) {
            html += `<div class="metric">
                <span class="metric-label">Iterations:</span>
                <span class="metric-value">${metadata.n_iterations}</span>
            </div>`;
        }
        
        if (metadata.learning_rate !== undefined) {
            html += `<div class="metric">
                <span class="metric-label">Learning Rate:</span>
                <span class="metric-value">${metadata.learning_rate}</span>
            </div>`;
        }
        
        if (metadata.converged !== undefined) {
            html += `<div class="metric">
                <span class="metric-label">Converged:</span>
                <span class="metric-value ${metadata.converged ? 'success' : 'warning'}">
                    ${metadata.converged ? '✓ Yes' : '✗ No'}
                </span>
            </div>`;
        }
        
        // Algorithm-specific details
        if (metadata.intercept !== undefined) {
            html += `<div class="metric">
                <span class="metric-label">Intercept:</span>
                <span class="metric-value">${metadata.intercept.toFixed(6)}</span>
            </div>`;
        }
        
        if (metadata.n_clusters !== undefined) {
            html += `<div class="metric">
                <span class="metric-label">Clusters:</span>
                <span class="metric-value">${metadata.n_clusters}</span>
            </div>`;
        }
        
        if (metadata.k !== undefined) {
            html += `<div class="metric">
                <span class="metric-label">K Value:</span>
                <span class="metric-value">${metadata.k}</span>
            </div>`;
        }
        
        return html || '<p>No training details</p>';
    }
    
    renderDatasetInfo(metadata) {
        if (!metadata) return '<p>No dataset info available</p>';
        
        let html = '';
        
        if (metadata.n_samples !== undefined) {
            html += `<div class="metric">
                <span class="metric-label">Samples:</span>
                <span class="metric-value">${metadata.n_samples}</span>
            </div>`;
        }
        
        if (metadata.n_features !== undefined) {
            html += `<div class="metric">
                <span class="metric-label">Features:</span>
                <span class="metric-value">${metadata.n_features}</span>
            </div>`;
        }
        
        return html || '<p>No dataset info</p>';
    }
    
    renderVisualization(metadata) {
        if (!metadata || !metadata.loss_history) return '';
        
        return `
            <div class="loss-chart-container">
                <h4><i class="fas fa-chart-area"></i> Training Loss History</h4>
                <div class="loss-summary">
                    <span>Initial Loss: ${metadata.loss_history[0].toFixed(2)}</span>
                    <span>Final Loss: ${(metadata.final_loss || metadata.loss_history[metadata.loss_history.length - 1]).toFixed(6)}</span>
                    <span>Reduction: ${((1 - (metadata.final_loss || metadata.loss_history[metadata.loss_history.length - 1]) / metadata.loss_history[0]) * 100).toFixed(2)}%</span>
                </div>
                <div class="loss-visualization">
                    ${this.createLossChart(metadata.loss_history)}
                </div>
            </div>
        `;
    }
                
    createLossChart(lossHistory) {
        const maxLoss = Math.max(...lossHistory);
        const minLoss = Math.min(...lossHistory);
        const range = maxLoss - minLoss;
        
        const points = lossHistory.map((loss, index) => {
            const x = (index / (lossHistory.length - 1)) * 100;
            const y = 100 - ((loss - minLoss) / range) * 100;
            return `${x},${y}`;
        }).join(' ');
        
        return `
            <svg class="loss-chart" viewBox="0 0 100 50" preserveAspectRatio="none">
                <polyline points="${points}" fill="none" stroke="var(--primary-color)" stroke-width="0.5"/>
                <circle cx="${100}" cy="${100 - ((lossHistory[lossHistory.length - 1] - minLoss) / range) * 100}" r="1" fill="var(--primary-color)"/>
            </svg>
        `;
    }

    showError(message) {
        // Simple error display - could be enhanced with a toast system
        alert(message);
    }

    async showCodeModal() {
        const modal = document.getElementById('code-modal');
        const modalTitle = document.getElementById('code-modal-title');
        const codeFilename = document.getElementById('code-filename');
        const codeLanguage = document.getElementById('code-language');
        const codeDescription = document.getElementById('code-description');
        const codeContent = document.getElementById('code-content');

        try {
            // Show loading state
            modalTitle.textContent = 'Loading Code...';
            codeFilename.textContent = 'loading...';
            codeLanguage.textContent = 'Python';
            codeDescription.textContent = 'Fetching implementation details...';
            codeContent.textContent = 'Loading code snippet...';
            
            // Show modal
            modal.classList.remove('hidden');
            
            // Fetch code snippet from API
            const response = await fetch(`${window.APP_CONFIG.API_BASE_URL}/algorithms/${this.algorithmId}/code`);
            
            if (!response.ok) {
                throw new Error(`Failed to fetch code: ${response.status}`);
            }
            
            const codeData = await response.json();
            
            // Update modal content
            modalTitle.textContent = `${codeData.description}`;
            codeFilename.textContent = codeData.filename;
            codeLanguage.textContent = codeData.language.toUpperCase();
            codeDescription.textContent = codeData.description;
            codeContent.textContent = codeData.code;
            
        } catch (error) {
            console.error('Failed to load code:', error);
            modalTitle.textContent = 'Error Loading Code';
            codeFilename.textContent = 'error';
            codeDescription.textContent = 'Failed to load the implementation code. Please try again.';
            codeContent.textContent = `Error: ${error.message}`;
        }
    }

    hideCodeModal() {
        const modal = document.getElementById('code-modal');
        modal.classList.add('hidden');
    }

    copyCode() {
        const codeContent = document.getElementById('code-content');
        const text = codeContent.textContent;
        
        navigator.clipboard.writeText(text).then(() => {
            const copyBtn = document.getElementById('copy-code-btn');
            const originalText = copyBtn.innerHTML;
            
            // Show success feedback
            copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
            copyBtn.style.background = 'var(--accent-color)';
            
            // Reset after 2 seconds
            setTimeout(() => {
                copyBtn.innerHTML = originalText;
                copyBtn.style.background = '';
            }, 2000);
        }).catch(err => {
            console.error('Failed to copy code:', err);
            alert('Failed to copy code to clipboard');
        });
    }
}

// Initialize the algorithm page app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.algorithmApp = new AlgorithmPageApp();
});

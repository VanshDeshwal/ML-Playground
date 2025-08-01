// Enhanced UI components for rich training results display
class EnhancedResultsDisplay {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.plotlyLoaded = false;
        this.initializePlotly();
    }

    // Initialize Plotly.js for charts
    async initializePlotly() {
        if (window.Plotly) {
            this.plotlyLoaded = true;
            console.log('Plotly already loaded');
            return;
        }

        try {
            // Load Plotly.js from CDN - using explicit version instead of latest
            const script = document.createElement('script');
            script.src = 'https://cdn.plot.ly/plotly-2.32.0.min.js';
            script.onload = () => {
                this.plotlyLoaded = true;
                console.log('Plotly.js loaded successfully');
            };
            script.onerror = () => {
                console.error('Failed to load Plotly.js');
                this.plotlyLoaded = false;
            };
            document.head.appendChild(script);
            
            // Wait for script to load
            await new Promise((resolve) => {
                script.onload = () => {
                    this.plotlyLoaded = true;
                    console.log('Plotly.js loaded and ready');
                    resolve();
                };
                script.onerror = () => {
                    this.plotlyLoaded = false;
                    console.error('Failed to load Plotly.js from CDN');
                    resolve();
                };
            });
        } catch (error) {
            console.error('Error loading Plotly.js:', error);
            this.plotlyLoaded = false;
        }
    }

    // Display enhanced training results
    displayResults(result) {
        console.log('EnhancedResultsDisplay.displayResults called with:', result);
        
        if (!result.success) {
            console.log('Result unsuccessful, showing error');
            this.displayError(result.error || 'Training failed');
            return;
        }

        console.log('Result successful, displaying enhanced results');
        
        // Clear previous results
        this.container.innerHTML = '';

        // Create main results container
        const resultsContainer = document.createElement('div');
        resultsContainer.className = 'enhanced-results-container';
        resultsContainer.innerHTML = `
            <div class="results-content">
                <div class="comparison-section">
                    <h4>Sklearn Comparison</h4>
                    <div id="comparison-display" class="comparison-container"></div>
                </div>
                <div class="charts-section">
                    <h4>Visualizations</h4>
                    <div id="charts-container" class="charts-grid"></div>
                </div>
                <div class="details-section">
                    <h4>Training Details</h4>
                    <div id="details-display" class="details-container"></div>
                </div>
            </div>
            <style>
                .charts-grid {
                    display: flex;
                    flex-direction: column;
                    gap: 30px;
                    margin: 20px 0;
                }
                .chart-container {
                    background: #fafafa;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .comparison-section, .details-section {
                    margin: 30px 0;
                    padding: 20px;
                    background: #f8f9fa;
                    border-radius: 8px;
                    border-left: 4px solid #007bff;
                }
                .results-content > div:not(:last-child) {
                    margin-bottom: 40px;
                }
            </style>
        `;

        this.container.appendChild(resultsContainer);

        // Display each section - handle the actual API response structure
        console.log('Full API result structure:', JSON.stringify(result, null, 2));
        console.log('Chart data:', result.charts);
        console.log('My implementation:', result.your_implementation);
        
        // Remove metrics display as it's redundant with sklearn comparison
        this.displayComparison(result);
        this.displayCharts(result.charts);  // Changed from chart_data to charts
        this.displayDetails(result);
    }

    // Display performance metrics
    displayMetrics(metrics) {
        console.log('displayMetrics called with:', metrics);
        const metricsContainer = document.getElementById('metrics-display');
        if (!metrics) {
            console.log('No metrics provided');
            metricsContainer.innerHTML = '<p>No metrics available</p>';
            return;
        }

        let metricsHTML = '';
        
        // Handle direct metrics object (our API structure)
        if (typeof metrics === 'object' && !metrics.primary_metrics && !metrics.additional_metrics) {
            console.log('Processing direct metrics object');
            // Create primary metrics from key performance indicators
            const primaryMetrics = {};
            const additionalMetrics = {};
            
            // Categorize metrics
            const primaryKeys = ['r2_score', 'mse', 'mae', 'rmse', 'accuracy', 'f1_score'];
            
            Object.entries(metrics).forEach(([key, value]) => {
                if (value !== null && value !== undefined) {
                    if (primaryKeys.includes(key)) {
                        primaryMetrics[key] = value;
                    } else {
                        additionalMetrics[key] = value;
                    }
                }
            });
            
            // Display primary metrics
            if (Object.keys(primaryMetrics).length > 0) {
                console.log('Found primary metrics:', primaryMetrics);
                Object.entries(primaryMetrics).forEach(([key, value]) => {
                    const displayName = this.formatMetricName(key);
                    const formattedValue = typeof value === 'number' ? value.toFixed(4) : value;
                    metricsHTML += `
                        <div class="metric-card primary">
                            <div class="metric-name">${displayName}</div>
                            <div class="metric-value">${formattedValue}</div>
                        </div>
                    `;
                });
            }
            
            // Display additional metrics
            if (Object.keys(additionalMetrics).length > 0) {
                console.log('Found additional metrics:', additionalMetrics);
                Object.entries(additionalMetrics).forEach(([key, value]) => {
                    const displayName = this.formatMetricName(key);
                    const formattedValue = typeof value === 'number' ? value.toFixed(4) : value;
                    metricsHTML += `
                        <div class="metric-card additional">
                            <div class="metric-name">${displayName}</div>
                            <div class="metric-value">${formattedValue}</div>
                        </div>
                    `;
                });
            }
        } else {
            // Handle structured metrics object
            // Primary metrics
            if (metrics.primary_metrics) {
                console.log('Found primary metrics:', metrics.primary_metrics);
                Object.entries(metrics.primary_metrics).forEach(([key, value]) => {
                    const displayName = this.formatMetricName(key);
                    const formattedValue = typeof value === 'number' ? value.toFixed(4) : value;
                    metricsHTML += `
                        <div class="metric-card primary">
                            <div class="metric-name">${displayName}</div>
                            <div class="metric-value">${formattedValue}</div>
                        </div>
                    `;
                });
            }

            // Additional metrics
            if (metrics.additional_metrics) {
                console.log('Found additional metrics:', metrics.additional_metrics);
                Object.entries(metrics.additional_metrics).forEach(([key, value]) => {
                    const displayName = this.formatMetricName(key);
                    const formattedValue = typeof value === 'number' ? value.toFixed(4) : value;
                    metricsHTML += `
                        <div class="metric-card additional">
                            <div class="metric-name">${displayName}</div>
                            <div class="metric-value">${formattedValue}</div>
                        </div>
                    `;
                });
            }
        }

        if (!metricsHTML) {
            metricsHTML = '<p>No metrics available</p>';
        }

        console.log('Setting metricsHTML:', metricsHTML);
        metricsContainer.innerHTML = metricsHTML;
    }

    // Display charts using Plotly.js
    async displayCharts(chartData) {
        console.log('displayCharts called with:', chartData);
        const chartsContainer = document.getElementById('charts-container');
        
        if (!chartData) {
            console.log('No chart data provided');
            chartsContainer.innerHTML = '<p>Charts not available</p>';
            return;
        }

        // Ensure Plotly is loaded
        if (!this.plotlyLoaded) {
            console.log('Plotly not loaded, initializing...');
            await this.initializePlotly();
        }

        // Wait for Plotly to be ready with extended timeout
        let attempts = 0;
        while (!window.Plotly && attempts < 20) {
            console.log(`Waiting for Plotly... attempt ${attempts + 1}`);
            await new Promise(resolve => setTimeout(resolve, 500));
            attempts++;
        }

        if (!window.Plotly) {
            console.log('Plotly failed to load after 20 attempts');
            chartsContainer.innerHTML = '<p>Unable to load charting library. Please check your internet connection.</p>';
            return;
        }

        console.log('Plotly is ready, creating charts...');
        chartsContainer.innerHTML = '';
        let chartsCreated = 0;

        // Create charts based on available data structure from our API
        if (chartData.loss_curve && chartData.loss_curve.training_history) {
            console.log('Creating loss curve chart with data:', chartData.loss_curve);
            try {
                this.createLossCurveChart(chartData.loss_curve, chartsContainer);
                chartsCreated++;
            } catch (error) {
                console.error('Error creating loss curve chart:', error);
            }
        }

        if (chartData.scatter_plot) {
            console.log('Creating scatter plot with data:', chartData.scatter_plot);
            try {
                this.createScatterPlotChart(chartData.scatter_plot, chartsContainer);
                chartsCreated++;
            } catch (error) {
                console.error('Error creating scatter plot:', error);
            }
        }

        if (chartData.residuals_plot) {
            console.log('Creating residuals chart with data:', chartData.residuals_plot);
            try {
                this.createResidualsPlotChart(chartData.residuals_plot, chartsContainer);
                chartsCreated++;
            } catch (error) {
                console.error('Error creating residuals chart:', error);
            }
        }

        // Legacy support for other chart types
        if (chartData.loss_history) {
            console.log('Creating loss history chart');
            try {
                this.createLossChart(chartData.loss_history, chartsContainer);
                chartsCreated++;
            } catch (error) {
                console.error('Error creating loss history chart:', error);
            }
        }

        if (chartData.predictions_vs_actual) {
            console.log('Creating predictions chart');
            try {
                this.createPredictionsChart(chartData.predictions_vs_actual, chartsContainer);
                chartsCreated++;
            } catch (error) {
                console.error('Error creating predictions chart:', error);
            }
        }

        if (chartData.feature_importance) {
            console.log('Creating feature importance chart');
            try {
                this.createFeatureImportanceChart(chartData.feature_importance, chartsContainer);
                chartsCreated++;
            } catch (error) {
                console.error('Error creating feature importance chart:', error);
            }
        }

        if (chartsCreated === 0) {
            console.log('No charts could be created');
            chartsContainer.innerHTML = '<p>No charts available for this algorithm</p>';
        } else {
            console.log(`Successfully created ${chartsCreated} charts`);
        }
    }

    // Create loss history chart
    createLossChart(lossHistory, container) {
        const chartDiv = document.createElement('div');
        chartDiv.id = 'loss-chart';
        chartDiv.className = 'chart-container';
        container.appendChild(chartDiv);

        const trace = {
            x: Array.from({length: lossHistory.length}, (_, i) => i + 1),
            y: lossHistory,
            type: 'scatter',
            mode: 'lines',
            name: 'Training Loss',
            line: {color: '#1f77b4', width: 2}
        };

        const layout = {
            title: 'Training Loss History',
            xaxis: {title: 'Iteration'},
            yaxis: {title: 'Loss'},
            margin: {l: 50, r: 50, t: 50, b: 50}
        };

        window.Plotly.newPlot('loss-chart', [trace], layout, {responsive: true});
    }

    // Create predictions vs actual chart
    createPredictionsChart(predVsActual, container) {
        const chartDiv = document.createElement('div');
        chartDiv.id = 'predictions-chart';
        chartDiv.className = 'chart-container';
        container.appendChild(chartDiv);

        const trace = {
            x: predVsActual.actual,
            y: predVsActual.predicted,
            mode: 'markers',
            type: 'scatter',
            name: 'Predictions',
            marker: {color: '#ff7f0e', size: 6}
        };

        // Add perfect prediction line
        const minVal = Math.min(...predVsActual.actual);
        const maxVal = Math.max(...predVsActual.actual);
        const perfectLine = {
            x: [minVal, maxVal],
            y: [minVal, maxVal],
            mode: 'lines',
            type: 'scatter',
            name: 'Perfect Prediction',
            line: {color: 'red', dash: 'dash', width: 1}
        };

        const layout = {
            title: 'Predictions vs Actual Values',
            xaxis: {title: 'Actual Values'},
            yaxis: {title: 'Predicted Values'},
            margin: {l: 50, r: 50, t: 50, b: 50}
        };

        window.Plotly.newPlot('predictions-chart', [trace, perfectLine], layout, {responsive: true});
    }

    // Create feature importance chart
    createFeatureImportanceChart(featureImportance, container) {
        const chartDiv = document.createElement('div');
        chartDiv.id = 'feature-importance-chart';
        chartDiv.className = 'chart-container';
        container.appendChild(chartDiv);

        const trace = {
            x: featureImportance.values,
            y: featureImportance.features,
            type: 'bar',
            orientation: 'h',
            marker: {color: '#2ca02c'}
        };

        const layout = {
            title: 'Feature Importance',
            xaxis: {title: 'Importance'},
            yaxis: {title: 'Features'},
            margin: {l: 100, r: 50, t: 50, b: 50}
        };

        window.Plotly.newPlot('feature-importance-chart', [trace], layout, {responsive: true});
    }

    // Create residuals chart
    createResidualsChart(residuals, container) {
        const chartDiv = document.createElement('div');
        chartDiv.id = 'residuals-chart';
        chartDiv.className = 'chart-container';
        container.appendChild(chartDiv);

        const trace = {
            x: residuals.fitted,
            y: residuals.residuals,
            mode: 'markers',
            type: 'scatter',
            name: 'Residuals',
            marker: {color: '#d62728', size: 6}
        };

        // Add zero line
        const minFitted = Math.min(...residuals.fitted);
        const maxFitted = Math.max(...residuals.fitted);
        const zeroLine = {
            x: [minFitted, maxFitted],
            y: [0, 0],
            mode: 'lines',
            type: 'scatter',
            name: 'Zero Line',
            line: {color: 'black', dash: 'dash', width: 1}
        };

        const layout = {
            title: 'Residual Plot',
            xaxis: {title: 'Fitted Values'},
            yaxis: {title: 'Residuals'},
            margin: {l: 50, r: 50, t: 50, b: 50}
        };

        window.Plotly.newPlot('residuals-chart', [trace, zeroLine], layout, {responsive: true});
    }

    // Create loss curve chart from our API structure
    createLossCurveChart(lossCurve, container) {
        const chartDiv = document.createElement('div');
        chartDiv.id = 'loss-curve-chart';
        chartDiv.className = 'chart-container';
        chartDiv.style.cssText = 'width: 100%; height: 400px; margin-bottom: 20px; border: 1px solid #ddd; border-radius: 5px; padding: 10px;';
        container.appendChild(chartDiv);

        const trace = {
            x: Array.from({length: lossCurve.training_history.length}, (_, i) => i + 1),
            y: lossCurve.training_history,
            type: 'scatter',
            mode: 'lines',
            name: 'Training Loss',
            line: {color: '#1f77b4', width: 2}
        };

        const layout = {
            title: lossCurve.title || 'Training Loss Curve',
            xaxis: {title: lossCurve.x_label || 'Iteration'},
            yaxis: {title: lossCurve.y_label || 'Loss'},
            margin: {l: 60, r: 50, t: 60, b: 60},
            autosize: true
        };

        const config = {responsive: true, displayModeBar: false};
        window.Plotly.newPlot('loss-curve-chart', [trace], layout, config);
    }

    // Create residuals plot chart from our API structure
    createResidualsPlotChart(residualsPlot, container) {
        const chartDiv = document.createElement('div');
        chartDiv.id = 'residuals-plot-chart';
        chartDiv.className = 'chart-container';
        chartDiv.style.cssText = 'width: 100%; height: 400px; margin-bottom: 20px; border: 1px solid #ddd; border-radius: 5px; padding: 10px;';
        container.appendChild(chartDiv);

        // My implementation residuals
        const yourTrace = {
            x: residualsPlot.your_predictions || residualsPlot.predicted_values,
            y: residualsPlot.your_residuals || residualsPlot.residuals,
            mode: 'markers',
            type: 'scatter',
            name: 'My Implementation',
            marker: {color: '#1f77b4', size: 6}
        };

        const traces = [yourTrace];

        // Add sklearn residuals if available
        if (residualsPlot.sklearn_predictions && residualsPlot.sklearn_residuals) {
            const sklearnTrace = {
                x: residualsPlot.sklearn_predictions,
                y: residualsPlot.sklearn_residuals,
                mode: 'markers',
                type: 'scatter',
                name: 'Sklearn',
                marker: {color: '#ff7f0e', size: 6}
            };
            traces.push(sklearnTrace);
        }

        // Add zero line
        const allPredictions = residualsPlot.your_predictions || residualsPlot.predicted_values || [];
        if (allPredictions.length > 0) {
            const minPred = Math.min(...allPredictions);
            const maxPred = Math.max(...allPredictions);
            const zeroLine = {
                x: [minPred, maxPred],
                y: [0, 0],
                mode: 'lines',
                type: 'scatter',
                name: 'Zero Line',
                line: {color: 'black', dash: 'dash', width: 1},
                showlegend: false
            };
            traces.push(zeroLine);
        }

        const layout = {
            title: residualsPlot.title || 'Residual Analysis',
            xaxis: {title: residualsPlot.x_label || 'Predicted Values'},
            yaxis: {title: residualsPlot.y_label || 'Residuals'},
            margin: {l: 60, r: 50, t: 60, b: 60},
            autosize: true
        };

        const config = {responsive: true, displayModeBar: false};
        window.Plotly.newPlot('residuals-plot-chart', traces, layout, config);
    }

    // Create scatter plot chart from our API structure
    createScatterPlotChart(scatterPlot, container) {
        const chartDiv = document.createElement('div');
        chartDiv.id = 'scatter-plot-chart';
        chartDiv.className = 'chart-container';
        chartDiv.style.cssText = 'width: 100%; height: 400px; margin-bottom: 20px; border: 1px solid #ddd; border-radius: 5px; padding: 10px;';
        container.appendChild(chartDiv);

        // My implementation predictions vs actual
        const yourTrace = {
            x: scatterPlot.actual_values || scatterPlot.x_values,
            y: scatterPlot.your_predictions || scatterPlot.y_values,
            mode: 'markers',
            type: 'scatter',
            name: 'My Implementation',
            marker: {color: '#1f77b4', size: 6}
        };

        const traces = [yourTrace];

        // Add sklearn predictions if available
        if (scatterPlot.sklearn_predictions && scatterPlot.actual_values) {
            const sklearnTrace = {
                x: scatterPlot.actual_values,
                y: scatterPlot.sklearn_predictions,
                mode: 'markers',
                type: 'scatter',
                name: 'Sklearn',
                marker: {color: '#ff7f0e', size: 6}
            };
            traces.push(sklearnTrace);
        }

        // Add perfect prediction line
        const actualValues = scatterPlot.actual_values || scatterPlot.x_values || [];
        if (actualValues.length > 0) {
            const minVal = Math.min(...actualValues);
            const maxVal = Math.max(...actualValues);
            const perfectLine = {
                x: [minVal, maxVal],
                y: [minVal, maxVal],
                mode: 'lines',
                type: 'scatter',
                name: 'Perfect Prediction',
                line: {color: 'red', dash: 'dash', width: 1},
                showlegend: true
            };
            traces.push(perfectLine);
        }

        const layout = {
            title: scatterPlot.title || 'Predictions vs Actual Values',
            xaxis: {title: scatterPlot.x_label || 'Actual Values'},
            yaxis: {title: scatterPlot.y_label || 'Predicted Values'},
            margin: {l: 60, r: 50, t: 60, b: 60},
            autosize: true
        };

        const config = {responsive: true, displayModeBar: false};
        window.Plotly.newPlot('scatter-plot-chart', traces, layout, config);
    }

    // Display sklearn comparison
    displayComparison(result) {
        console.log('displayComparison called with:', result);
        const comparisonContainer = document.getElementById('comparison-display');
        
        // Extract sklearn comparison data from the API response structure
        const residualsData = result.charts?.residuals_plot;  // Changed from chart_data to charts
        const yourMetrics = result.your_implementation?.metrics;
        const sklearnMetrics = result.sklearn_implementation?.metrics;
        const comparison = result.comparison;
        
        console.log('Residuals data:', residualsData);
        console.log('Your metrics:', yourMetrics);
        console.log('Sklearn metrics:', sklearnMetrics);
        console.log('Comparison data:', comparison);
        
        if (!yourMetrics || !sklearnMetrics) {
            console.log('Missing metrics data for sklearn comparison. Your metrics:', !!yourMetrics, 'Sklearn metrics:', !!sklearnMetrics);
            comparisonContainer.innerHTML = '<p>Sklearn comparison not available - missing metrics data</p>';
            return;
        }

        try {
            const comparisonHTML = `
                <div class="comparison-metrics">
                    <div class="metric-comparison">
                        <div class="metric-name">R² Score</div>
                        <div class="metric-values">
                            <div class="custom-value">
                                <span class="label">My Implementation:</span>
                                <span class="value">${yourMetrics.r2_score?.toFixed(4) || 'N/A'}</span>
                            </div>
                            <div class="sklearn-value">
                                <span class="label">Sklearn:</span>
                                <span class="value">${sklearnMetrics.r2_score?.toFixed(4) || 'N/A'}</span>
                            </div>
                            <div class="difference">
                                <span class="label">Difference:</span>
                                <span class="value">${((yourMetrics.r2_score || 0) - (sklearnMetrics.r2_score || 0)).toFixed(4)}</span>
                            </div>
                        </div>
                    </div>
                    <div class="metric-comparison">
                        <div class="metric-name">Mean Squared Error</div>
                        <div class="metric-values">
                            <div class="custom-value">
                                <span class="label">My Implementation:</span>
                                <span class="value">${yourMetrics.mse?.toFixed(4) || 'N/A'}</span>
                            </div>
                            <div class="sklearn-value">
                                <span class="label">Sklearn:</span>
                                <span class="value">${sklearnMetrics.mse?.toFixed(4) || 'N/A'}</span>
                            </div>
                            <div class="difference">
                                <span class="label">Difference:</span>
                                <span class="value">${((yourMetrics.mse || 0) - (sklearnMetrics.mse || 0)).toFixed(4)}</span>
                            </div>
                        </div>
                    </div>
                    <div class="metric-comparison">
                        <div class="metric-name">Mean Absolute Error</div>
                        <div class="metric-values">
                            <div class="custom-value">
                                <span class="label">My Implementation:</span>
                                <span class="value">${yourMetrics.mae?.toFixed(4) || 'N/A'}</span>
                            </div>
                            <div class="sklearn-value">
                                <span class="label">Sklearn:</span>
                                <span class="value">${sklearnMetrics.mae?.toFixed(4) || 'N/A'}</span>
                            </div>
                            <div class="difference">
                                <span class="label">Difference:</span>
                                <span class="value">${((yourMetrics.mae || 0) - (sklearnMetrics.mae || 0)).toFixed(4)}</span>
                            </div>
                        </div>
                    </div>
                </div>
            `;

            comparisonContainer.innerHTML = comparisonHTML;
        } catch (error) {
            console.error('Error creating sklearn comparison:', error);
            comparisonContainer.innerHTML = '<p>Error displaying sklearn comparison</p>';
        }
    }

    // Display training details
    displayDetails(result) {
        console.log('displayDetails called with:', result);
        const detailsContainer = document.getElementById('details-display');
        
        const metrics = result.your_implementation?.metrics;
        
        const detailsHTML = `
            <div class="details-content">
                <div class="training-info">
                    <h5>Training Information</h5>
                    <p><strong>Algorithm:</strong> ${result.algorithm_id || 'Unknown'}</p>
                    <p><strong>Dataset:</strong> ${result.dataset?.name || 'Unknown'}</p>
                    <p><strong>Training Time:</strong> ${metrics?.training_time?.toFixed(3) || 'N/A'}s</p>
                    <p><strong>Iterations:</strong> ${metrics?.convergence_iterations || 'N/A'}</p>
                </div>
                <div class="hyperparameters">
                    <h5>Hyperparameters</h5>
                    <pre>${JSON.stringify(result.hyperparameters, null, 2)}</pre>
                </div>
                <div class="dataset-info">
                    <h5>Dataset Information</h5>
                    ${result.dataset ? `
                        <p><strong>Samples:</strong> ${result.dataset.n_samples}</p>
                        <p><strong>Features:</strong> ${result.dataset.n_features}</p>
                        <p><strong>Train Size:</strong> ${result.dataset.train_size}</p>
                        <p><strong>Test Size:</strong> ${result.dataset.test_size}</p>
                        <p><strong>Features:</strong> ${result.dataset.feature_names?.join(', ') || 'N/A'}</p>
                    ` : '<p>No dataset information available</p>'}
                </div>
            </div>
        `;

        detailsContainer.innerHTML = detailsHTML;
    }

    // Display error message
    displayError(error) {
        this.container.innerHTML = `
            <div class="error-container">
                <h3>Training Failed</h3>
                <p class="error-message">${error}</p>
                <button onclick="location.reload()" class="retry-button">Try Again</button>
            </div>
        `;
    }

    // Format metric names for display
    formatMetricName(name) {
        return name
            .replace(/_/g, ' ')
            .replace(/\b\w/g, l => l.toUpperCase());
    }
}

// Enhanced dataset selector component
class DatasetSelector {
    constructor(containerId, apiService) {
        this.container = document.getElementById(containerId);
        this.apiService = apiService;
        this.selectedDataset = 'diabetes';
        this.datasets = [];
        this.initialize();
    }

    async initialize() {
        try {
            const response = await this.apiService.getAvailableDatasets();
            this.datasets = response.datasets || [];
            this.render();
        } catch (error) {
            console.error('Failed to load datasets:', error);
            this.renderError();
        }
    }

    render() {
        const selectorHTML = `
            <div class="dataset-selector">
                <label for="dataset-select">Select Dataset:</label>
                <select id="dataset-select" class="dataset-dropdown">
                    ${this.datasets.map(dataset => `
                        <option value="${dataset.id}" ${dataset.id === this.selectedDataset ? 'selected' : ''}>
                            ${dataset.name} (${dataset.type}, ${dataset.samples} samples)
                        </option>
                    `).join('')}
                </select>
                <div class="dataset-info">
                    <div id="dataset-description"></div>
                </div>
            </div>
        `;

        this.container.innerHTML = selectorHTML;

        // Add event listener
        const selectElement = document.getElementById('dataset-select');
        selectElement.addEventListener('change', (e) => {
            this.selectedDataset = e.target.value;
            this.updateDatasetInfo();
        });

        this.updateDatasetInfo();
    }

    renderError() {
        this.container.innerHTML = `
            <div class="dataset-selector-error">
                <p>Failed to load datasets. Using default dataset.</p>
            </div>
        `;
    }

    updateDatasetInfo() {
        const selectedDataset = this.datasets.find(d => d.id === this.selectedDataset);
        const infoContainer = document.getElementById('dataset-description');
        
        if (selectedDataset && infoContainer) {
            infoContainer.innerHTML = `
                <p><strong>Description:</strong> ${selectedDataset.description}</p>
                <p><strong>Type:</strong> ${selectedDataset.type}</p>
                <p><strong>Features:</strong> ${selectedDataset.features}</p>
            `;
        }
    }

    getSelectedDataset() {
        return this.selectedDataset;
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { EnhancedResultsDisplay, DatasetSelector };
}

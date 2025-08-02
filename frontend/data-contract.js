/**
 * Data Contract Validation and Normalization System
 * Ensures consistent data flow between backend and frontend
 */

class DataContractValidator {
    constructor() {
        // Define expected data structures for frontend
        this.expectedSchemas = {
            // Main training result structure
            trainingResult: {
                success: 'boolean',
                algorithm_id: 'string',
                algorithm_type: 'string',
                dataset: 'object',
                hyperparameters: 'object',
                your_implementation: 'object',
                sklearn_implementation: 'object',
                comparison: 'object',
                charts: 'object',
                total_duration: 'number',
                timestamp: 'string'
            },

            // Chart data structures expected by frontend
            charts: {
                loss_curve: {
                    iterations: 'array',
                    loss: 'array',
                    x_label: 'string',
                    y_label: 'string',
                    title: 'string'
                },
                scatter_plot: {
                    actual: 'array',
                    your_predictions: 'array',
                    sklearn_predictions: 'array',
                    x_label: 'string',
                    y_label: 'string',
                    title: 'string'
                },
                residuals_plot: {
                    your_predictions: 'array',
                    your_residuals: 'array',
                    sklearn_predictions: 'array',
                    sklearn_residuals: 'array',
                    x_label: 'string',
                    y_label: 'string',
                    title: 'string'
                },
                confusion_matrix: {
                    your_matrix: 'array',
                    sklearn_matrix: 'array', 
                    labels: 'array'
                },
                decision_boundary: {
                    x_data: 'array',
                    y_data: 'array',
                    labels: 'array',
                    x_label: 'string',
                    y_label: 'string',
                    title: 'string'
                },
                coefficients_data: {
                    coefficients: 'array',
                    intercept: 'number'
                }
            }
        };

        // Define data normalization rules
        this.normalizationRules = {
            // Handle different field name variations
            fieldMappings: {
                'actual_values': 'actual',
                'training_history': 'loss',
                'loss_history': 'loss',
                'predictions': 'your_predictions'
            },

            // Data type conversions
            typeConversions: {
                numpy_array_to_js: (data) => Array.isArray(data) ? data : (data?.tolist ? data.tolist() : []),
                string_to_number: (data) => typeof data === 'string' ? parseFloat(data) : data
            }
        };
    }

    /**
     * Validates incoming backend data against expected frontend schema
     */
    validateTrainingResult(data) {
        console.log('🔍 Validating training result data contract...');
        const errors = [];
        const warnings = [];

        try {
            // Check main structure
            this._validateObject(data, this.expectedSchemas.trainingResult, 'trainingResult', errors, warnings);
            
            // Check charts structure if present
            if (data.charts) {
                this._validateCharts(data.charts, errors, warnings);
            }

            // Report validation results
            if (errors.length > 0) {
                console.error('❌ Data contract validation failed:', errors);
                throw new Error(`Data contract validation failed: ${errors.join(', ')}`);
            }

            if (warnings.length > 0) {
                console.warn('⚠️ Data contract warnings:', warnings);
            }

            console.log('✅ Data contract validation passed');
            return true;

        } catch (error) {
            console.error('❌ Data contract validation error:', error);
            throw error;
        }
    }

    /**
     * Normalizes backend data to match frontend expectations
     */
    normalizeTrainingResult(data) {
        console.log('🔄 Normalizing training result data...');
        
        try {
            // Deep clone to avoid mutating original
            const normalized = JSON.parse(JSON.stringify(data));

            // Normalize main structure
            this._normalizeObject(normalized);

            // Normalize charts data
            if (normalized.charts) {
                this._normalizeCharts(normalized.charts);
            }

            // Ensure required fields exist with defaults
            this._ensureDefaults(normalized);

            console.log('✅ Data normalization completed');
            return normalized;

        } catch (error) {
            console.error('❌ Data normalization failed:', error);
            throw error;
        }
    }

    /**
     * Validates and normalizes data in one step
     */
    processTrainingResult(data) {
        console.log('🔄 Processing training result with data contract validation...');
        
        try {
            // First normalize the data
            const normalized = this.normalizeTrainingResult(data);
            
            // Then validate the normalized data
            this.validateTrainingResult(normalized);
            
            return normalized;

        } catch (error) {
            console.error('❌ Data contract processing failed:', error);
            
            // Return a safe fallback structure
            return this._createFallbackResult(data, error.message);
        }
    }

    // Private validation methods
    _validateObject(obj, schema, path, errors, warnings) {
        for (const [key, expectedType] of Object.entries(schema)) {
            const fullPath = `${path}.${key}`;
            
            if (obj[key] === undefined || obj[key] === null) {
                warnings.push(`Missing field: ${fullPath}`);
                continue;
            }

            const actualType = this._getType(obj[key]);
            if (actualType !== expectedType) {
                if (this._isCoercible(obj[key], expectedType)) {
                    warnings.push(`Type mismatch (coercible): ${fullPath} expected ${expectedType}, got ${actualType}`);
                } else {
                    errors.push(`Type mismatch: ${fullPath} expected ${expectedType}, got ${actualType}`);
                }
            }
        }
    }

    _validateCharts(charts, errors, warnings) {
        const chartSchemas = this.expectedSchemas.charts;
        
        for (const [chartType, chartData] of Object.entries(charts)) {
            if (chartData === null || chartData === undefined) {
                continue; // Skip null charts
            }

            const schema = chartSchemas[chartType];
            if (schema) {
                this._validateObject(chartData, schema, `charts.${chartType}`, errors, warnings);
            } else {
                warnings.push(`Unknown chart type: ${chartType}`);
            }
        }
    }

    // Private normalization methods
    _normalizeObject(obj) {
        // Apply field mappings
        for (const [oldField, newField] of Object.entries(this.normalizationRules.fieldMappings)) {
            if (obj[oldField] !== undefined && obj[newField] === undefined) {
                obj[newField] = obj[oldField];
                console.log(`🔄 Mapped field: ${oldField} → ${newField}`);
            }
        }

        // Convert numpy arrays to JavaScript arrays
        for (const [key, value] of Object.entries(obj)) {
            if (value && typeof value === 'object' && value.tolist) {
                obj[key] = value.tolist();
                console.log(`🔄 Converted numpy array: ${key}`);
            }
        }
    }

    _normalizeCharts(charts) {
        // Normalize each chart's data structure
        for (const [chartType, chartData] of Object.entries(charts)) {
            if (chartData === null || chartData === undefined) {
                continue;
            }

            // Apply field mappings for chart data
            this._normalizeObject(chartData);

            // Special handling for specific chart types
            switch (chartType) {
                case 'loss_curve':
                    this._normalizeLossCurve(chartData);
                    break;
                case 'scatter_plot':
                    this._normalizeScatterPlot(chartData);
                    break;
                case 'residuals_plot':
                    this._normalizeResidualsPlot(chartData);
                    break;
            }
        }
    }

    _normalizeLossCurve(chartData) {
        // Ensure iterations array exists
        if (!chartData.iterations && chartData.loss) {
            chartData.iterations = Array.from({length: chartData.loss.length}, (_, i) => i + 1);
            console.log('🔄 Generated iterations array for loss curve');
        }

        // Handle training_history field
        if (chartData.training_history && !chartData.loss) {
            chartData.loss = chartData.training_history;
            chartData.iterations = Array.from({length: chartData.loss.length}, (_, i) => i + 1);
            console.log('🔄 Converted training_history to loss/iterations');
        }

        // Ensure labels exist
        chartData.x_label = chartData.x_label || 'Iteration';
        chartData.y_label = chartData.y_label || 'Loss';
        chartData.title = chartData.title || 'Training Loss Curve';
    }

    _normalizeScatterPlot(chartData) {
        // Handle actual_values field
        if (chartData.actual_values && !chartData.actual) {
            chartData.actual = chartData.actual_values;
            console.log('🔄 Mapped actual_values to actual');
        }

        // Ensure labels exist
        chartData.x_label = chartData.x_label || 'Actual Values';
        chartData.y_label = chartData.y_label || 'Predicted Values';
        chartData.title = chartData.title || 'Predictions vs Actual';
    }

    _normalizeResidualsPlot(chartData) {
        // Ensure labels exist
        chartData.x_label = chartData.x_label || 'Predicted Values';
        chartData.y_label = chartData.y_label || 'Residuals';
        chartData.title = chartData.title || 'Residual Analysis';
    }

    _ensureDefaults(data) {
        // Ensure basic structure exists
        data.success = data.success !== undefined ? data.success : false;
        data.algorithm_id = data.algorithm_id || 'unknown';
        data.algorithm_type = data.algorithm_type || 'unknown';
        data.timestamp = data.timestamp || new Date().toISOString();
        data.total_duration = data.total_duration || 0;

        // Ensure dataset info exists
        if (!data.dataset) {
            data.dataset = {
                name: 'Unknown Dataset',
                n_samples: 0,
                n_features: 0
            };
        }

        // Ensure charts object exists
        if (!data.charts) {
            data.charts = {};
        }
    }

    _createFallbackResult(originalData, errorMessage) {
        console.warn('🔄 Creating fallback result due to data contract failure');
        
        return {
            success: false,
            algorithm_id: originalData?.algorithm_id || 'unknown',
            algorithm_type: originalData?.algorithm_type || 'unknown',
            dataset: originalData?.dataset || { name: 'Unknown', n_samples: 0, n_features: 0 },
            hyperparameters: originalData?.hyperparameters || {},
            charts: {},
            error: `Data contract validation failed: ${errorMessage}`,
            timestamp: new Date().toISOString(),
            total_duration: 0,
            your_implementation: { metrics: {} },
            sklearn_implementation: { metrics: {} },
            comparison: {}
        };
    }

    // Utility methods
    _getType(value) {
        if (Array.isArray(value)) return 'array';
        if (value === null) return 'null';
        return typeof value;
    }

    _isCoercible(value, expectedType) {
        // Check if value can be converted to expected type
        switch (expectedType) {
            case 'number':
                return !isNaN(Number(value));
            case 'string':
                return true; // Almost anything can be converted to string
            case 'array':
                return value && typeof value === 'object' && value.tolist;
            default:
                return false;
        }
    }

    /**
     * Generate a data contract report for debugging
     */
    generateReport(data) {
        console.log('📊 Generating data contract report...');
        
        const report = {
            timestamp: new Date().toISOString(),
            dataStructure: this._analyzeStructure(data),
            chartTypes: data.charts ? Object.keys(data.charts).filter(k => data.charts[k] !== null) : [],
            validation: null,
            recommendations: []
        };

        try {
            this.validateTrainingResult(data);
            report.validation = 'PASSED';
        } catch (error) {
            report.validation = 'FAILED';
            report.error = error.message;
        }

        // Generate recommendations
        if (data.charts) {
            for (const [chartType, chartData] of Object.entries(data.charts)) {
                if (chartData === null) {
                    report.recommendations.push(`Consider implementing ${chartType} chart for better insights`);
                }
            }
        }

        console.log('📋 Data Contract Report:', report);
        return report;
    }

    _analyzeStructure(obj, path = '', depth = 0) {
        if (depth > 3) return '[...truncated]';
        
        const structure = {};
        for (const [key, value] of Object.entries(obj || {})) {
            const currentPath = path ? `${path}.${key}` : key;
            
            if (value === null || value === undefined) {
                structure[key] = 'null';
            } else if (Array.isArray(value)) {
                structure[key] = `array[${value.length}]`;
            } else if (typeof value === 'object') {
                structure[key] = this._analyzeStructure(value, currentPath, depth + 1);
            } else {
                structure[key] = typeof value;
            }
        }
        return structure;
    }
}

// Export for use throughout the application
window.DataContractValidator = DataContractValidator;

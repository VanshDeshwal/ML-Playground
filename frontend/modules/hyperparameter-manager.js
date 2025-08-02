// Hyperparameter controls management
class HyperparameterManager {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        if (!this.container) {
            throw new Error(`Hyperparameter container '${containerId}' not found`);
        }
    }

    generateControls(hyperparameters) {
        if (Object.keys(hyperparameters).length === 0) {
            this.container.innerHTML = `
                <div class="no-hyperparameters">
                    <i class="fas fa-info-circle"></i>
                    <p>This algorithm uses default parameters with no adjustable hyperparameters.</p>
                </div>
            `;
            return;
        }

        const controlsHTML = Object.entries(hyperparameters)
            .map(([key, param]) => this.createControl(key, param))
            .join('');

        this.container.innerHTML = controlsHTML;
    }

    createControl(key, param) {
        const { type, name, default: defaultValue, min, max } = param;
        const displayName = name || key;
        
        switch (type) {
            case 'float':
            case 'int':
                return this.createNumericControl(key, displayName, type, defaultValue, min, max);
            case 'boolean':
                return this.createBooleanControl(key, displayName, defaultValue);
            case 'select':
                return this.createSelectControl(key, displayName, param.options, defaultValue);
            default:
                return '';
        }
    }

    createNumericControl(key, displayName, type, defaultValue, min, max) {
        const step = type === 'float' ? '0.001' : '1';
        const minVal = min !== undefined ? min : (type === 'float' ? 0.0 : 1);
        const maxVal = max !== undefined ? max : (type === 'float' ? 10.0 : 100);
        const currentValue = defaultValue !== undefined ? defaultValue : minVal;
        
        return `
            <div class="control-group">
                <label for="${key}" class="control-label">
                    <strong>${displayName}</strong>
                </label>
                <div class="control-input slider-control">
                    <input 
                        type="range" 
                        id="${key}" 
                        name="${key}" 
                        value="${currentValue}" 
                        min="${minVal}" 
                        max="${maxVal}" 
                        step="${step}"
                        class="slider-input"
                        oninput="this.nextElementSibling.value = this.value"
                    >
                    <input 
                        type="number" 
                        value="${currentValue}" 
                        min="${minVal}" 
                        max="${maxVal}" 
                        step="${step}"
                        class="number-display"
                        oninput="this.previousElementSibling.value = this.value"
                    >
                    <span class="control-range">(${minVal} - ${maxVal})</span>
                </div>
            </div>
        `;
    }

    createBooleanControl(key, displayName, defaultValue) {
        return `
            <div class="control-group">
                <label for="${key}" class="control-label">
                    <strong>${displayName}</strong>
                </label>
                <div class="control-input">
                    <input 
                        type="checkbox" 
                        id="${key}" 
                        name="${key}" 
                        ${defaultValue ? 'checked' : ''}
                        class="checkbox-input"
                    >
                </div>
            </div>
        `;
    }

    createSelectControl(key, displayName, options, defaultValue) {
        const optionsHTML = options.map(option => 
            `<option value="${option}" ${option === defaultValue ? 'selected' : ''}>${option}</option>`
        ).join('');
        
        return `
            <div class="control-group">
                <label for="${key}" class="control-label">
                    <strong>${displayName}</strong>
                </label>
                <div class="control-input">
                    <select id="${key}" name="${key}" class="select-input">
                        ${optionsHTML}
                    </select>
                </div>
            </div>
        `;
    }

    getValues() {
        const hyperparameters = {};
        
        // Handle slider controls (range inputs)
        const sliders = this.container.querySelectorAll('input[type="range"]');
        sliders.forEach(slider => {
            const value = slider.step === '1' ? 
                parseInt(slider.value) : 
                parseFloat(slider.value);
            hyperparameters[slider.name] = value;
        });
        
        // Handle other controls (checkboxes, selects, number inputs not paired with sliders)
        const otherControls = this.container.querySelectorAll('input[type="checkbox"], select, input[type="number"]:not(.number-display)');
        otherControls.forEach(control => {
            if (control.type === 'checkbox') {
                hyperparameters[control.name] = control.checked;
            } else if (control.type === 'number') {
                const value = control.step === '1' ? 
                    parseInt(control.value) : 
                    parseFloat(control.value);
                hyperparameters[control.name] = value;
            } else {
                hyperparameters[control.name] = control.value;
            }
        });
        
        return hyperparameters;
    }
}

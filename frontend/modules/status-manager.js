// Status manager for backend connection monitoring
class StatusManager {
    constructor() {
        this.statusButton = document.getElementById('status-button');
        this.statusDot = this.statusButton?.querySelector('.status-dot');
        this.statusText = this.statusButton?.querySelector('.status-text');
        
        this.setupEventListeners();
    }

    setupEventListeners() {
        if (this.statusButton) {
            this.statusButton.onclick = () => {
                const apiBaseUrl = window.apiService?.baseUrl || 'https://api.playground.vanshdeshwal.dev';
                window.open(`${apiBaseUrl}/docs`, '_blank');
            };
        }
    }

    async checkConnection() {
        if (!this.statusDot || !this.statusText) return;

        try {
            const isConnected = await window.apiService.checkBackendStatus();
            
            if (isConnected) {
                this.statusDot.className = 'status-dot connected';
                this.statusText.textContent = 'Backend Connected';
            } else {
                this.statusDot.className = 'status-dot disconnected';
                this.statusText.textContent = 'Backend Disconnected';
            }
        } catch (error) {
            this.statusDot.className = 'status-dot status-error';
            this.statusText.textContent = 'Connection Failed';
        }
    }
}

// Deployment notice for production environment
class DeploymentNotice {
    static showBackendNotice() {
        if (window.APP_CONFIG.ENVIRONMENT === 'production') {
            const notice = document.createElement('div');
            notice.id = 'backend-notice';
            notice.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                background: #f59e0b;
                color: white;
                padding: 10px;
                text-align: center;
                z-index: 1000;
                font-weight: bold;
            `;
            notice.innerHTML = `
                ⚠️ Backend Required: This demo requires a deployed backend API. 
                <a href="https://github.com/VanshDeshwal/ML-Playground" style="color: white; text-decoration: underline;">
                    View deployment instructions
                </a>
            `;
            document.body.prepend(notice);
            
            // Auto-hide after 10 seconds
            setTimeout(() => {
                if (notice && notice.parentNode) {
                    notice.remove();
                }
            }, 10000);
        }
    }
    
    static async checkAndNotify() {
        // Check if backend is accessible
        try {
            const response = await fetch(window.APP_CONFIG.API_BASE_URL + '/', {
                method: 'GET',
                mode: 'cors'
            });
            if (!response.ok) {
                this.showBackendNotice();
            }
        } catch (error) {
            console.warn('Backend not accessible:', error);
            this.showBackendNotice();
        }
    }
}

// Auto-check on page load for production
if (window.APP_CONFIG && window.APP_CONFIG.ENVIRONMENT === 'production') {
    document.addEventListener('DOMContentLoaded', () => {
        DeploymentNotice.checkAndNotify();
    });
}

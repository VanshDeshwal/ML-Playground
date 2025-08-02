// Code modal management
class CodeModalManager {
    constructor() {
        this.modal = document.getElementById('code-modal');
        this.modalTitle = document.getElementById('code-modal-title');
        this.codeFilename = document.getElementById('code-filename');
        this.codeLanguage = document.getElementById('code-language');
        this.codeDescription = document.getElementById('code-description');
        this.codeContent = document.getElementById('code-content');
        
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Modal close events
        const closeBtn = document.getElementById('code-modal-close');
        const backdrop = document.getElementById('code-modal-backdrop');
        const copyBtn = document.getElementById('copy-code-btn');
        
        if (closeBtn) {
            closeBtn.addEventListener('click', () => this.hide());
        }
        
        if (backdrop) {
            backdrop.addEventListener('click', () => this.hide());
        }
        
        if (copyBtn) {
            copyBtn.addEventListener('click', () => this.copyCode());
        }
        
        // Close modal on Escape key
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this.modal && !this.modal.classList.contains('hidden')) {
                this.hide();
            }
        });
    }

    async show(algorithmId) {
        if (!this.modal) return;

        try {
            // Show loading state
            this.modalTitle.textContent = 'Loading Code...';
            this.codeFilename.textContent = 'loading...';
            this.codeLanguage.textContent = 'Python';
            this.codeDescription.textContent = 'Fetching implementation details...';
            this.codeContent.textContent = 'Loading code snippet...';
            
            // Show modal
            this.modal.classList.remove('hidden');
            
            // Fetch code snippet from API
            const response = await fetch(`${window.APP_CONFIG.API_BASE_URL}/algorithms/${algorithmId}/code`);
            
            if (!response.ok) {
                throw new Error(`Failed to fetch code: ${response.status}`);
            }
            
            const codeData = await response.json();
            
            // Update modal content
            this.modalTitle.textContent = codeData.description;
            this.codeFilename.textContent = codeData.filename;
            this.codeLanguage.textContent = codeData.language.toUpperCase();
            this.codeDescription.textContent = codeData.description;
            
            // Set the code content and apply syntax highlighting
            this.codeContent.textContent = codeData.code;
            
            // Determine the Prism language class
            const languageClass = this.getPrismLanguageClass(codeData.language, codeData.filename);
            this.codeContent.className = `language-${languageClass}`;
            
            // Apply Prism syntax highlighting
            if (window.Prism) {
                window.Prism.highlightElement(this.codeContent);
            }
            
        } catch (error) {
            console.error('Failed to load code:', error);
            this.showError(error.message);
        }
    }

    showError(message) {
        this.modalTitle.textContent = 'Error Loading Code';
        this.codeFilename.textContent = 'error';
        this.codeDescription.textContent = 'Failed to load the implementation code. Please try again.';
        this.codeContent.textContent = `Error: ${message}`;
        this.codeContent.className = 'language-python';
    }

    hide() {
        if (this.modal) {
            this.modal.classList.add('hidden');
        }
    }

    copyCode() {
        if (!this.codeContent) return;

        const text = this.codeContent.textContent;
        
        navigator.clipboard.writeText(text).then(() => {
            const copyBtn = document.getElementById('copy-code-btn');
            if (!copyBtn) return;

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

    // Helper function to map languages/file extensions to Prism language classes
    getPrismLanguageClass(language, filename) {
        const languageMap = {
            'python': 'python',
            'javascript': 'javascript',
            'typescript': 'typescript',
            'java': 'java',
            'cpp': 'cpp',
            'c++': 'cpp',
            'c': 'c',
            'r': 'r',
            'sql': 'sql',
            'json': 'json',
            'yaml': 'yaml',
            'yml': 'yaml',
            'xml': 'xml',
            'html': 'html',
            'css': 'css',
            'markdown': 'markdown',
            'bash': 'bash',
            'shell': 'bash'
        };

        // Check language first
        if (language && languageMap[language.toLowerCase()]) {
            return languageMap[language.toLowerCase()];
        }

        // Fall back to file extension
        if (filename) {
            const extensionMap = {
                '.py': 'python',
                '.js': 'javascript',
                '.ts': 'typescript',
                '.java': 'java',
                '.cpp': 'cpp',
                '.c': 'c',
                '.r': 'r',
                '.sql': 'sql',
                '.json': 'json',
                '.yaml': 'yaml',
                '.yml': 'yaml',
                '.xml': 'xml',
                '.html': 'html',
                '.css': 'css',
                '.md': 'markdown',
                '.sh': 'bash'
            };

            const extension = filename.toLowerCase().match(/\.[^.]+$/);
            if (extension && extensionMap[extension[0]]) {
                return extensionMap[extension[0]];
            }
        }

        // Default to python for ML playground
        return 'python';
    }
}

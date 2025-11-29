// RAG Web App JavaScript
class RAGApp {
    constructor() {
        this.files = [];
        this.indexed = false;
        this.results = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.updateStatus();
    }

    setupEventListeners() {
        // File upload events
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');

        uploadArea.addEventListener('click', () => fileInput.click());

        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            this.handleFiles(e.dataTransfer.files);
        });

        fileInput.addEventListener('change', (e) => {
            this.handleFiles(e.target.files);
        });

        // Button events
        document.getElementById('uploadBtn').addEventListener('click', () => this.uploadFiles());
        document.getElementById('buildIndexBtn').addEventListener('click', () => this.buildIndex());
        document.getElementById('clearBtn').addEventListener('click', () => this.clearAll());
        document.getElementById('submitBtn').addEventListener('click', () => this.submitQuery());

        // Query input
        const queryInput = document.getElementById('queryInput');
        queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !document.getElementById('submitBtn').disabled) {
                this.submitQuery();
            }
        });

        // Auto-refresh status
        setInterval(() => this.updateStatus(), 5000);
    }

    handleFiles(fileList) {
        this.files = Array.from(fileList);
        this.displayFileList();
    }

    displayFileList() {
        const fileListDiv = document.getElementById('fileList');
        fileListDiv.innerHTML = '';

        this.files.forEach((file, index) => {
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            fileItem.innerHTML = `
                <span class="file-item-name">📄 ${file.name}</span>
                <span class="file-item-size">${this.formatFileSize(file.size)}</span>
                <button class="file-item-remove" onclick="app.removeFile(${index})">&times;</button>
            `;
            fileListDiv.appendChild(fileItem);
        });
    }

    removeFile(index) {
        this.files.splice(index, 1);
        this.displayFileList();
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
    }

    async uploadFiles() {
        if (this.files.length === 0) {
            this.showMessage('No files selected', 'error');
            return;
        }

        const uploadBtn = document.getElementById('uploadBtn');
        const progressContainer = document.getElementById('uploadProgress');
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');

        uploadBtn.disabled = true;
        progressContainer.style.display = 'block';

        const formData = new FormData();
        this.files.forEach(file => {
            formData.append('files', file);
        });

        try {
            // Simulate progress
            let progress = 0;
            const progressInterval = setInterval(() => {
                if (progress < 90) {
                    progress += Math.random() * 30;
                    this.updateProgress(progress, progressBar, progressText);
                }
            }, 200);

            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            clearInterval(progressInterval);
            progress = 100;
            this.updateProgress(progress, progressBar, progressText);

            const data = await response.json();

            if (data.success) {
                this.showMessage(`✅ ${data.message}`, 'success');
                this.files = [];
                this.displayFileList();
                document.getElementById('fileInput').value = '';
            } else {
                this.showMessage(`❌ ${data.message}`, 'error');
            }

            if (data.errors && data.errors.length > 0) {
                data.errors.forEach(err => this.showMessage(`⚠️ ${err}`, 'error'));
            }
        } catch (error) {
            this.showMessage(`Error uploading files: ${error.message}`, 'error');
        } finally {
            uploadBtn.disabled = false;
            setTimeout(() => {
                progressContainer.style.display = 'none';
                progressBar.style.width = '0%';
                progressText.textContent = '0%';
            }, 500);
        }
    }

    updateProgress(progress, bar, text) {
        bar.style.width = progress + '%';
        text.textContent = Math.round(progress) + '%';
    }

    async buildIndex() {
        const buildBtn = document.getElementById('buildIndexBtn');
        buildBtn.disabled = true;
        const originalText = buildBtn.innerHTML;
        buildBtn.innerHTML = '<span class="spinner"></span> Building...';

        try {
            const response = await fetch('/build-index', {
                method: 'POST'
            });

            const data = await response.json();

            if (data.success) {
                this.showMessage(`✅ ${data.message}`, 'success');
                this.indexed = true;
                this.enableQueryInput();
                this.updateStatus();
            } else {
                this.showMessage(`❌ ${data.message}`, 'error');
            }
        } catch (error) {
            this.showMessage(`Error building index: ${error.message}`, 'error');
        } finally {
            buildBtn.disabled = false;
            buildBtn.innerHTML = originalText;
        }
    }

    async submitQuery() {
        const queryInput = document.getElementById('queryInput');
        const query = queryInput.value.trim();

        if (!query) {
            this.showMessage('Query cannot be empty', 'error');
            return;
        }

        const submitBtn = document.getElementById('submitBtn');
        const originalText = submitBtn.innerHTML;
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span class="spinner"></span> Searching...';

        try {
            const response = await fetch('/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query })
            });

            const data = await response.json();

            if (data.success) {
                this.addResult(data.query, data.answer);
                queryInput.value = '';
            } else {
                this.showMessage(`❌ ${data.message}`, 'error');
            }
        } catch (error) {
            this.showMessage(`Error querying: ${error.message}`, 'error');
        } finally {
            submitBtn.disabled = false;
            submitBtn.innerHTML = originalText;
        }
    }

    addResult(query, answer) {
        const resultsList = document.getElementById('resultsList');
        const timestamp = new Date().toLocaleTimeString();

        const resultItem = document.createElement('div');
        resultItem.className = 'result-item';
        resultItem.innerHTML = `
            <div class="result-question">Q: ${this.escapeHtml(query)}</div>
            <div class="result-answer">${this.escapeHtml(answer)}</div>
            <div class="result-timestamp">${timestamp}</div>
        `;

        resultsList.insertBefore(resultItem, resultsList.firstChild);

        // Keep only last 10 results
        while (resultsList.children.length > 10) {
            resultsList.removeChild(resultsList.lastChild);
        }
    }

    enableQueryInput() {
        document.getElementById('queryInput').disabled = false;
        document.getElementById('submitBtn').disabled = false;
    }

    disableQueryInput() {
        document.getElementById('queryInput').disabled = true;
        document.getElementById('submitBtn').disabled = true;
    }

    async updateStatus() {
        try {
            const response = await fetch('/status');
            const data = await response.json();

            const statusValue = document.getElementById('statusValue');
            const statusDetails = document.getElementById('statusDetails');

            if (data.indexed) {
                statusValue.textContent = '✅ Indexed';
                statusValue.style.color = 'var(--secondary-color)';
                this.indexed = true;
                this.enableQueryInput();
            } else {
                statusValue.textContent = '⏳ Not Indexed';
                statusValue.style.color = 'var(--warning-color)';
                this.indexed = false;
                this.disableQueryInput();
            }

            statusDetails.innerHTML = `
                <div class="status-item">
                    <span class="status-item-label">Documents</span>
                    <span class="status-item-value">${data.num_documents}</span>
                </div>
                <div class="status-item">
                    <span class="status-item-label">Chunks</span>
                    <span class="status-item-value">${data.num_chunks}</span>
                </div>
            `;
        } catch (error) {
            console.error('Error updating status:', error);
        }
    }

    async clearAll() {
        if (!confirm('Are you sure you want to clear all documents and index? This cannot be undone.')) {
            return;
        }

        const clearBtn = document.getElementById('clearBtn');
        clearBtn.disabled = true;

        try {
            const response = await fetch('/clear-index', {
                method: 'POST'
            });

            const data = await response.json();

            if (data.success) {
                this.showMessage(`✅ ${data.message}`, 'success');
                this.indexed = false;
                this.disableQueryInput();
                document.getElementById('resultsList').innerHTML = '';
                this.updateStatus();
            } else {
                this.showMessage(`❌ ${data.message}`, 'error');
            }
        } catch (error) {
            this.showMessage(`Error clearing: ${error.message}`, 'error');
        } finally {
            clearBtn.disabled = false;
        }
    }

    showMessage(message, type) {
        const messageDiv = document.getElementById('uploadMessage');
        messageDiv.textContent = message;
        messageDiv.className = `message show ${type}`;
        setTimeout(() => {
            messageDiv.classList.remove('show');
        }, 5000);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new RAGApp();
});

// NeuralBlitz Documentation JavaScript
// Generated: 2026-02-08

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initScrollEffects();
    initCodeHighlighting();
    initSearch();
});


function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabId = btn.dataset.tab;
            
            // Remove active from all buttons and contents
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            // Add active to clicked button and corresponding content
            btn.classList.add('active');
            document.getElementById(tabId)?.classList.add('active');
        });
    });
}


function initScrollEffects() {
    const sections = document.querySelectorAll('.section');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, {
        threshold: 0.1
    });
    
    sections.forEach(section => {
        section.classList.add('fade-in');
        observer.observe(section);
    });
    
    // Smooth scroll for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}


function initCodeHighlighting() {
    // Add line numbers and copy buttons to code blocks
    document.querySelectorAll('pre code').forEach(block => {
        // Add line highlighting class
        block.parentElement.classList.add('code-block');
    });
}


function initSearch() {
    // Simple search functionality
    const searchInput = document.querySelector('.search-input');
    if (!searchInput) return;
    
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        const cards = document.querySelectorAll('.capability-card, .sdk-card, .example-card');
        
        cards.forEach(card => {
            const text = card.textContent.toLowerCase();
            card.style.display = text.includes(query) ? 'block' : 'none';
        });
    });
}


class NeuralBlitzDocs {
    constructor() {
        this.baseUrl = '/api/v1';
        this.apiKey = null;
        this.init();
    }
    
    init() {
        // Check for stored API key
        const storedKey = localStorage.getItem('neuralblitz_api_key');
        if (storedKey) {
            this.apiKey = storedKey;
        }
    }
    
    setApiKey(key) {
        this.apiKey = key;
        localStorage.setItem('neuralblitz_api_key', key);
    }
    
    async testConnection() {
        if (!this.apiKey) {
            return { error: 'API key not set' };
        }
        
        try {
            const response = await fetch(`${this.baseUrl}/capabilities`, {
                headers: {
                    'X-API-Key': this.apiKey
                }
            });
            
            return await response.json();
        } catch (error) {
            return { error: error.message };
        }
    }
    
    async getCapabilities() {
        return this.testConnection();
    }
    
    async makeRequest(method, endpoint, data = null) {
        if (!this.apiKey) {
            return { error: 'API key not set' };
        }
        
        const options = {
            method,
            headers: {
                'X-API-Key': this.apiKey,
                'Content-Type': 'application/json'
            }
        };
        
        if (data) {
            options.body = JSON.stringify(data);
        }
        
        try {
            const response = await fetch(`${this.baseUrl}${endpoint}`, options);
            return await response.json();
        } catch (error) {
            return { error: error.message };
        }
    }
    
    // Convenience methods
    async processQuantum(inputData, current = 20.0, duration = 200.0) {
        return this.makeRequest('POST', '/core/process', {
            input_data: inputData,
            current,
            duration
        });
    }
    
    async evolveNetwork(numRealities = 4, nodesPerReality = 50, cycles = 50) {
        return this.makeRequest('POST', '/core/evolve', {
            num_realities: numRealities,
            nodes_per_reality: nodesPerReality,
            cycles
        });
    }
    
    async runAgent(agentType, task) {
        return this.makeRequest('POST', '/agent/run', {
            agent_type: agentType,
            task
        });
    }
    
    async getConsciousnessLevel() {
        return this.makeRequest('GET', '/consciousness/level');
    }
}


// Initialize when DOM is ready
const docs = new NeuralBlitzDocs();


// Export for use in other scripts
window.NeuralBlitzDocs = NeuralBlitzDocs;
window.docs = docs;

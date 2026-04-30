const API_BASE = "http://localhost:8000/api/v1";

function showTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.sidebar li').forEach(li => li.classList.remove('active'));
    document.getElementById(tabId).classList.add('active');
    event.currentTarget.classList.add('active');
}

window.onload = async () => {
    try {
        const res = await fetch("http://localhost:8000/health");
        const data = await res.json();
        document.getElementById("health-card").innerHTML = `<h3>API Health</h3><p>Status: <span style="color:var(--success)">${data.status}</span></p><p style="font-size:1rem;color:var(--text-muted);margin-top:5px;">Uptime: ${data.uptime_seconds}s</p>`;
        
        const res2 = await fetch("http://localhost:8000/models/info");
        const data2 = await res2.json();
        document.getElementById("models-card").innerHTML = `<h3>Loaded Models</h3><p>${Object.keys(data2).length} families</p><p style="font-size:1rem;color:var(--text-muted);margin-top:5px;">Ready for inference</p>`;
        
        loadSentimentChart();
        checkPipelineStatus(); // Initial check in case it's already running
    } catch (e) {
        document.getElementById("health-card").innerHTML = `<p style="color:var(--danger)">Error connecting to API</p>`;
    }
};

async function loadSentimentChart() {
    try {
        const res = await fetch(`${API_BASE}/timeseries/sentiment`);
        const data = await res.json();
        if(data.error) return;
        
        const ctx = document.getElementById('sentimentChart').getContext('2d');
        const labels = data.historical.map(d => d.decade);
        const values = data.historical.map(d => d.avg_sentiment);
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Avg Sentiment Polarity',
                    data: values,
                    borderColor: '#6366f1',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#f8fafc' } }
                },
                scales: {
                    x: { ticks: { color: '#a1a1aa' }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { ticks: { color: '#a1a1aa' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            }
        });
    } catch (e) {
        console.error("Failed to load chart:", e);
    }
}

// Polling Variables
let pipelinePollInterval = null;

async function runPipeline() {
    const btn = document.getElementById("runPipelineBtn");
    btn.disabled = true;
    updatePipelineStatus("Starting...", "running");
    
    try {
        const res = await fetch(`${API_BASE}/pipeline/run`, { method: 'POST' });
        const data = await res.json();
        
        // Start polling logs
        startPipelinePolling();
    } catch (e) {
        updatePipelineStatus("Error", "error");
        appendPipelineLog(`[ERROR] ${e.message}`, "error");
        btn.disabled = false;
    }
}

function startPipelinePolling() {
    if (pipelinePollInterval) clearInterval(pipelinePollInterval);
    updatePipelineStatus("Running", "running");
    document.getElementById("runPipelineBtn").disabled = true;
    
    pipelinePollInterval = setInterval(async () => {
        await checkPipelineStatus();
    }, 1000);
}

async function checkPipelineStatus() {
    try {
        const res = await fetch(`${API_BASE}/pipeline/status`);
        if (!res.ok) return;
        const data = await res.json();
        
        const viewer = document.getElementById("pipelineLogViewer");
        
        // Only update if logs changed to prevent flickering
        if (data.logs && data.logs !== window._lastLogs) {
            viewer.innerHTML = escapeHtml(data.logs)
                .split('\n')
                .map(line => `<div class="terminal-line">${line}</div>`)
                .join('');
            window._lastLogs = data.logs;
            viewer.scrollTop = viewer.scrollHeight;
        }
        
        if (!data.is_running && pipelinePollInterval) {
            clearInterval(pipelinePollInterval);
            pipelinePollInterval = null;
            document.getElementById("runPipelineBtn").disabled = false;
            updatePipelineStatus("Completed", "completed");
            appendPipelineLog("\n[SYSTEM] Pipeline execution finished.", "info");
        } else if (data.is_running && !pipelinePollInterval) {
            startPipelinePolling(); // Resumes polling if refreshed while running
        }
    } catch (e) {
        console.error("Polling error:", e);
    }
}

function updatePipelineStatus(text, type) {
    const badge = document.getElementById("pipelineStatus");
    badge.innerText = text;
    badge.className = `pipeline-status-badge ${type}`;
}

function appendPipelineLog(text, type="") {
    const viewer = document.getElementById("pipelineLogViewer");
    const div = document.createElement("div");
    div.className = `terminal-line ${type}`;
    div.innerText = text;
    viewer.appendChild(div);
    viewer.scrollTop = viewer.scrollHeight;
}

function escapeHtml(unsafe) {
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}

async function classifyText() {
    const text = document.getElementById("classifyText").value;
    const resBox = document.getElementById("classifyResult");
    resBox.innerText = "Processing...";
    try {
        const res = await fetch(`${API_BASE}/classify/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        const data = await res.json();
        resBox.innerText = JSON.stringify(data, null, 2);
    } catch (e) {
        resBox.innerText = "Error: " + e.message;
    }
}

async function predictInfluence() {
    const text = document.getElementById("regressText").value;
    const resBox = document.getElementById("regressionResult");
    resBox.innerText = "Processing...";
    try {
        const res = await fetch(`${API_BASE}/regression/`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ full_text: text })
        });
        const data = await res.json();
        resBox.innerText = JSON.stringify(data, null, 2);
    } catch (e) {
        resBox.innerText = "Error: " + e.message;
    }
}

async function getRecommendations() {
    const query = document.getElementById("recommendQuery").value;
    const resBox = document.getElementById("recommendResult");
    resBox.innerText = "Processing...";
    try {
        const res = await fetch(`${API_BASE}/recommend/?query=${encodeURIComponent(query)}&n=5`);
        const data = await res.json();
        resBox.innerText = JSON.stringify(data, null, 2);
    } catch (e) {
        resBox.innerText = "Error: " + e.message;
    }
}

async function loadClusters() {
    const resBox = document.getElementById("clustersResult");
    resBox.innerText = "Loading...";
    try {
        const res = await fetch(`${API_BASE}/clusters/`);
        const data = await res.json();
        resBox.innerText = JSON.stringify(data, null, 2);
    } catch (e) {
        resBox.innerText = "Error: " + e.message;
    }
}

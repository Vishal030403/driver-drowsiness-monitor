// Authentication-aware Dashboard Logic

const API_BASE_URL = 'http://127.0.0.1:8000';
const WS_BASE_URL = 'ws://127.0.0.1:8000';

let authToken = localStorage.getItem('authToken') || null;
let currentUser = null;
let journeyTimer = null;
let startTime = null;
let alertCount = 0;
let sessionId = null;
let socket = null;
let frameIntervalId = null;
let videoStream = null;
let isDetecting = false;

const captureCanvas = document.createElement('canvas');
const captureContext = captureCanvas.getContext('2d');

// NEW: overlay canvas for green face rectangle
const overlayCanvas = document.getElementById('faceOverlay');
const overlayCtx = overlayCanvas ? overlayCanvas.getContext('2d') : null;

function authHeaders(additional = {}) {
    const headers = { ...additional };
    if (authToken) {
        headers['Authorization'] = `Bearer ${authToken}`;
    }
    return headers;
}

function clearAuthState() {
    localStorage.removeItem('authToken');
    authToken = null;
    currentUser = null;
}

async function fetchCurrentUser() {
    if (!authToken) return;
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/auth/me`, {
            headers: authHeaders()
        });

        if (!response.ok) {
            throw new Error('Session expired');
        }

        currentUser = await response.json();
    } catch (error) {
        console.error('Failed to fetch current user:', error);
        throw error;
    }
}

// Handle Login
async function handleLogin(event) {
    event.preventDefault();
    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value.trim();

    if (!username || !password) {
        alert('Please enter both username and password.');
        return;
    }

    try {
        console.log('Attempting login to:', `${API_BASE_URL}/api/auth/login`);
        
        const formData = new URLSearchParams();
        formData.append('username', username);
        formData.append('password', password);

        const response = await fetch(`${API_BASE_URL}/api/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: formData
        });
        
        const payload = await response.json().catch(() => ({}));

        if (!response.ok) {
            throw new Error(payload.detail || 'Invalid username or password.');
        }

        authToken = payload.access_token;
        localStorage.setItem('authToken', authToken);
        await fetchCurrentUser();
        await showDashboard();
    } catch (error) {
        console.error('Login error:', error);
        alert(error.message || 'Unable to login. Please check if the backend is running on port 8000.');
    }
}

// Handle Signup
async function handleSignup(event) {
    event.preventDefault();
    const username = document.getElementById('newUsername').value.trim();
    const email = document.getElementById('email').value.trim();
    const password = document.getElementById('newPassword').value.trim();

    if (!username || !email || !password) {
        alert('Please fill in all signup fields.');
        return;
    }

    if (username.length < 3) {
        alert('Username must be at least 3 characters long.');
        return;
    }

    if (password.length < 6) {
        alert('Password must be at least 6 characters long.');
        return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
        alert('Please enter a valid email address.');
        return;
    }

    try {
        console.log('Attempting signup to:', `${API_BASE_URL}/api/auth/signup`);
        
        const response = await fetch(`${API_BASE_URL}/api/auth/signup`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, email, password })
        });
        
        const payload = await response.json().catch(() => ({}));

        if (!response.ok) {
            throw new Error(payload.detail || 'Signup failed. Please try again.');
        }

        alert('Signup successful! Please login with your credentials.');
        showLogin();
        document.getElementById('username').value = username;
        
    } catch (error) {
        console.error('Signup error:', error);
        alert(error.message || 'Unable to sign up. Please check if the backend is running on port 8000.');
    }
}

// Toggle Signup/Login Forms
function showSignup() {
    document.getElementById('loginForm').classList.add('hidden');
    document.getElementById('signupForm').classList.remove('hidden');
}

function showLogin() {
    document.getElementById('signupForm').classList.add('hidden');
    document.getElementById('loginForm').classList.remove('hidden');
}

// Dashboard View
async function showDashboard() {
    if (!currentUser) return;
    document.getElementById('loginScreen').classList.add('hidden');
    document.getElementById('dashboardScreen').classList.remove('hidden');
    document.getElementById('welcomeMessage').textContent = `Welcome, ${currentUser.username}!`;
    await loadJourneyHistory();
}

// Logout
async function logout() {
    await stopDetection();
    clearAuthState();
    document.getElementById('dashboardScreen').classList.add('hidden');
    document.getElementById('loginScreen').classList.remove('hidden');
}

async function startDetection() {
    if (isDetecting) return;

    if (!authToken) {
        alert('Please login before starting detection.');
        return;
    }

    const video = document.getElementById('videoFeed');
    const startBtn = document.getElementById('startDetection');
    const stopBtn = document.getElementById('stopDetection');
    document.getElementById('lastAlert').textContent = 'None';
    document.getElementById('journeyTimer').textContent = '00:00:00';

    try {
        startBtn.classList.add('hidden');
        stopBtn.classList.remove('hidden');
        updateDetectionStatus('Connecting to camera...', 'bg-yellow-500', 'text-yellow-800');
    
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            throw new Error('Camera not supported in this browser.');
        }
    
        videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = videoStream;
        await video.play();
    
        // Wait for video metadata to load properly
        await new Promise((resolve) => {
            if (video.videoWidth && video.videoHeight) {
                resolve();
            } else {
                video.addEventListener('loadedmetadata', resolve, { once: true });
            }
        });
    
        // NOW size overlay canvas to match video
        if (overlayCanvas && overlayCtx) {
            overlayCanvas.width = video.videoWidth;
            overlayCanvas.height = video.videoHeight;
            console.log(`Canvas sized: ${overlayCanvas.width}x${overlayCanvas.height}`);
        }

        isDetecting = true;
        alertCount = 0;
        startTime = new Date();
        if (journeyTimer) {
            clearInterval(journeyTimer);
        }
        startJourneyTimer();

        updateDetectionStatus('Starting session...', 'bg-blue-500', 'text-white');
        sessionId = await createSession();

        updateDetectionStatus('Connecting to detector...', 'bg-blue-500', 'text-white');
        await connectWebSocket(sessionId);

        updateDetectionStatus('Detection Active', 'bg-green-500', 'text-green-700');
    } catch (error) {
        console.error('Start detection error:', error);
        alert(error.message || 'Unable to start detection.');
        await stopDetection();
    }
}

async function stopDetection() {
    if (!isDetecting && !videoStream && !sessionId && !socket) {
        return;
    }

    const video = document.getElementById('videoFeed');
    const startBtn = document.getElementById('startDetection');
    const stopBtn = document.getElementById('stopDetection');

    isDetecting = false;

    if (frameIntervalId) {
        clearInterval(frameIntervalId);
        frameIntervalId = null;
    }

    if (socket) {
        socket.onclose = null;
        socket.close();
        socket = null;
    }

    if (sessionId) {
        await endSession(sessionId);
        sessionId = null;
    }

    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }

    video.srcObject = null;

    // Clear overlay
    if (overlayCanvas && overlayCtx) {
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }

    stopBtn.classList.add('hidden');
    startBtn.classList.remove('hidden');
    updateDetectionStatus('Detection Inactive', 'bg-gray-400', 'text-gray-600');

    if (journeyTimer) {
        clearInterval(journeyTimer);
        journeyTimer = null;
    }

    await refreshJourneyData();
    startTime = null;
    console.log("Camera stopped.");
}

function updateDetectionStatus(message, dotColor, textColor) {
    const status = document.getElementById('detectionStatus');
    status.innerHTML = `
        <div class="w-3 h-3 ${dotColor} rounded-full mr-2"></div>
        <span class="${textColor} font-medium">${message}</span>
    `;
}

async function createSession() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/session/start`, {
            method: 'POST',
            headers: authHeaders()
        });
        
        const payload = await response.json().catch(() => ({}));

        if (!response.ok) {
            throw new Error(payload.detail || 'Failed to start session.');
        }

        return payload.session_id;
    } catch (error) {
        console.error('Create session error:', error);
        throw error;
    }
}

async function endSession(id) {
    try {
        await fetch(`${API_BASE_URL}/api/session/end/${id}`, {
            method: 'POST',
            headers: authHeaders()
        });
    } catch (error) {
        console.warn('Failed to end session', error);
    }
}

function connectWebSocket(id) {
    return new Promise((resolve, reject) => {
        if (!authToken) {
            reject(new Error('Missing auth token'));
            return;
        }

        const tokenParam = encodeURIComponent(authToken);
        const wsUrl = `${WS_BASE_URL}/ws/${id}?token=${tokenParam}`;
        console.log('Connecting to WebSocket ->', wsUrl);

        socket = new WebSocket(wsUrl);

        socket.onopen = () => {
            console.log('WebSocket connected');
            startFrameStreaming();
            resolve();
        };

        socket.onmessage = event => {
            try {
                const data = JSON.parse(event.data);
                console.log('WS message received:', {
                    type: data.type,
                    status: data.status,
                    alert: data.alert,
                    confidence: data.confidence
                });
                if (data.type === 'detection') {
                    handleDetectionResult(data);
                }
            } catch (err) {
                console.warn('Failed to parse WS message', err, event.data);
            }
        };

        socket.onerror = err => {
            console.error('WebSocket error', err);
            reject(new Error('Failed to connect to detector.'));
        };

        socket.onclose = ev => {
            console.warn('WebSocket closed', ev.code, ev.reason);
            if (isDetecting) {
                updateDetectionStatus('Connection lost. Stopping detection.', 'bg-red-500', 'text-red-600');
                stopDetection();
            }
        };
    });
}

function startFrameStreaming() {
    const video = document.getElementById('videoFeed');

    frameIntervalId = setInterval(() => {
        if (!socket || socket.readyState !== WebSocket.OPEN) {
            console.debug('Socket not open, skipping frame send. State:', socket ? socket.readyState : 'no-socket');
            return;
        }

        if (!video || !video.videoWidth || !video.videoHeight) {
            console.debug('Video not ready, width/height are zero');
            return;
        }

        captureCanvas.width = video.videoWidth;
        captureCanvas.height = video.videoHeight;
        captureContext.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

        // Use dataURL; backend now supports with/without prefix
        const frameData = captureCanvas.toDataURL('image/jpeg', 0.6);
        try {
            socket.send(JSON.stringify({ type: 'frame', frame: frameData }));
        } catch (err) {
            console.error('Failed to send frame:', err);
        }
    }, 300);
}

// NEW: draw green rectangle on overlay
function drawFaceBox(faceCoords) {
    if (!overlayCanvas || !overlayCtx || !faceCoords) return;

    const video = document.getElementById('videoFeed');
    const [x, y, w, h] = faceCoords;

    // Clear old box
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    if (!video || !video.videoWidth || !video.videoHeight) return;

    // In most cases, coordinates match 1:1
    const scaleX = overlayCanvas.width / video.videoWidth;
    const scaleY = overlayCanvas.height / video.videoHeight;

    overlayCtx.lineWidth = 3;
    overlayCtx.strokeStyle = 'lime';
    overlayCtx.strokeRect(x * scaleX, y * scaleY, w * scaleX, h * scaleY);
}

function handleDetectionResult(result) {
    console.log("Detection Result:", result);

    const alertPanel = document.getElementById('alertPanel');
    const now = new Date();
    const confidence = result.confidence || 0;

    const faceCoords = result.face_coords || null;
    const faceDetected = !!result.face_detected;

    // ALWAYS draw or clear face box based on detection
    if (faceDetected && faceCoords) {
        drawFaceBox(faceCoords);
    } else {
        // Clear the overlay when no face
        if (overlayCanvas && overlayCtx) {
            overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
        }
    }

    // Rest of your existing code...
    let bg = "";
    let icon = "";
    let message = "";

    if (result.status === "drowsy") {
        icon = "🚨";
        message = "SHER uth jaa neend nhi leni warna gaadi rok k chai pii le ";
        bg = "bg-red-600";
        playAlertSound();
        alertCount++;
        document.getElementById("lastAlert").textContent = now.toLocaleTimeString();
    } 
    else if (result.status === "no_face") {
        icon = "❓";
        message = "Arey SHER , muh to dikhao ";
        bg = "bg-yellow-500";
    }
    else {
        icon = "😊";
        message = "SHER is handsome and alert ";
        bg = "bg-green-600";
    }

    alertPanel.className = `${bg} rounded-lg p-6 text-center text-white`;
    alertPanel.innerHTML = `
        <div class="text-3xl mb-2">${icon}</div>
        <h3 class="text-xl font-bold mb-2">${message}</h3>
        <p class="text-xs opacity-80">Confidence: ${(confidence * 100).toFixed(1)}%</p>
    `;
}

// Beep sound
function playAlertSound() {
    try {
        const context = new AudioContext();
        const oscillator = context.createOscillator();
        const gainNode = context.createGain();

        oscillator.connect(gainNode);
        gainNode.connect(context.destination);

        oscillator.type = 'square';
        oscillator.frequency.value = 800;
        gainNode.gain.setValueAtTime(0.1, context.currentTime);

        oscillator.start();
        oscillator.stop(context.currentTime + 1);
    } catch (error) {
        console.warn('Could not play alert sound:', error);
    }
}

// Journey Timer
function startJourneyTimer() {
    const timerElement = document.getElementById('journeyTimer');
    journeyTimer = setInterval(() => {
        const now = new Date();
        const diff = new Date(now - startTime);
        const hours = String(diff.getUTCHours()).padStart(2, '0');
        const mins = String(diff.getUTCMinutes()).padStart(2, '0');
        const secs = String(diff.getUTCSeconds()).padStart(2, '0');
        timerElement.textContent = `${hours}:${mins}:${secs}`;
    }, 1000);
}

async function refreshJourneyData() {
    if (!authToken) return;
    try {
        await loadJourneyHistory();
    } catch (error) {
        console.warn('Unable to refresh journey data', error);
    }
}

// Load Journey History
async function loadJourneyHistory() {
    if (!authToken) return;

    const tbody = document.getElementById('journeyHistory');
    tbody.innerHTML = `
        <tr>
            <td colspan="5" class="px-4 py-3 text-sm text-gray-500">Loading...</td>
        </tr>
    `;

    try {
        const response = await fetch(`${API_BASE_URL}/api/session`, {
            headers: authHeaders()
        });
        
        if (!response.ok) {
            throw new Error('Failed to load sessions');
        }

        const sessions = await response.json();

        if (!Array.isArray(sessions) || sessions.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="5" class="px-4 py-3 text-sm text-gray-500 text-center">
                        No journeys yet. Start a detection session to see history.
                    </td>
                </tr>
            `;
            updateJourneyStats([]);
            return;
        }

        tbody.innerHTML = '';
        sessions.forEach(session => {
            const start = new Date(session.start_time);
            const end = session.end_time ? new Date(session.end_time) : null;
            const durationSeconds = end
                ? Math.max(0, Math.floor((end - start) / 1000))
                : Math.max(0, Math.floor((Date.now() - start.getTime()) / 1000));
            const durationLabel = end ? formatDuration(durationSeconds) : 'In progress';
            const firstAlert = session.total_alerts > 0 ? 'Triggered' : 'None';
            const statusClass = session.total_alerts > 0 ? 'text-red-600' : 'text-green-600';
            const statusLabel = session.status === 'active'
                ? 'Active'
                : (session.total_alerts > 0 ? 'Drowsy' : 'Safe');

            const row = document.createElement('tr');
            row.innerHTML = `
                <td class="px-4 py-3 text-sm text-gray-700">${start.toLocaleDateString()}</td>
                <td class="px-4 py-3 text-sm text-gray-700">${durationLabel}</td>
                <td class="px-4 py-3 text-sm text-gray-700">${firstAlert}</td>
                <td class="px-4 py-3 text-sm text-gray-700">${session.total_alerts}</td>
                <td class="px-4 py-3 text-sm font-semibold ${statusClass}">${statusLabel}</td>
            `;
            tbody.appendChild(row);
        });

        updateJourneyStats(sessions);
    } catch (error) {
        console.error('History load error:', error);
        tbody.innerHTML = `
            <tr>
                <td colspan="5" class="px-4 py-3 text-sm text-red-600 text-center">
                    Unable to load journey history. ${error.message}
                </td>
            </tr>
        `;
        updateJourneyStats([]);
    }
}

function updateJourneyStats(sessions) {
    const total = sessions.length;
    document.getElementById('totalJourneys').textContent = total;

    const safeCount = sessions.filter(session => session.total_alerts === 0).length;
    const safePercent = total ? Math.round((safeCount / total) * 100) : 0;
    document.getElementById('safeJourneys').textContent = `${safePercent}%`;

    const avgDrowsySeconds = total
        ? sessions.reduce((sum, session) => sum + (session.total_drowsy_time || 0), 0) / total
        : 0;
    document.getElementById('avgDrowsyTime').textContent = formatAverageDrowsyTime(avgDrowsySeconds);
}

function formatDuration(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    return `${h}h ${m}m ${s}s`;
}

function formatAverageDrowsyTime(seconds) {
    const totalMinutes = Math.floor(seconds / 60);
    const hours = Math.floor(totalMinutes / 60);
    const minutes = totalMinutes % 60;
    return `${hours}h ${minutes}m`;
}

// Auto-load dashboard if user is logged in
window.onload = async function() {
    console.log('Application starting...');
    console.log('API Base URL:', API_BASE_URL);
    console.log('WS Base URL:', WS_BASE_URL);
    
    try {
        const response = await fetch(`${API_BASE_URL}/health`, { 
            method: 'GET',
            mode: 'cors'
        });
        if (response.ok) {
            console.log('✅ Backend is reachable');
        } else {
            console.warn('⚠️ Backend returned non-OK status:', response.status);
        }
    } catch (error) {
        console.error('❌ Cannot reach backend at', API_BASE_URL);
        console.error('Please ensure backend is running on port 8000.');
        alert('Cannot connect to backend server. Please ensure the backend is running on port 8000.');
    }
    
    if (!authToken) {
        console.log('No auth token found, showing login screen');
        return;
    }

    try {
        await fetchCurrentUser();
        await showDashboard();
        console.log('✅ Auto-login successful');
    } catch (error) {
        console.warn('Auto-login failed:', error);
        clearAuthState();
    }
};

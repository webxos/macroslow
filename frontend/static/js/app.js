class AnnotationApp {
    constructor() {
        this.ws = new WebSocket(`ws://${location.host}/ws/1`);
        this.canvas = document.getElementById('annotationCanvas');
        this.annotations = [];
        this.setupEventListeners();
    }

    setupEventListeners() {
        this.canvas.addEventListener('click', (e) => this.addAnnotation(e));
        this.ws.onmessage = (event) => this.handleMessage(JSON.parse(event.data));
    }

    addAnnotation(e) {
        const rect = this.canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width * 100;
        const y = (e.clientY - rect.top) / rect.height * 100;
        const annotation = { text: prompt('Enter annotation text:'), x, y, token: localStorage.getItem('token') };
        this.ws.send(JSON.stringify(annotation));
        this.annotations.push(annotation);
        this.renderAnnotations();
    }

    handleMessage(data) {
        if (data.user && data.text) {
            this.annotations.push(data);
            this.renderAnnotations();
        }
    }

    renderAnnotations() {
        this.canvas.innerHTML = '';
        this.annotations.forEach(a => {
            const span = document.createElement('span');
            span.style.position = 'absolute';
            span.style.left = `${a.x}%`;
            span.style.top = `${a.y}%`;
            span.textContent = `${a.user}: ${a.text}`;
            this.canvas.appendChild(span);
        });
    }
}

new AnnotationApp();

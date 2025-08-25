class Annot8Viz {
    constructor() {
        this.canvas = document.getElementById('annot8Canvas');
        this.ctx = this.canvas.getContext('2d');
        this.annotations = [];
        this.fetchAnnotations();
    }

    async fetchAnnotations() {
        const response = await fetch('/api/annot8/enhanced', { headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` } });
        this.annotations = await response.json();
        this.render();
    }

    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.annotations.forEach(a => {
            this.ctx.beginPath();
            this.ctx.arc(a.x * this.canvas.width / 100, a.y * this.canvas.height / 100, 5, 0, 2 * Math.PI);
            this.ctx.fillStyle = 'red';
            this.ctx.fill();
            this.ctx.fillText(a.text, a.x * this.canvas.width / 100 + 10, a.y * this.canvas.height / 100);
        });
    }
}

document.addEventListener('DOMContentLoaded', () => new Annot8Viz());

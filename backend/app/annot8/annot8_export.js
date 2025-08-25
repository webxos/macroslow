class Annot8Export {
    constructor() {
        this.exportBtn = document.getElementById('exportBtn');
        this.setupEventListeners();
    }

    setupEventListeners() {
        this.exportBtn.addEventListener('click', () => this.exportAnnotations());
    }

    async exportAnnotations() {
        const token = localStorage.getItem('token');
        const response = await fetch('/api/annot8/export?format=csv', {
            headers: { 'Authorization': `Bearer ${token}` }
        });
        if (response.ok) {
            const data = await response.json();
            const blob = new Blob([data.data], { type: 'text/csv' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'annotations.csv';
            a.click();
            window.URL.revokeObjectURL(url);
        } else {
            alert('Export failed');
        }
    }
}

document.addEventListener('DOMContentLoaded', () => new Annot8Export());

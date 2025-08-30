// app.component.ts: Angular.js dashboard for real-time quantum metrics
// CUSTOMIZATION POINT: Update metrics and visualizations for specific datasets
import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Component({
  selector: 'app-root',
  template: `
    <h1>Quantum Mathematics Dashboard</h1>
    <div *ngIf="metrics$ | async as metrics">
      <plotly-plot [data]="metrics.data" [layout]="metrics.layout"></plotly-plot>
    </div>
  `
})
export class AppComponent implements OnInit {
  metrics$: Observable<any>;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.metrics$ = this.http.get('http://localhost:8000/quantum_metrics');
  }
}
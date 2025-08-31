# AngularJS Dashboard for GLASTONBURY 2048

## Overview

The **AngularJS dashboard** visualizes **Apple Watch** biometrics and **AirTag** location data in real-time, supporting health professionals in disaster relief and industrial settings. It integrates with the **2048-bit AES VPN chain** and **MCP**.

## Setup

Create an Angular app:
```bash
ng new glastonbury-dashboard
cd glastonbury-dashboard
npm install angular
```

Update `src/app/app.component.ts`:
```typescript
import { Component } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-root',
  template: `
    <div>
      <h1>GLASTONBURY 2048 Dashboard</h1>
      <div *ngFor="let result of results">
        <p>Heart Rate: {{ result.heart_rate }}</p>
        <p>SpO2: {{ result.spo2 }}</p>
        <p>Location: {{ result.location }}</p>
        <p>Environment: {{ result.environment }}</p>
        <p *ngIf="result.alert">911 Alert: {{ result.alert }}</p>
      </div>
    </div>
  `
})
export class AppComponent {
  results: any[] = [];
  constructor(private http: HttpClient) {
    this.http.get('http://localhost:8000/vpn-chain?token=YOUR_JWT&heart_rate=120&spo2=88&location=lat:6.5244,lon:3.3792&environment=cave')
      .subscribe(data => this.results.push(data));
  }
}
```

## Running the Dashboard

1. Run: `ng serve`.
2. Visit `http://localhost:4200` to see biometric and location data.
3. Alerts trigger for critical values (heart rate > 100, SpO2 < 90).

## Security

Data is encrypted with **2048-bit AES** via **PROJECT DUNES**, ensuring secure visualization.

# MAML/MU Guide: U - User Interface (INFINITY UI)

## Overview
The INFINITY UI in MAML/MU provides a single-page HTML interface (`infinity.html`) for syncing and exporting API/IoT data, integrated with GLASTONBURY 2048.

## Technical Details
- **MAML Role**: Defines UI-driven workflows in `infinity_server.py`.
- **MU Role**: Validates UI outputs in `infinity_workflow_validation.mu.md`.
- **Implementation**: Uses Axios for backend communication, embedded Tailwind CSS.
- **Dependencies**: `axios`.

## Use Cases
- Enable Nigerian clinicians to sync medical APIs via UI.
- Export IoT data for SPACE HVAC monitoring.
- Generate RAG datasets with UI controls.

## Guidelines
- **Compliance**: Ensure UI data is encrypted (HIPAA-compliant).
- **Best Practices**: Optimize UI for low-bandwidth environments.
- **Code Standards**: Use semantic HTML for accessibility.

## Example
```html
<button id="sync-btn" class="bg-blue-600">SYNC</button>
```
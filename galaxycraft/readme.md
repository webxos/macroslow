🌌 GALAXYCRAFT: A Browser-Based Space MMO Ecosystem
GALAXYCRAFT is an ambitious, open-source, browser-based space MMO ecosystem built with Three.js, React Three Fiber (R3F), Cannon.js, and React, targeting seamless 30-60 FPS performance across desktop and mobile devices. Inspired by games like Unreal Tournament, StarSiege Tribes, and Team Fortress 2, GALAXYCRAFT offers a variety of game modes, from open-world crafting to intense team-based shooters and flight simulators, all unified by a single universe with a centralized economy ($WEBXOS) and NFT-based crafting. This project invites the open-source community to contribute to its development, from game mechanics to multiplayer scalability.
Developed by WebXOS © 2025
🚀 Project Vision
GALAXYCRAFT aims to deliver a lightweight, modular, and extensible gaming platform that runs smoothly on low-end devices while supporting up to 1000 players in its MMO mode. The ecosystem includes multiple game modes, each leveraging a shared architecture for rendering, physics, and UI. Key features include:

Seamless Universe: A 3x3x3x3 grid (81 sectors) with 500 entities per sector, connected via 3D gateway portals for smooth transitions.
NFT Crafting: A Continuum Transfunctioner for crafting NFT engines with randomized stats, inspired by Diablo 2’s Horadric Cube.
Universal Economy: A centralized auction house for trading resources and NFTs using $WEBXOS currency.
Modular UI: Draggable, resizable pop-up windows with Tailwind CSS styling, ensuring a consistent and intuitive user experience.
Community-Driven: Open-source with a focus on modularity, enabling developers to create new game modes or extend existing ones.

This README serves as a guide for players, developers, and contributors, detailing each game mode, the technical stack, and how to get involved.
🎮 Game Modes
GALAXYCRAFT offers a diverse set of game modes, each designed to leverage the core engine while providing unique gameplay experiences. Below is an overview of each mode, including their mechanics, objectives, and technical details.
1. GALAXYCRAFT MMO (Crafting Open World)
The flagship mode, an open-world MMO supporting up to 1000 players, focused on mining, crafting, and trading in a seamless universe.

Core Features:

Mother Ship: A low-poly Star Destroyer-like ship with WASD controls, mouse-based 360° steering, and a 3rd-person camera. Players mine resources using the F key.
Drones: Three unique drones (Blue, Green, Red) for automated or manual mining, deployed from the neon green glowing home planet.
Home Planet: A large, neon green sphere with a neural network texture, serving as a hub for resource storage, auction house access, and drone deployment.
Mining Mechanics: Mine asteroids (low yield) and planets (high yield, deplete and respawn) for resources: Iron, Silicon, Gold, Platinum, Titanium, Uranium, Quantum, and Dark Matter (1% drop chance).
Continuum Transfunctioner: Craft NFT engines with random stats (Common: +20% speed, Rare: +50% boost, Ultra-Rare: +100% boost, increased armor).
Auction House: Trade resources and NFTs using $WEBXOS or credits, with a dedicated chat tab for trading.
Screen Saver/Auto-Mine Mode: Cycles camera between mother ship and drones every 10 seconds, showcasing mining with dynamic angles.
Chat System: Tabbed for Sector, Universal, and Auction chats, with multiplayer placeholders.


Technical Details:

Rendering: Three.js with React Three Fiber, using instanced meshes, LOD, and frustum culling for 500-entity sectors.
Physics: Cannon.js for ship movement, collisions, and mining interactions.
Networking: Mock WebSocket endpoints (/nft/mint, /token/transfer) stored in local JSON, with FastAPI and PostgreSQL planned for multiplayer.
Performance: Optimized for 60 FPS with WebGL optimizations.


Future Plans: Full multiplayer support with WebSocket-based networking, secure NFT minting, and a persistent world state.


2. GALAXYCRAFT: Red vs Blue (10v10 Team Deathmatch)
A fast-paced, first-person shooter with 10v10 team deathmatch, where players attack opposing team cores on a sci-fi Mars-like map.

Core Features:

Classes:
Scout: Equipped with a jetpack (inspired by Unreal Tournament), scope (right-click, 4x zoom), and scan (left-click). Light damage to cores, one-shot kills on drones/players.
Assault: Fastest run speed, shotgun (left-click, yellow neurots, high core damage at close range), energy shield (right-click, like Overwatch’s Reinhardt). One-shot kills on drones/players.
Medic: Homing neurots (left-click, purple), shield repair (right-click) for allies. Light core damage, one-shot kills on drones/players.


Gameplay: Players spawn behind mirrored tower bases, aiming to destroy the enemy core (300,000 HP). Drones auto-balance teams but cannot attack cores.
Controls: WASD for movement (Unreal Tournament-style), spacebar for auto-fire neurots (unlimited ammo), shift for jetpack (Scout only), and mouse for aiming.
UI: Simplified ESC menu with stats (score, health, armor, energy), settings (mouse sensitivity, invert W/S), and team status.
Map: Randomized, paintball-like Mars surface with boxes and obstacles, 500 entities per map.


Technical Details:

Rendering: Three.js with R3F, optimized for 30 FPS on mobile with simplified shaders.
Physics: Cannon.js for player movement and collision detection.
Networking: Local JSON for single-player, placeholders for Socket.io-based multiplayer.
Multiplayer Placeholders: $WEBXOS earnings and leaderboards stored locally.



3. GALAXYCRAFT: Tower Defense
A strategic mode where players defend a core against waves of enemy drones in a 500-entity sector.

Core Features:

Objective: Protect a core (300,000 HP) from 500 waves of drones using a neuro-gatling weapon (unlimited ammo).
Upgrades: Spend credits on health (+20), armor (+20), damage (+10%), speed (+10%), energy (+20%), or core repair (+5,000).
Controls: WASD for movement, spacebar for auto-fire, shift for boost, and ESC for a settings menu (mouse sensitivity, FOV, sound volume).
UI: Displays score, wave count, drone count, core health, and upgrade options in a pop-up window.
Map: Sci-fi arena with a neon green aesthetic, randomized asteroid and planet placements.


Technical Details:

Rendering: Instanced meshes and frustum culling for performance.
Physics: Cannon.js for drone movement and collision detection.
Performance: Targets 30-60 FPS on low-end devices.



4. GALAXYCRAFT: VTEC Paintball (3v3 Team Deathmatch)
A compact version of Red vs Blue, designed for 3v3 matches with a paintball-like feel.

Core Features:

Classes: Same as Red vs Blue (Scout, Assault, Medic), with identical mechanics.
Gameplay: Smaller, mirrored maps with faster-paced matches, focusing on one-shot kills and core attacks.
Controls: Identical to Red vs Blue, with WASD, spacebar for auto-fire, and class-specific abilities.
UI: Streamlined ESC menu, with mobile-optimized one-click buttons.


Technical Details:

Rendering: Optimized for mobile with reduced entity count (300 per map).
Networking: Local JSON with multiplayer placeholders.



5. GALAXYCRAFT: Rocket Building Studio
A creative mode for designing and simulating rockets, inspired by real-world aerospace engineering.

Core Features:

Parts Library: Includes body tubes, nose cones, fins, engines (e.g., Merlin 1D), and IoT components (Arduino, Raspberry Pi, sensors).
Tools: Select, move, rotate, scale, measure, wire, and code for precise rocket assembly.
UI: Left panel for parts and properties, top toolbar for actions (new, save, undo), and a 3D canvas for real-time visualization.
Physics: Cannon.js for simulating rocket dynamics and connections.
Output: Export designs as JSON for integration with other GALAXYCRAFT modes.


Technical Details:

Rendering: Three.js with OrbitControls and TransformControls for intuitive manipulation.
Optimization: Uses basic materials and GPU instancing for performance.



6. GALAXYCRAFT: Dog Fight (Flight Simulator)
A 360° flight simulator with WASD and mouse controls, focusing on dogfights in a 500-entity sector.

Core Features:

Ship: Jet-black with neon green accents, aerodynamic nose, and a glowing green flame during boost (shift key, 10-second lightspeed effect).
Weapons:
Gatling Neurots: Spacebar/left-click, three spiral streams in a cone, auto-aim with neon green crosshair lock and red-to-green scope transition.
Homing Neurots: R/right-click, 10x larger explosion radius with a nuke-like glitch effect.


Controls:
W: Forward thrust (fastest).
S: Vertical thrust up.
A/D: Side thrust left/right.
Q/E: Tilt left/right.
Z/X: 360° barrel roll.
C: 180° turn.
CTRL: Scan for red drone targets.
Shift: Lightspeed boost.
ESC: Lock/unlock mouse for joystick simulation.


Crosshair: 2x larger with range-finder markings, perfectly aligned with gatling neurots for pinpoint accuracy.
Map: Multicolored planets (Mars, Saturn-like) and neon green stars, with randomized asteroid belts.


Technical Details:

Rendering: Three.js with R3F, optimized for 30 FPS with frustum culling.
Physics: Cannon.js for ship and projectile dynamics.
Auto-Aim: Lightweight line box around locked targets for fly-by aiming.



7. GALAXYCRAFT: Tempest 2K25 Edition
A high-energy action mode with a focus on survival and combat in a neon-green sci-fi arena.

Core Features:

Gameplay: Defend against waves of red drones using a neuro-gatling weapon and three controllable drones (follow formation).
Controls: WASD for movement, spacebar for auto-fire, shift for boost, alt for combat mode, and tab for eco mode.
UI: Displays wave count, score, health, shield, and credits, with a settings menu for mouse sensitivity and invert Y-axis.
Map: 500-entity sector with neon green planets and randomized celestial bodies.


Technical Details:

Rendering: Optimized for 30 FPS with instanced meshes.
Physics: Cannon.js for drone and player interactions.



🛠️ Technical Architecture
GALAXYCRAFT’s architecture is designed for modularity, performance, and scalability, making it easy for contributors to extend or modify the game.
Core Technologies

Frontend: React, React Three Fiber, Three.js, Tailwind CSS.
Physics: Cannon.js for collisions, movement, and mining interactions.
Networking: Mock WebSocket endpoints (local JSON) with FastAPI and PostgreSQL planned for multiplayer.
Storage: Local JSON for single-player, .maml.ml (Markdown-based) save files for player progress and NFT assets.
Rendering Optimizations: Instanced meshes, LOD, frustum culling, and WebGL/WebGPU for 30-60 FPS.

Key Components

GameEngine: A central R3F component managing the game loop, input events, and state using useFrame.
Input System: Standardized keyboard, mouse, and touch input handling, with customizable sensitivity and invert options.
Asset Management: Uses useGLTF from @react-three/drei for efficient model loading.
UI System: Draggable, resizable pop-up windows (Inventory, Map, Drones, Auction House, Settings, Chat) with red "X" buttons and bottom toggles.
NFT Integration: Mock server-side validation for crafting NFT engines, with placeholders for blockchain integration.

Performance Goals

30-60 FPS: Achieved through instanced meshes, LOD, and frustum culling.
Mobile Support: One-click buttons for minimal mobile controls, optimized shaders, and reduced entity counts in smaller modes.
Scalability: Designed for 1000-player servers with zone-divided WebSocket updates.

💎 NFT Engine Crafting (Continuum Transfunctioner)
The Continuum Transfunctioner is a core feature of the MMO mode, allowing players to craft NFT engines with randomized stats. It draws inspiration from Diablo 2’s Horadric Cube and integrates with the universal auction house.

Mechanics:
Recipes: Combine resources (e.g., 4 Dark Matter + others) to craft engines.
Stats:
Common: +20% engine speed.
Rare: +50% boost chance, double shot.
Ultra-Rare: +100% boost chance, increased armor (gold-colored).


Rarity: Dark Matter (1% drop chance) required for high-tier engines.


Integration: Mock /nft/mint endpoint stores NFTs locally, with placeholders for blockchain integration via AWS Secrets Manager.
Trading: NFT engines can be auctioned in the universal marketplace, accessible from any sector.

🌍 Universal Marketplace
The Universal Marketplace unifies the economy across all game modes, allowing players to trade resources and NFTs using $WEBXOS or credits.

Features:
Listings for resources (e.g., Tritanium: 2,400 CR) and NFT engines (e.g., Advanced Warp Drive: 12,500 CR).
Dedicated Auction Chat tab with clickable links for trading.
Mock transactions stored in local JSON, with placeholders for WebSocket-based multiplayer trading.


Future Plans: Dynamic pricing and server-side validation for secure transactions.

🖥️ UI Design
The UI is consistent across all game modes, featuring:

Pop-Up Windows: Draggable, resizable windows for Inventory, Map, Drone Control, Auction House, Settings, Chat, and Continuum Transfunctioner, each with a red "X" to close and a corresponding bottom toggle.
Top Bar: Displays game title, credits ($WEBXOS), energy, shield, and battery stats.
Console: Shows system messages (e.g., “3 celestial bodies detected”).
Sector Info: Displays coordinates, velocity, throttle, and server status (mocked).
Mobile Support: One-click buttons for minimal controls, ensuring accessibility.

🚀 Getting Started
Prerequisites

Node.js: v16 or higher.
npm: For dependency management.
Browser: Chrome, Firefox, or Safari (WebGL/WebGPU support).

Installation

Clone the repository:git clone https://github.com/your-username/galaxycraft.git
cd galaxycraft


Install dependencies:npm install


Start the development server:npm start


Open http://localhost:3000 in your browser to play.

Project Structure
galaxycraft/
├── src/
│   ├── components/        # React components for UI and game elements
│   ├── scenes/            # R3F scenes for each game mode
│   ├── assets/            # 3D models, textures, and sounds
│   ├── physics/           # Cannon.js configurations
│   ├── api/               # Mock WebSocket endpoints and FastAPI stubs
├── public/                # Static assets and index.html
├── README.md              # This file
├── package.json           # Dependencies and scripts

🤝 Contributing
We welcome contributions from the open-source community to make GALAXYCRAFT a leading browser-based MMO. Here’s how you can get involved:
Contribution Guidelines

Fork the Repository: Create your own fork to work on features or bug fixes.
Follow Coding Standards:
Use ESLint for JavaScript/React linting.
Adhere to Tailwind CSS conventions for UI styling.
Write modular, reusable R3F components.


Submit Pull Requests:
Clearly describe your changes in the PR description.
Reference related issues or feature requests.
Ensure tests pass (if applicable).


Focus Areas:
Game Modes: Add new modes or enhance existing ones (e.g., new classes, maps, or mechanics).
Multiplayer: Implement WebSocket-based networking and FastAPI backend.
NFT Integration: Develop blockchain integration for secure NFT minting.
Performance: Optimize rendering and physics for low-end devices.
UI/UX: Improve pop-up windows, mobile controls, or visual effects.



Issues and Feature Requests

Check the Issues tab for open tasks.
Submit feature requests or bug reports with detailed descriptions.

Community

Join our Discord for discussions and playtests.
Follow updates on X for project news.

📋 Development Roadmap

Q1 2026: Complete single-player modes with local JSON storage.
Q2 2026: Implement multiplayer networking with WebSockets and FastAPI.
Q3 2026: Launch beta with NFT integration and universal marketplace.
Q4 2026: Support 1000-player servers and community modding tools.

📜 License
GALAXYCRAFT is licensed under the MIT License. Feel free to fork, modify, and distribute, but please credit WebXOS © 2025.
🙏 Acknowledgments

Three.js: For powering the 3D rendering engine.
React Three Fiber: For declarative 3D scene management.
Cannon.js: For lightweight physics simulations.
Open-Source Community: For contributions and inspiration.

Join us in building the future of browser-based gaming! 🌌

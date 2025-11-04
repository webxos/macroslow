// index.js - MACROSLOW Real-Time Resource Loader 🐪
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const ROOT = process.cwd();
const LIB = {
  templates: [],
  guides: [],
  yamlFiles: [],
  quantumReady: false
};

// Auto-scan repo (real-time on load)
function scanDir(dir, ext = null, target = []) {
  if (!fs.existsSync(dir)) return target;
  const files = fs.readdirSync(dir);
  files.forEach(file => {
    const fullPath = path.join(dir, file);
    if (fs.statSync(fullPath).isDirectory()) {
      scanDir(fullPath, ext, target);
    } else if (!ext || path.extname(file) === ext) {
      target.push(fullPath.replace(ROOT + '/', ''));
    }
  });
  return target;
}

// Load all resources
console.log('🚀 MACROSLOW Resource Library Booting...');
LIB.templates = scanDir(path.join(ROOT, 'templates'));
LIB.guides = scanDir(path.join(ROOT, 'guides'));
LIB.yamlFiles = scanDir(ROOT, '.yml').concat(scanDir(ROOT, '.yaml'));

// Check quantum deps
try {
  require.resolve('qiskit');
  LIB.quantumReady = true;
  console.log('✅ Qiskit ready');
} catch (e) {
  console.log('⚠️  Qiskit not installed (run: pip install -r requirements.txt)');
}

// Export for scripts / REPL
module.exports = LIB;

// CLI landing
if (require.main === module) {
  console.log('\n🐪 MACROSLOW LIBRARY LOADED');
  console.log(`   Templates: ${LIB.templates.length}`);
  console.log(`   Guides:    ${LIB.guides.length}`);
  console.log(`   YAMLs:     ${LIB.yamlFiles.length}`);
  console.log(`   Quantum:   ${LIB.quantumReady ? 'ONLINE' : 'OFFLINE'}\n`);
  console.log('Run `node` → `const lib = require("./index.js")` to explore.');
}

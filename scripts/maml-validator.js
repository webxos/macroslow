const fs = require('fs');
const yaml = require('js-yaml');

const files = [
  'nigeria/about/page10.md',
  'docs/emergency/part6.markdown'
];

files.forEach(file => {
  try {
    const content = fs.readFileSync(file, 'utf8');
    // Validate YAML front matter
    const frontMatter = content.match(/^---\n([\s\S]*?)\n---/);
    if (!frontMatter) {
      throw new Error('Missing YAML front matter');
    }
    yaml.load(frontMatter[1]);
    console.log(`${file}: YAML valid`);

    // Validate MAML structure (e.g., Code_Blocks section)
    if (!content.includes('## Code_Blocks')) {
      throw new Error('Missing MAML Code_Blocks section');
    }
    console.log(`${file}: MAML structure valid`);
  } catch (e) {
    console.error(`${file}: Validation failed - ${e.message}`);
    process.exit(1);
  }
});
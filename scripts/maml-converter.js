const fs = require('fs');
const path = require('path');

function reverseString(str) {
    return str.split('').reverse().join('');
}

function convertToMarkup(filePath) {
    try {
        const content = fs.readFileSync(filePath, 'utf8');
        // Extract YAML front matter (if any)
        const frontMatterMatch = content.match(/^---\n([\s\S]*?)\n---/);
        let frontMatter = '';
        let body = content;
        if (frontMatterMatch) {
            frontMatter = frontMatterMatch[0];
            body = content.slice(frontMatterMatch[0].length);
        }

        // Reverse body content for .mu format
        const reversedBody = reverseString(body.trim());
        const output = `${frontMatter}\n${reversedBody}`;

        // Save as .mu file
        const outputPath = path.join(
            path.dirname(filePath),
            path.basename(filePath, path.extname(filePath)) + '.mu'
        );
        fs.writeFileSync(outputPath, output);
        console.log(`Converted ${filePath} to ${outputPath}`);
    } catch (error) {
        console.error(`Error converting ${filePath}:`, error.message);
        process.exit(1);
    }
}

// Convert specified MAML files
const files = [
    'nigeria/about/page10.md',
    'docs/emergency/part6.markdown'
];

files.forEach(convertToMarkup);
// mamlParser.js
// Description: Parses MAML/MU files for BELUGA workflows.
// Validates and extracts metadata, code blocks, and schemas.
// Usage: Import and instantiate MAMLParser to process MAML files.

import { parse } from 'yaml';

class MAMLParser {
    /**
     * Parses a MAML/MU file into structured components.
     * @param {string} mamlContent - Raw MAML file content.
     * @returns {Object} - Parsed MAML components.
     */
    parse(mamlContent) {
        const [frontMatter, ...body] = mamlContent.split('---\n');
        const metadata = parse(frontMatter.replace(/^---\n/, ''));
        const sections = body.join('---\n').split('## ').slice(1).reduce((acc, section) => {
            const [header, content] = section.split('\n', 1);
            acc[header.trim()] = content;
            return acc;
        }, {});
        return { metadata, sections };
    }
}

// Example usage:
// const parser = new MAMLParser();
// const mamlContent = `---\nmaml_version: "2.0.0"\n---\n## Intent\nTest workflow`;
// console.log(parser.parse(mamlContent));
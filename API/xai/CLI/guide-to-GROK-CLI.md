# MACROSLOW Guide to GROK CLI

## Introduction
The **Grok CLI** (`grok-cli-hurry-mode`) is a powerful command-line interface (CLI) tool developed by xAI, designed to bring conversational AI capabilities powered by Grok models to your terminal. It offers advanced code intelligence, file operations, and automation features, comparable to Claude Code-level functionality. This guide, tailored for the **MACROSLOW** community, provides a comprehensive walkthrough for integrating and customizing Grok CLI for your development workflows, with a focus on practical use cases and server-side customization.

## What is Grok CLI?
Grok CLI is a TypeScript-based, Node.js-powered tool that combines AI-driven code analysis, file manipulation, and task automation in a terminal environment. It supports interactive and headless modes, making it ideal for developers, DevOps engineers, and automation scripts.

### Key Features
- **Conversational AI**: Natural language interaction powered by Grok models (e.g., `grok-3-fast`, `grok-code-fast-1`).
- **Code Intelligence**: Abstract Syntax Tree (AST) parsing, symbol search, dependency analysis, and refactoring for TypeScript, JavaScript, and Python.
- **File Operations**: Multi-file editing, regex search/replace, file tree visualization, and operation history with undo/redo.
- **Bash Integration**: Execute shell commands via natural language (e.g., `git commit`).
- **Extensibility**: Model Context Protocol (MCP) for integrating external tools (e.g., Linear, GitHub).
- **Reliability**: Automated issue detection (`/heal`), standardized imports, and bash fallbacks.
- **Performance**: Optional Morph Fast Apply for high-speed editing (4,500+ tokens/sec, 98% accuracy).

### Use Cases
- **Code Refactoring**: Automate renaming, extracting, or inlining code with previews and rollbacks.
- **Codebase Analysis**: Detect circular dependencies, generate dependency graphs, or analyze code quality.
- **Automation**: Integrate into CI/CD pipelines for linting, testing, or auto-fixing code.
- **Documentation**: Auto-generate JSDoc, READMEs, or changelogs via `/docs` or `/comments`.
- **Project Management**: Use MCP to create Linear issues or manage GitHub tasks.
- **Server-Side Tasks**: Embed in server workflows for AI-driven code reviews or build automation.

## Installation
### Prerequisites
- **Node.js**: Version 18+ (20+ recommended).
- **Grok API Key**: Obtain from [xAI API](https://x.ai/api).
- **Optional**: Morph API key for Fast Apply editing (from Morph Dashboard).

### Quick Install
Run the automated installer (handles edge cases):
```bash
curl -fsSL https://raw.githubusercontent.com/hinetapora/grok-cli-hurry-mode/main/install.sh | bash
```

Or use npm:
```bash
npm install -g grok-cli-hurry-mode
```

Alternative package managers:
```bash
# Yarn
yarn global add grok-cli-hurry-mode
# pnpm
pnpm add -g grok-cli-hurry-mode
# Bun
bun add -g grok-cli-hurry-mode
```

### API Key Setup
Set your Grok API key (choose one method):

1. **Environment Variable**:
   ```bash
   export GROK_API_KEY=your_api_key_here
   ```
2. **.env File**:
   ```bash
   cp .env.example .env
   # Edit .env to add GROK_API_KEY=your_api_key_here
   ```
3. **Command Line**:
   ```bash
   grok --api-key your_api_key_here
   ```
4. **User Settings** (~/.grok/user-settings.json):
   ```json
   {
     "apiKey": "your_api_key_here"
   }
   ```

For Morph Fast Apply (optional):
```bash
export MORPH_API_KEY=your_morph_api_key_here
# Or add to .env
```

## Configuration
Grok CLI supports user-level and project-level configurations for flexibility.

### User-Level Settings (~/.grok/user-settings.json)
Global settings for all projects:
```json
{
  "apiKey": "your_api_key_here",
  "baseURL": "https://api.x.ai/v1",
  "defaultModel": "grok-code-fast-1",
  "models": [
    "grok-code-fast-1",
    "grok-4-latest",
    "grok-3-latest",
    "grok-3-fast",
    "grok-3-mini-fast"
  ]
}
```

### Project-Level Settings (./.grok/settings.json)
Project-specific overrides:
```json
{
  "model": "grok-3-fast",
  "mcpServers": {
    "linear": {
      "name": "linear",
      "transport": "stdio",
      "command": "npx",
      "args": ["@linear/mcp-server"]
    }
  }
}
```

### Custom Instructions (./.grok/GROK.md)
Add project-specific rules:
```markdown
# Custom Instructions for Grok CLI
- Use TypeScript for new code.
- Prefer functional React components with hooks.
- Add JSDoc comments for public functions.
- Follow existing project style.
```

Grok loads these automatically when in the project directory.

## Usage
### Interactive Mode
Start a conversational session:
```bash
grok
# Or specify directory
grok -d /path/to/project
```
Example prompts:
- "Refactor utils.js to use async/await."
- "Show dependency graph for src/."
- "Generate README for this project."

### Headless Mode
For scripting/CI:
```bash
grok --prompt "fix lint errors in src/" --directory /path/to/project
grok -p "run npm test and summarize results" --max-tool-rounds 20
```

### Model Selection
Choose AI model:
```bash
grok --model grok-3-fast
# Or env var
export GROK_MODEL=grok-code-fast-1
grok
```

### Tool Execution Control
Limit tool rounds for performance:
```bash
grok --prompt "list files" --max-tool-rounds 10
# Complex tasks
grok --prompt "refactor entire codebase" --max-tool-rounds 1000
```

## Integration with Your Repository
### Steps
1. **Install Globally**:
   ```bash
   npm install -g grok-cli-hurry-mode
   ```
2. **Set Up Project**:
   - Create `.grok/` folder in repo root.
   - Add `GROK.md` with custom rules (e.g., "Use Next.js patterns").
   - Add `.grok/settings.json` for model/MCP configs.
3. **Use in Workflows**:
   - **Scripts**: Add to `package.json`:
     ```json
     "scripts": {
       "ai-refactor": "grok -p 'refactor src/ to TypeScript' -d . --model grok-3-fast",
       "ai-test": "grok -p 'run tests, fix fails' --max-tool-rounds 20"
     }
     ```
   - **Git Hooks** (via Husky):
     ```bash
     npx husky add .husky/pre-commit "grok -p 'lint-fix staged files' --max-tool-rounds 10"
     ```
   - **CI/CD** (GitHub Actions):
     ```yaml
     name: AI Code Check
     on: [push]
     jobs:
       ai-lint:
         runs-on: ubuntu-latest
         steps:
           - uses: actions/checkout@v3
           - run: npm install -g grok-cli-hurry-mode
           - run: grok -p "run npm test, suggest fixes" >> report.md
             env:
               GROK_API_KEY: ${{ secrets.GROK_API_KEY }}
     ```
4. **MCP Extensions**:
   - Add GitHub/Linear tools:
     ```bash
     grok mcp add github --transport http --url "http://your-mcp-server"
     ```

### Example Commands
- Analyze: `grok -p "find circular deps in src/" -d .`
- Refactor: `grok -p "convert src/utils.js to TypeScript" --max-tool-rounds 50`
- Docs: `grok -p "/docs generate README" -d .`

## Custom Build for Your Server
For server-side use (e.g., API for AI-driven tasks), customize the CLI into a library.

### Steps
1. **Clone Fork**:
   ```bash
   git clone https://github.com/hinetapora/grok-cli-hurry-mode
   cd grok-cli-hurry-mode
   ```
2. **Modify Structure**:
   - **Extract Core**: In `src/agent.ts`, export main logic:
     ```typescript
     export async function createGrokAgent({ apiKey, baseUrl, model = 'grok-code-fast-1' }) {
       // Initialize Grok client (OpenAI compat)
       const client = new OpenAI({ apiKey, baseUrl });
       // Tool setup (edit_file, bash, etc.)
       const tools = [...]; // From src/tools/
       return {
         run: async (prompt, { cwd, maxToolRounds }) => {
           // Process prompt, execute tools
           return { text: 'output', fileOps: [] };
         }
       };
     }
     ```
   - **Disable UI**: In `src/ui/`, comment `render` calls (Ink.js). Output JSON instead:
     ```typescript
     // Before: render(<Spinner />);
     // After: console.log(JSON.stringify({ status: 'processing' }));
     ```
   - **Tools**: Keep `edit_file` (Morph), `str_replace_editor`, `bash`. Add custom (e.g., DB query).
   - **Config**: Hardcode API key/base URL in server env; ignore `.grok/`.
3. **Build**:
   ```bash
   npm install
   npm run build
   ```
4. **Integrate in Server** (Express example):
   ```typescript
   // server.ts
   import express from 'express';
   import { createGrokAgent } from './dist/agent';

   const app = express();
   app.use(express.json());

   app.post('/ai-task', async (req, res) => {
     const { prompt, dir, maxRounds = 50, model = 'grok-3-fast' } = req.body;
     try {
       const agent = createGrokAgent({
         apiKey: process.env.GROK_API_KEY,
         baseUrl: 'https://api.x.ai/v1'
       });
       const result = await agent.run(prompt, { cwd: dir, maxToolRounds: maxRounds, model });
       res.json({ output: result.text, changes: result.fileOps });
     } catch (e) {
       res.status(500).json({ error: e.message });
     }
   });

   app.listen(3000, () => console.log('Server on port 3000'));
   ```
5. **Run**:
   ```bash
   node server.ts
   ```
   Test endpoint:
   ```bash
   curl -X POST http://localhost:3000/ai-task -H "Content-Type: application/json" \
     -d '{"prompt":"refactor /repo/file.js","dir":"/path/to/repo","maxRounds":50}'
   ```
6. **Deploy** (Docker):
   ```dockerfile
   # Dockerfile
   FROM node:20
   WORKDIR /app
   COPY . .
   RUN npm install && npm run build
   CMD ["node", "server.js"]
   ```
   ```bash
   docker build -t grok-server .
   docker run -p 3000:3000 -e GROK_API_KEY=your_key grok-server
   ```

### Customization Tips
- **Reliability**: Enable `/heal` for auto-fixing tool issues.
- **Morph**: Use `edit_file` tool for complex edits if Morph key is set.
- **Limits**: Set `max-tool-rounds` low (10-50) for simple tasks to avoid hangs.
- **Logging**: Add Winston for server logs:
  ```typescript
  import winston from 'winston';
  const logger = winston.createLogger({ transports: [new winston.transports.File({ filename: 'grok.log' })] });
  ```

## Troubleshooting
- **Tool Errors** (e.g., `fs.readFile is not a function`):
  - CLI falls back to bash; functionality preserved.
  - Fix: Ensure Node.js 18+ and check file paths.
- **API Key Issues**: Verify key at [x.ai/api](https://x.ai/api).
- **Timeouts**: Increase `max-tool-rounds` for complex tasks or reduce for CI.
- **MCP Failures**: Test servers with `grok mcp test server-name`.

## Example Use Cases for MACROSLOW
- **Refactor Legacy Code**:
  ```bash
  grok -p "convert src/legacy.js to TypeScript with types" -d .
  ```
- **CI Automation**:
  ```yaml
  - run: grok -p "fix lint errors, commit changes" --max-tool-rounds 20
  ```
- **Docs Generation**:
  ```bash
  grok -p "/docs generate README and changelog" -d .
  ```
- **Server-Side PR Reviews**:
  ```bash
  curl -X POST http://your-server/ai-task -d '{"prompt":"review PR diff in src/","dir":"./repo"}'
  ```

## Resources
- **NPM**: [grok-cli-hurry-mode](https://www.npmjs.com/package/grok-cli-hurry-mode)
- **GitHub**: [hinetapora/grok-cli-hurry-mode](https://github.com/hinetapora/grok-cli-hurry-mode)
- **xAI Discord**: [xAI Community](https://discord.com/invite/xai)
- **API Docs**: [xAI API](https://x.ai/api)

## License
MIT

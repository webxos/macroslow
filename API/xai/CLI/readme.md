# Custom Instructions for Grok CLI
- Use TypeScript for all new code files.
- Prefer functional components for React.
- Add JSDoc comments for public functions and interfaces.
- Follow existing code style (e.g., 2-space indentation).
- Prioritize `grok-3-fast` model for performance.

### Quick Setup Steps
1. **Create Repo**: Initialize a new GitHub repository or use an existing one.
2. **Add Files**: Create the above files in the specified paths (e.g., `.grok/GROK.md`, `src/index.ts`).
3. **Install Dependencies**:
   ```bash
   npm install
   ```
4. **Set API Key**:
   - Copy `.env.example` to `.env` and add your Grok API key from [xAI API](https://x.ai/api).
   - Alternatively, set `GROK_API_KEY` environment variable:
     ```bash
     export GROK_API_KEY=your_api_key_here
     ```
5. **Test CLI**:
   ```bash
   grok -p "run src/index.ts" -d .
   ```
6. **Push to GitHub**: Commit and push files. Ensure `GROK_API_KEY` is added as a GitHub Actions secret for CI.
7. **Run CI**: Verify the `ai-check.yml` workflow runs on push, executing `ai-lint` and `ai-test` scripts.

These files provide a minimal setup for using Grok CLI in your repo, with scripts for linting, testing, and documentation, plus CI integration. You can expand by adding more scripts or MCP tools (e.g., for GitHub or Linear).

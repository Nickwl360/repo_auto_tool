# Setting Up repo-improver for Development

## Prerequisites

1. **Python 3.11+**
   ```bash
   python --version  # Should be 3.11 or higher
   ```

2. **Claude Code CLI**
   ```bash
   npm install -g @anthropic-ai/claude-code
   claude --version
   
   # Make sure you're authenticated
   claude auth login
   ```

3. **Git** (recommended)
   ```bash
   git --version
   ```

---

## Build & Install Locally

### Option A: From the tarball (what you have now)

```bash
# 1. Extract
tar -xzf repo-improver.tar.gz
cd repo-improver

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install in editable/development mode
pip install -e ".[dev]"

# 4. Verify installation
repo-improver --help
```

### Option B: After pushing to GitHub

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/repo-improver.git
cd repo-improver

# Create venv & install
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

---

## Verify Everything Works

```bash
# Check CLI is installed
repo-improver --help

# Check Claude Code is accessible
claude --version

# Run a quick analysis (no changes)
repo-improver . "Improve code quality" --analyze-only

# Run linting on the package itself
ruff check src/

# Run any existing tests
pytest  # (will fail initially - no tests yet!)
```

---

## First Improvement Run

### Manual single prompt (test the waters)

```bash
# Just analyze, see what Claude thinks
repo-improver . "Add error handling" --analyze-only

# Do one iteration, see what happens
repo-improver . "Add a custom ImproverError exception class" --max-iterations 1
```

### Let it run for a few iterations

```bash
# This will:
# 1. Create branch: repo-improver/auto
# 2. Make improvements
# 3. Run ruff (linting) after each change
# 4. Commit successes, rollback failures
# 5. Stop after 5 iterations or goal complete

repo-improver . "Add custom exception classes for all error types" \
  --max-iterations 5 \
  --no-tests  # Skip pytest since we have no tests yet
```

### Check results

```bash
# See what changed
git log --oneline repo-improver/auto
git diff main..repo-improver/auto

# If you like it, merge
git checkout main
git merge repo-improver/auto
```

---

## Development Workflow

```bash
# 1. Make manual changes to the code
vim src/repo_improver/improver.py

# 2. Test your changes
ruff check src/
python -c "from repo_improver import RepoImprover; print('OK')"

# 3. Or let repo-improver improve itself!
repo-improver . "Your improvement goal" --max-iterations 10

# 4. Review changes on the auto branch
git diff main..repo-improver/auto

# 5. Merge if good
git checkout main
git merge repo-improver/auto
git push
```

---

## Troubleshooting

### "Claude Code CLI not found"
```bash
npm install -g @anthropic-ai/claude-code
# Make sure npm bin is in PATH
export PATH="$PATH:$(npm bin -g)"
```

### "No module named repo_improver"
```bash
# Make sure you installed in editable mode
pip install -e ".[dev]"
# And that your venv is activated
source venv/bin/activate
```

### "Permission denied" on bootstrap.sh
```bash
chmod +x bootstrap.sh
```

### Claude CLI hangs or times out
- Check your API key: `claude auth status`
- Try a simple test: `claude -p "Hello" --output-format json`
- Increase timeout in config if needed

---

## Project Structure

```
repo-improver/
├── src/repo_improver/
│   ├── __init__.py         # Package exports
│   ├── cli.py              # Entry point (repo-improver command)
│   ├── config.py           # Configuration dataclass
│   ├── improver.py         # Main orchestration loop ← THE BRAIN
│   ├── claude_interface.py # Claude Code CLI wrapper
│   ├── validators.py       # Test/lint validation
│   ├── git_helper.py       # Git operations
│   └── state.py            # Persistent state
├── pyproject.toml          # Package config & dependencies
├── TODO.md                 # Feature roadmap
├── README.md               # User docs
├── SETUP.md                # This file
└── bootstrap.sh            # Quick start script
```

---

## Next Steps

1. Push to GitHub
2. Run first improvement: error handling
3. Add tests
4. Check off items in TODO.md

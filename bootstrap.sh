#!/bin/bash
# bootstrap.sh - Apply repo-improver to itself!
#
# This script runs the first manual improvement iteration on the repo-improver
# codebase itself. Meta!

set -e

# First, install the package in development mode
echo "📦 Installing repo-improver in dev mode..."
pip install -e ".[dev]"

# Verify Claude Code is available
echo "🔍 Checking for Claude Code CLI..."
if ! command -v claude &> /dev/null; then
    echo "❌ Claude Code CLI not found. Install with:"
    echo "   npm install -g @anthropic-ai/claude-code"
    exit 1
fi

echo "✅ Claude Code CLI found: $(claude --version)"

# First improvement goal - you can customize this!
GOAL="${1:-Add comprehensive unit tests for all modules with at least 80% coverage}"

echo ""
echo "🚀 Applying repo-improver to itself with goal:"
echo "   \"$GOAL\""
echo ""

# Run it!
repo-improver . "$GOAL" --max-iterations 10

echo ""
echo "🎉 Done! Check the repo-improver/auto branch for changes."
echo "   git diff main..repo-improver/auto"

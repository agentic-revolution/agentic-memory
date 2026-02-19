#!/bin/bash
set -e

echo "=== AgenticMemory + Claude Code Integration Test ==="

# Build release binary
cargo build --release

# Create test memory file
TEST_MEMORY="/tmp/claude_test_$(date +%s).amem"

# Configure Claude Desktop (macOS example)
CONFIG_FILE="$HOME/Library/Application Support/Claude/claude_desktop_config.json"

# Backup existing config
if [ -f "$CONFIG_FILE" ]; then
    cp "$CONFIG_FILE" "$CONFIG_FILE.bak"
fi

# Write test config
cat > "$CONFIG_FILE" << EOF
{
  "mcpServers": {
    "agentic-memory": {
      "command": "$(pwd)/target/release/agentic-memory-mcp",
      "args": ["serve", "--memory", "$TEST_MEMORY"]
    }
  }
}
EOF

echo "Config written to: $CONFIG_FILE"
echo "Memory file: $TEST_MEMORY"
echo ""
echo "Now:"
echo "1. Restart Claude Desktop"
echo "2. Ask Claude to use the memory tools"
echo "3. Example prompts:"
echo "   - 'Remember that I prefer Rust over Python'"
echo "   - 'What do you remember about my preferences?'"
echo "   - 'Why did you recommend X last time?'"
echo ""
echo "Press Enter when done testing..."
read

# Verify memory was created
if [ -f "$TEST_MEMORY" ]; then
    echo "✓ Memory file created"
    ls -la "$TEST_MEMORY"
else
    echo "✗ Memory file not created - test failed"
    exit 1
fi

# Restore config
if [ -f "$CONFIG_FILE.bak" ]; then
    mv "$CONFIG_FILE.bak" "$CONFIG_FILE"
fi

echo "=== Test Complete ==="

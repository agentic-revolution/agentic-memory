# AgenticMemory MCP Server

> Universal LLM access to persistent graph memory via the Model Context Protocol.

AgenticMemory MCP Server bridges **any MCP-compatible LLM client** (Claude, GPT, Gemini, Ollama, etc.) to the AgenticMemory persistent binary graph memory system. One server, universal access.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     LLM CLIENT                           │
│       (Claude, GPT, Gemini, Ollama, etc.)               │
└─────────────────────┬───────────────────────────────────┘
                      │ MCP Protocol (JSON-RPC 2.0)
                      │ (stdio / SSE)
┌─────────────────────▼───────────────────────────────────┐
│                AGENTIC-MEMORY-MCP                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │  TOOLS   │  │RESOURCES │  │ PROMPTS  │              │
│  │ 12 tools │  │ 6 URIs   │  │ 4 tmpls  │              │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘              │
│       └──────────────┼─────────────┘                    │
│              SESSION MANAGER                             │
│              AGENTIC-MEMORY CORE                         │
└─────────────────────────────────────────────────────────┘
                      │
                      ▼
               brain.amem (binary graph file)
```

## Quick Start

### Build

```bash
cargo build --release
```

### Run (stdio transport — default)

```bash
agentic-memory-mcp serve --memory /path/to/brain.amem
```

### Run (SSE transport)

```bash
cargo build --release --features sse
agentic-memory-mcp serve-http --memory /path/to/brain.amem --addr 127.0.0.1:3000
```

## Configuration with MCP Clients

### Claude Desktop

Add to your Claude Desktop MCP config (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "agentic-memory": {
      "command": "agentic-memory-mcp",
      "args": ["serve", "--memory", "/path/to/brain.amem"]
    }
  }
}
```

### VS Code / Cursor

Add to `.vscode/settings.json`:

```json
{
  "mcp.servers": {
    "agentic-memory": {
      "command": "agentic-memory-mcp",
      "args": ["serve", "--memory", "${workspaceFolder}/.memory/project.amem"]
    }
  }
}
```

## Tools (12)

| Tool | Purpose |
|------|---------|
| `memory_add` | Add cognitive event to graph |
| `memory_query` | Pattern query for matching nodes |
| `memory_traverse` | Walk graph following edges |
| `memory_correct` | Record correction to past belief |
| `memory_resolve` | Follow supersedes chain |
| `memory_context` | Get subgraph around node |
| `memory_similar` | Similarity search |
| `memory_causal` | Impact analysis |
| `memory_temporal` | Compare across time |
| `memory_stats` | Graph statistics |
| `session_start` | Begin new session |
| `session_end` | End session, create episode |

## Resources (6)

| URI Pattern | Returns |
|-------------|---------|
| `amem://node/{id}` | Single node with edges |
| `amem://session/{id}` | All nodes from session |
| `amem://types/{type}` | All nodes of type |
| `amem://graph/stats` | Graph statistics |
| `amem://graph/recent` | Recent nodes |
| `amem://graph/important` | High decay score nodes |

## Prompts (4)

| Prompt | Purpose | Required Args |
|--------|---------|---------------|
| `remember` | Guide for storing new info | `information` |
| `reflect` | Guide for understanding past decisions | `topic` |
| `correct` | Guide for updating beliefs | `old_belief`, `new_information` |
| `summarize` | Guide for session summary | (none) |

## Event Types

- **Fact** — Declarative knowledge
- **Decision** — Choices made with reasoning
- **Inference** — Derived conclusions
- **Correction** — Updates to prior beliefs
- **Skill** — Procedural knowledge
- **Episode** — Session summaries

## Edge Types

- **CausedBy** — Causal relationship
- **Supports** — Supporting evidence
- **Contradicts** — Conflicting information
- **Supersedes** — Correction chain
- **RelatedTo** — General association
- **PartOf** — Hierarchical containment
- **TemporalNext** — Temporal ordering

## CLI Commands

```bash
# Start server (stdio)
agentic-memory-mcp serve --memory brain.amem

# Start server (SSE, requires --features sse)
agentic-memory-mcp serve-http --addr 127.0.0.1:3000

# Validate a memory file
agentic-memory-mcp validate --memory brain.amem

# Print server info as JSON
agentic-memory-mcp info
```

## Development

```bash
# Run tests
cargo test

# Run clippy
cargo clippy

# Format code
cargo fmt

# Build release
cargo build --release
```

## Protocol

This server implements MCP (Model Context Protocol) spec version **2024-11-05** over JSON-RPC 2.0. Supported transports:

- **stdio** — Newline-delimited JSON over stdin/stdout (default)
- **SSE** — Server-Sent Events over HTTP (feature flag `sse`)

## License

MIT

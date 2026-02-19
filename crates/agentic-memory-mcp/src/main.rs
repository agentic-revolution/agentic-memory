//! AgenticMemory MCP Server — entry point.

use std::sync::Arc;
use tokio::sync::Mutex;

use clap::{Parser, Subcommand};

use agentic_memory_mcp::config::resolve_memory_path;
use agentic_memory_mcp::protocol::ProtocolHandler;
use agentic_memory_mcp::session::SessionManager;
use agentic_memory_mcp::tools::ToolRegistry;
use agentic_memory_mcp::transport::StdioTransport;

#[derive(Parser)]
#[command(
    name = "agentic-memory-mcp",
    about = "MCP server for AgenticMemory — universal LLM access to persistent graph memory",
    version
)]
struct Cli {
    /// Path to .amem memory file.
    #[arg(short, long)]
    memory: Option<String>,

    /// Configuration file path.
    #[arg(short, long)]
    config: Option<String>,

    /// Log level (trace, debug, info, warn, error).
    #[arg(long, default_value = "info")]
    log_level: String,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand)]
enum Commands {
    /// Start MCP server over stdio (default).
    Serve {
        /// Path to .amem memory file.
        #[arg(short, long)]
        memory: Option<String>,

        /// Configuration file path.
        #[arg(short, long)]
        config: Option<String>,

        /// Log level (trace, debug, info, warn, error).
        #[arg(long)]
        log_level: Option<String>,
    },

    /// Start MCP server over SSE (HTTP).
    #[cfg(feature = "sse")]
    ServeHttp {
        /// Listen address.
        #[arg(long, default_value = "127.0.0.1:3000")]
        addr: String,

        /// Path to .amem memory file.
        #[arg(short, long)]
        memory: Option<String>,

        /// Configuration file path.
        #[arg(short, long)]
        config: Option<String>,

        /// Log level (trace, debug, info, warn, error).
        #[arg(long)]
        log_level: Option<String>,
    },

    /// Validate a memory file.
    Validate,

    /// Print server capabilities as JSON.
    Info,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(&cli.log_level));

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .init();

    match cli.command.unwrap_or(Commands::Serve {
        memory: None,
        config: None,
        log_level: None,
    }) {
        Commands::Serve {
            memory,
            config: _,
            log_level: _,
        } => {
            let effective_memory = memory.or(cli.memory);
            let memory_path = resolve_memory_path(effective_memory.as_deref());
            let session = SessionManager::open(&memory_path)?;
            let session = Arc::new(Mutex::new(session));
            let handler = ProtocolHandler::new(session);
            let transport = StdioTransport::new(handler);
            transport.run().await?;
        }

        #[cfg(feature = "sse")]
        Commands::ServeHttp {
            addr,
            memory,
            config: _,
            log_level: _,
        } => {
            let effective_memory = memory.or(cli.memory);
            let memory_path = resolve_memory_path(effective_memory.as_deref());
            let session = SessionManager::open(&memory_path)?;
            let session = Arc::new(Mutex::new(session));
            let handler = ProtocolHandler::new(session);
            let transport = agentic_memory_mcp::transport::SseTransport::new(handler);
            transport.run(&addr).await?;
        }

        Commands::Validate => {
            let memory_path = resolve_memory_path(cli.memory.as_deref());
            match SessionManager::open(&memory_path) {
                Ok(session) => {
                    let graph = session.graph();
                    println!("Valid memory file: {memory_path}");
                    println!("  Nodes: {}", graph.node_count());
                    println!("  Edges: {}", graph.edge_count());
                    println!("  Dimension: {}", graph.dimension());
                    println!("  Sessions: {}", graph.session_index().session_count());
                }
                Err(e) => {
                    eprintln!("Invalid memory file: {e}");
                    std::process::exit(1);
                }
            }
        }

        Commands::Info => {
            let capabilities = agentic_memory_mcp::types::InitializeResult::default_result();
            let tools = ToolRegistry::list_tools();
            let info = serde_json::json!({
                "server": capabilities.server_info,
                "protocol_version": capabilities.protocol_version,
                "capabilities": capabilities.capabilities,
                "tools": tools.iter().map(|t| &t.name).collect::<Vec<_>>(),
                "tool_count": tools.len(),
            });
            println!("{}", serde_json::to_string_pretty(&info)?);
        }
    }

    Ok(())
}

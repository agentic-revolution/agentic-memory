//! Example: Custom MCP client in Rust.
//!
//! Demonstrates how to connect to the agentic-memory-mcp server
//! over stdio and perform basic operations.
//!
//! Usage:
//!   cargo run --example custom_client
//!
//! Note: This example requires the server binary to be built first.
//! It spawns the server as a subprocess and communicates over stdio.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

use serde_json::{json, Value};

/// Simple MCP client that communicates with the server over stdio.
struct SimpleClient {
    stdin: std::process::ChildStdin,
    reader: BufReader<std::process::ChildStdout>,
    next_id: i64,
}

impl SimpleClient {
    /// Send a JSON-RPC request and read the response.
    fn request(&mut self, method: &str, params: Option<Value>) -> Value {
        self.next_id += 1;
        let request = json!({
            "jsonrpc": "2.0",
            "id": self.next_id,
            "method": method,
            "params": params
        });

        let line = serde_json::to_string(&request).unwrap() + "\n";
        self.stdin.write_all(line.as_bytes()).unwrap();
        self.stdin.flush().unwrap();

        let mut response_line = String::new();
        self.reader.read_line(&mut response_line).unwrap();
        serde_json::from_str(&response_line).unwrap()
    }

    /// Send a JSON-RPC notification (no response expected).
    fn notify(&mut self, method: &str, params: Option<Value>) {
        let notification = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params
        });
        let line = serde_json::to_string(&notification).unwrap() + "\n";
        self.stdin.write_all(line.as_bytes()).unwrap();
        self.stdin.flush().unwrap();
    }
}

fn main() {
    println!("=== AgenticMemory MCP Client Example ===\n");

    // Find the server binary
    let server_path = std::env::current_dir()
        .unwrap()
        .join("target/debug/agentic-memory-mcp");

    if !server_path.exists() {
        eprintln!("Server binary not found. Run `cargo build` first.");
        std::process::exit(1);
    }

    // Create a temporary memory file
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let memory_path = temp_dir.path().join("example.amem");

    // Start the server as a subprocess
    let mut child = Command::new(&server_path)
        .args(["serve", "--memory", &memory_path.display().to_string()])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .expect("Failed to start server");

    let stdin = child.stdin.take().expect("Failed to open stdin");
    let stdout = child.stdout.take().expect("Failed to open stdout");

    let mut client = SimpleClient {
        stdin,
        reader: BufReader::new(stdout),
        next_id: 0,
    };

    // 1. Initialize
    println!("1. Initializing...");
    let init_response = client.request(
        "initialize",
        Some(json!({
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {
                "name": "example-client",
                "version": "1.0.0"
            }
        })),
    );
    println!(
        "   Server: {} v{}",
        init_response["result"]["serverInfo"]["name"],
        init_response["result"]["serverInfo"]["version"]
    );

    // Send initialized notification
    client.notify("initialized", None);

    // 2. List tools
    println!("\n2. Listing tools...");
    let tools_response = client.request("tools/list", None);
    let tools = tools_response["result"]["tools"].as_array().unwrap();
    println!("   Available tools ({}):", tools.len());
    for tool in tools {
        println!("   - {}", tool["name"].as_str().unwrap());
    }

    // 3. Add a memory
    println!("\n3. Adding a memory...");
    let add_response = client.request(
        "tools/call",
        Some(json!({
            "name": "memory_add",
            "arguments": {
                "event_type": "fact",
                "content": "Rust is a systems programming language",
                "confidence": 0.95
            }
        })),
    );
    let add_text = add_response["result"]["content"][0]["text"]
        .as_str()
        .unwrap();
    let add_parsed: Value = serde_json::from_str(add_text).unwrap();
    println!("   Added node #{}", add_parsed["node_id"]);

    // 4. Query memories
    println!("\n4. Querying memories...");
    let query_response = client.request(
        "tools/call",
        Some(json!({
            "name": "memory_query",
            "arguments": { "event_types": ["fact"] }
        })),
    );
    let query_text = query_response["result"]["content"][0]["text"]
        .as_str()
        .unwrap();
    let query_parsed: Value = serde_json::from_str(query_text).unwrap();
    println!("   Found {} memories", query_parsed["count"]);

    // 5. Read graph stats resource
    println!("\n5. Reading graph statistics...");
    let stats_response = client.request(
        "resources/read",
        Some(json!({ "uri": "amem://graph/stats" })),
    );
    let stats_text = stats_response["result"]["contents"][0]["text"]
        .as_str()
        .unwrap();
    let stats: Value = serde_json::from_str(stats_text).unwrap();
    println!(
        "   Nodes: {}, Edges: {}",
        stats["node_count"], stats["edge_count"]
    );

    // 6. Shutdown
    println!("\n6. Shutting down...");
    let _shutdown = client.request("shutdown", None);
    println!("   Server shutdown complete.");

    // Clean up
    drop(client);
    let _ = child.wait();

    println!("\n=== Example complete ===");
}

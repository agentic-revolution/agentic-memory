//! Phase 3: Prompt expansion tests.

mod common;

use serde_json::json;

use agentic_memory_mcp::prompts::PromptRegistry;
use agentic_memory_mcp::types::ToolContent;

use common::fixtures::create_test_session;

#[tokio::test]
async fn test_prompt_list() {
    let prompts = PromptRegistry::list_prompts();
    assert!(prompts.len() >= 4);

    let names: Vec<&str> = prompts.iter().map(|p| p.name.as_str()).collect();
    assert!(names.contains(&"remember"));
    assert!(names.contains(&"reflect"));
    assert!(names.contains(&"correct"));
    assert!(names.contains(&"summarize"));
}

#[tokio::test]
async fn test_prompt_remember() {
    let session = create_test_session();
    let result = PromptRegistry::get(
        "remember",
        Some(json!({
            "information": "User loves cats",
            "context": "Mentioned during onboarding"
        })),
        &session,
    )
    .await
    .unwrap();

    assert!(!result.messages.is_empty());
    let text = match &result.messages[0].content {
        ToolContent::Text { text } => text,
        _ => panic!("Expected text"),
    };

    assert!(text.contains("User loves cats"));
    assert!(text.contains("Mentioned during onboarding"));
    assert!(text.contains("memory_add"));
}

#[tokio::test]
async fn test_prompt_reflect() {
    let session = create_test_session();
    let result = PromptRegistry::get(
        "reflect",
        Some(json!({"topic": "database choice"})),
        &session,
    )
    .await
    .unwrap();

    let text = match &result.messages[0].content {
        ToolContent::Text { text } => text,
        _ => panic!("Expected text"),
    };

    assert!(text.contains("database choice"));
    assert!(text.contains("memory_traverse"));
}

#[tokio::test]
async fn test_prompt_correct() {
    let session = create_test_session();
    let result = PromptRegistry::get(
        "correct",
        Some(json!({
            "old_belief": "Python is fastest",
            "new_information": "Rust is fastest",
            "reason": "Benchmark results"
        })),
        &session,
    )
    .await
    .unwrap();

    let text = match &result.messages[0].content {
        ToolContent::Text { text } => text,
        _ => panic!("Expected text"),
    };

    assert!(text.contains("Python is fastest"));
    assert!(text.contains("Rust is fastest"));
    assert!(text.contains("memory_correct"));
}

#[tokio::test]
async fn test_prompt_summarize() {
    let session = create_test_session();
    let result = PromptRegistry::get("summarize", Some(json!({})), &session)
        .await
        .unwrap();

    let text = match &result.messages[0].content {
        ToolContent::Text { text } => text,
        _ => panic!("Expected text"),
    };

    assert!(text.contains("session_end"));
}

#[tokio::test]
async fn test_prompt_not_found() {
    let session = create_test_session();
    let result = PromptRegistry::get("nonexistent", None, &session).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_prompt_remember_missing_arg() {
    let session = create_test_session();
    let result = PromptRegistry::get("remember", Some(json!({})), &session).await;
    assert!(result.is_err());
}

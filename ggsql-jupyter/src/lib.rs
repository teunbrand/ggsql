//! ggsql-jupyter library
//!
//! This module exposes the internal components for testing.

pub mod message;
pub mod executor;
pub mod display;

// Re-export commonly used types
pub use message::{ConnectionInfo, JupyterMessage, MessageHeader};
pub use executor::{QueryExecutor, ExecutionResult};
pub use display::format_display_data;

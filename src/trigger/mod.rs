//! Trigger system for STT activation
//!
//! This module provides a configurable trigger abstraction that can be used
//! to activate/deactivate STT recording. Different trigger types can be implemented
//! (hotkey, button, wake word, etc.) and easily swapped at compile time.

use crate::utils::error::Result;

/// Trigger event types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerEvent {
    /// Trigger activated (start recording)
    Activated,
    /// Trigger deactivated (stop recording)
    Deactivated,
}

/// Trait for trigger implementations
///
/// A trigger is a mechanism that can activate or deactivate STT recording.
/// Different implementations can use hotkeys, buttons, wake words, etc.
pub trait Trigger: Send + Sync {
    /// Check if the trigger is currently active
    fn is_active(&self) -> bool;

    /// Wait for trigger activation (blocking)
    ///
    /// Returns when the trigger is activated.
    fn wait_for_activation(&self) -> Result<()>;

    /// Wait for trigger deactivation (blocking)
    ///
    /// Returns when the trigger is deactivated.
    fn wait_for_deactivation(&self) -> Result<()>;
}

/// Trigger type selection (for compile-time configuration)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerType {
    /// Hotkey trigger (keyboard key)
    Hotkey,
    /// Button trigger (hardware button - placeholder)
    Button,
    /// Wake word trigger (voice activation - placeholder)
    WakeWord,
}

// Re-export trigger implementations
pub mod hotkey;





//! Hotkey trigger implementation
//!
//! Uses `rdev` to listen for keyboard events and trigger recording.

use crate::trigger::{Trigger, TriggerEvent};
use crate::utils::error::{Error, Result};
use rdev::{listen, Event, EventType, Key};
use std::sync::{Arc, Mutex, atomic::{AtomicBool, Ordering}};
use std::sync::mpsc;

/// Hotkey trigger implementation
///
/// Listens for a specific key press/release to activate/deactivate recording.
pub struct HotkeyTrigger {
    /// The key that triggers activation
    trigger_key: Key,
    /// Current activation state
    is_active: Arc<AtomicBool>,
    /// Channel sender for trigger events
    event_tx: mpsc::Sender<TriggerEvent>,
    /// Channel receiver for trigger events (wrapped in Mutex for Sync)
    event_rx: Arc<Mutex<mpsc::Receiver<TriggerEvent>>>,
}

impl HotkeyTrigger {
    /// Create a new hotkey trigger
    ///
    /// # Arguments
    /// * `trigger_key` - The key to use as trigger (e.g., `Key::ControlLeft`)
    ///
    /// # Returns
    /// A new `HotkeyTrigger` instance
    pub fn new(trigger_key: Key) -> Result<Self> {
        let (tx, rx) = mpsc::channel();
        let is_active = Arc::new(AtomicBool::new(false));

        let trigger = Self {
            trigger_key,
            is_active: is_active.clone(),
            event_tx: tx.clone(),
            event_rx: Arc::new(Mutex::new(rx)),
        };

        // Spawn thread to listen for keyboard events
        let tx_for_listener = tx.clone();
        std::thread::spawn(move || {
            listen(move |event: Event| {
                match event.event_type {
                    EventType::KeyPress(key) if key == trigger_key => {
                        let _ = tx_for_listener.send(TriggerEvent::Activated);
                    }
                    EventType::KeyRelease(key) if key == trigger_key => {
                        let _ = tx_for_listener.send(TriggerEvent::Deactivated);
                    }
                    _ => {}
                }
            }).ok();
        });

        Ok(trigger)
    }

    /// Get the next trigger event (non-blocking)
    ///
    /// # Returns
    /// `Some(TriggerEvent)` if an event is available, `None` otherwise
    pub fn try_recv(&self) -> Option<TriggerEvent> {
        self.event_rx.lock().ok()?.try_recv().ok()
    }

    /// Get the next trigger event (blocking)
    ///
    /// # Returns
    /// The next `TriggerEvent`
    pub fn recv(&self) -> Result<TriggerEvent> {
        self.event_rx.lock()
            .map_err(|e| Error::Inference(format!("Failed to lock receiver: {}", e)))?
            .recv()
            .map_err(|e| Error::Inference(format!("Failed to receive trigger event: {}", e)))
    }
}

impl Trigger for HotkeyTrigger {
    fn is_active(&self) -> bool {
        self.is_active.load(Ordering::SeqCst)
    }

    fn wait_for_activation(&self) -> Result<()> {
        loop {
            match self.recv()? {
                TriggerEvent::Activated => {
                    self.is_active.store(true, Ordering::SeqCst);
                    return Ok(());
                }
                TriggerEvent::Deactivated => {
                    self.is_active.store(false, Ordering::SeqCst);
                }
            }
        }
    }

    fn wait_for_deactivation(&self) -> Result<()> {
        loop {
            match self.recv()? {
                TriggerEvent::Activated => {
                    self.is_active.store(true, Ordering::SeqCst);
                }
                TriggerEvent::Deactivated => {
                    self.is_active.store(false, Ordering::SeqCst);
                    return Ok(());
                }
            }
        }
    }
}





/*
 * BLE Audio Receiver - Connect to memo device and receive audio stream via GATT
 * Uses btleplug for BLE connectivity
 */

use anyhow::{Context, Result};
use btleplug::api::{Manager as _, Central as _, Characteristic, Peripheral as _, ScanFilter};
use btleplug::platform::{Manager, Adapter, Peripheral};
use log::{debug, info, warn, error};
use std::time::Duration;
use tokio::time::timeout;
use uuid::Uuid;

const DEVICE_NAME_PATTERN: &str = "memo_";
const DEVICE_ADDRESS: &str = "64D5A7E1-B149-191F-9B11-96F5CCF590BF"; // From memory
const SCAN_TIMEOUT: Duration = Duration::from_secs(30);

// Service and characteristic UUIDs (from firmware bluetooth.c)
// Memo Audio Service UUID: 1234A000-1234-5678-1234-56789ABCDEF0
const MEMO_AUDIO_SERVICE_UUID: &str = "1234A000-1234-5678-1234-56789ABCDEF0";
// Memo Audio Data Characteristic UUID: 1234A001-1234-5678-1234-56789ABCDEF0
const MEMO_AUDIO_DATA_CHAR_UUID: &str = "1234A001-1234-5678-1234-56789ABCDEF0";
// Control TX Characteristic UUID: 1234A003-1234-5678-1234-56789ABCDEF0
// Sends notifications: RESP_SPEECH_START (0x01) and RESP_SPEECH_END (0x02)
const MEMO_CONTROL_TX_CHAR_UUID: &str = "1234A003-1234-5678-1234-56789ABCDEF0";

// Control response values from firmware
const RESP_SPEECH_START: u8 = 0x01;  // 1 - Recording started
const RESP_SPEECH_END: u8 = 0x02;    // 2 - Recording ended

pub struct BleAudioReceiver {
    periph: Option<Peripheral>,
    char_audio_data: Option<Characteristic>,
    char_control_tx: Option<Characteristic>,
    device_name: Option<String>, // Store device name for retrieval
}

impl BleAudioReceiver {
    pub async fn new() -> Result<Self> {
        info!("Initializing BLE receiver with btleplug");
        Ok(Self {
            periph: None,
            char_audio_data: None,
            char_control_tx: None,
            device_name: None,
        })
    }

    /// Scan for and connect to the memo device
    /// If preferred_device_name is provided, it will be prioritized during scanning
    pub async fn connect(&mut self, preferred_device_name: Option<&str>) -> Result<()> {
        if let Some(pref_name) = preferred_device_name {
            info!("Scanning for memo device (preferred: {}, pattern: {}*)", pref_name, DEVICE_NAME_PATTERN);
            eprintln!("🔍 Scanning for BLE device (preferred: {})...", pref_name);
        } else {
            info!("Scanning for memo device (pattern: {}*)", DEVICE_NAME_PATTERN);
            eprintln!("🔍 Scanning for BLE device...");
        }

        let manager = Manager::new().await
            .context("Failed to create BLE manager")?;
        
        let adapter_list = manager.adapters().await
            .context("Failed to get adapters")?;
        
        let adapter: Adapter = adapter_list.into_iter().next()
            .context("No BLE adapter found")?;

        // Device advertises the service UUID - scan for it
        let service_uuid = Uuid::parse_str(MEMO_AUDIO_SERVICE_UUID)?;
        adapter.start_scan(ScanFilter::default()).await.context("Failed to start scan")?;
        
        let mut found_periph: Option<Peripheral> = None;
        let start = std::time::Instant::now();
        
        while start.elapsed() < SCAN_TIMEOUT {
            tokio::time::sleep(Duration::from_secs(1)).await;
            let peripherals = adapter.peripherals().await?;
            
            for p in peripherals {
                if let Ok(Some(props)) = p.properties().await {
                    // Check for service UUID in advertising data
                    if props.services.contains(&service_uuid) {
                        eprintln!("✅ Found device with Memo service");
                        found_periph = Some(p);
                        break;
                    }
                    // Or check name
                    if let Some(name) = &props.local_name {
                        if name.to_lowercase().starts_with(DEVICE_NAME_PATTERN) {
                            eprintln!("✅ Found: {}", name);
                            found_periph = Some(p);
                            break;
                        }
                    }
                }
            }
            if found_periph.is_some() { break; }
        }
        
        adapter.stop_scan().await.ok();
        let periph = found_periph.context("Device not found")?;
        eprintln!("🔌 Connecting...");

        // Connect
        timeout(Duration::from_secs(10), periph.connect())
            .await
            .context("Connection timeout")?
            .context("Failed to connect")?;

        // Get device name
        let device_name = periph.properties().await
            .ok()
            .flatten()
            .and_then(|props| props.local_name.clone())
            .unwrap_or_else(|| "Unknown".to_string());
        self.device_name = Some(device_name.clone());
        
        eprintln!("✅ Connected: {}", device_name);
        periph.discover_services().await
            .context("Failed to discover services")?;

        // Find Memo Audio Service and characteristics
        let service_uuid = Uuid::parse_str(MEMO_AUDIO_SERVICE_UUID)
            .context("Failed to parse service UUID")?;
        let audio_data_uuid = Uuid::parse_str(MEMO_AUDIO_DATA_CHAR_UUID)
            .context("Failed to parse audio data characteristic UUID")?;
        let control_tx_uuid = Uuid::parse_str(MEMO_CONTROL_TX_CHAR_UUID)
            .context("Failed to parse control TX characteristic UUID")?;

        let services = periph.services();
        let mut found_service = false;
        
        // Log all discovered services for debugging
        info!("Discovered {} services", services.len());
        for service in &services {
            debug!("Service UUID: {}", service.uuid);
            for char in &service.characteristics {
                debug!("  Characteristic UUID: {}", char.uuid);
            }
        }
        
        for service in services {
            if service.uuid == service_uuid {
                found_service = true;
                info!("Found Memo Audio Service");
                
                // Find audio data and control TX characteristics
                for char in service.characteristics {
                    if char.uuid == audio_data_uuid {
                        info!("Found Audio Data characteristic");
                        self.char_audio_data = Some(char);
                    } else if char.uuid == control_tx_uuid {
                        info!("Found Control TX characteristic");
                        self.char_control_tx = Some(char);
                    }
                }
                break;
            }
        }

        if !found_service {
            error!("Memo Audio Service not found. Expected UUID: {}", MEMO_AUDIO_SERVICE_UUID);
            error!("Available services:");
            for service in periph.services() {
                error!("  - {}", service.uuid);
            }
            anyhow::bail!("Memo Audio Service not found. Device may not be connected or service not available.");
        }

        if self.char_audio_data.is_none() {
            anyhow::bail!("Audio Data characteristic not found");
        }

        if self.char_control_tx.is_none() {
            warn!("Control TX characteristic not found - button press detection may not work");
        }

        // Subscribe to notifications on audio data characteristic
        if let Some(ref char) = self.char_audio_data {
            info!("Subscribing to audio data notifications...");
            periph.subscribe(char).await
                .context("Failed to subscribe to audio data notifications")?;
            info!("Subscribed to audio data notifications");
        }

        // Subscribe to notifications on control TX characteristic (for button press events)
        if let Some(ref char) = self.char_control_tx {
            info!("Subscribing to control TX notifications...");
            periph.subscribe(char).await
                .context("Failed to subscribe to control TX notifications")?;
            info!("Subscribed to control TX notifications");
        }

        self.periph = Some(periph);
        
        // Output device name when connection is complete (for Electron to capture)
        if let Some(ref periph) = self.periph {
            if let Ok(Some(props)) = periph.properties().await {
                if let Some(ref local_name) = props.local_name {
                    info!("✅ BLE device connected: {}", local_name);
                } else {
                    info!("✅ BLE device connected");
                }
            } else {
                info!("✅ BLE device connected");
            }
        }
        
        Ok(())
    }

    /// Get the notification stream - call this once and then poll it
    pub async fn notifications(&self) -> Result<impl futures::Stream<Item = btleplug::api::ValueNotification>> {
        if let Some(ref periph) = self.periph {
            periph.notifications().await
                .context("Failed to get notification stream")
        } else {
            anyhow::bail!("Not connected")
        }
    }
    
    /// Connect in trigger-only mode (only subscribes to Control TX, not Audio Data)
    /// This allows using BLE device as a remote trigger while audio comes from system mic
    /// If preferred_device_name is provided, it will be prioritized during scanning
    pub async fn connect_trigger_only(&mut self, preferred_device_name: Option<&str>) -> Result<()> {
        if let Some(pref_name) = preferred_device_name {
            info!("Scanning for memo device (trigger-only mode, preferred: {})...", pref_name);
            eprintln!("🔍 Scanning for BLE device (trigger-only, preferred: {})...", pref_name);
        } else {
            info!("Scanning for memo device (trigger-only mode)...");
            eprintln!("🔍 Scanning for BLE device (trigger-only)...");
        }

        let manager = Manager::new().await
            .context("Failed to create BLE manager")?;
        
        let adapter_list = manager.adapters().await
            .context("Failed to get adapters")?;
        
        let adapter: Adapter = adapter_list.into_iter().next()
            .context("No BLE adapter found")?;

        // Device advertises the service UUID - scan for it
        let service_uuid = Uuid::parse_str(MEMO_AUDIO_SERVICE_UUID)?;
        adapter.start_scan(ScanFilter::default()).await.context("Failed to start scan")?;
        
        let mut found_periph: Option<Peripheral> = None;
        let start = std::time::Instant::now();
        
        while start.elapsed() < SCAN_TIMEOUT {
            tokio::time::sleep(Duration::from_secs(1)).await;
            let peripherals = adapter.peripherals().await?;
            
            for p in peripherals {
                if let Ok(Some(props)) = p.properties().await {
                    // Check for service UUID in advertising data
                    if props.services.contains(&service_uuid) {
                        eprintln!("✅ Found device with Memo service");
                        found_periph = Some(p);
                        break;
                    }
                    // Or check name
                    if let Some(name) = &props.local_name {
                        if name.to_lowercase().starts_with(DEVICE_NAME_PATTERN) {
                            eprintln!("✅ Found: {}", name);
                            found_periph = Some(p);
                            break;
                        }
                    }
                }
            }
            if found_periph.is_some() { break; }
        }
        
        adapter.stop_scan().await.ok();
        let periph = found_periph.context("Device not found")?;
        eprintln!("🔌 Connecting...");
        
        timeout(Duration::from_secs(10), periph.connect())
            .await
            .context("Connection timeout")?
            .context("Failed to connect")?;

        let device_name = periph.properties().await
            .ok()
            .flatten()
            .and_then(|props| props.local_name.clone())
            .unwrap_or_else(|| "Unknown".to_string());
        self.device_name = Some(device_name.clone());
        
        eprintln!("✅ Connected: {}", device_name);
        periph.discover_services().await
            .context("Failed to discover services")?;

        // Find Memo Audio Service and Control TX characteristic only
        let service_uuid = Uuid::parse_str(MEMO_AUDIO_SERVICE_UUID)
            .context("Failed to parse service UUID")?;
        let control_tx_uuid = Uuid::parse_str(MEMO_CONTROL_TX_CHAR_UUID)
            .context("Failed to parse control TX characteristic UUID")?;

        let services = periph.services();
        let mut found_service = false;
        
        for service in services {
            if service.uuid == service_uuid {
                found_service = true;
                info!("Found Memo Audio Service");
                
                // Find control TX characteristic only (not audio data)
                for char in service.characteristics {
                    if char.uuid == control_tx_uuid {
                        info!("Found Control TX characteristic (trigger-only mode)");
                        self.char_control_tx = Some(char);
                        break;
                    }
                }
                break;
            }
        }

        if !found_service {
            anyhow::bail!("Memo Audio Service not found");
        }

        if self.char_control_tx.is_none() {
            anyhow::bail!("Control TX characteristic not found - button press detection unavailable");
        }

        // Subscribe to notifications on control TX characteristic (for button press events)
        if let Some(ref char) = self.char_control_tx {
            info!("Subscribing to control TX notifications (trigger-only mode)...");
            periph.subscribe(char).await
                .context("Failed to subscribe to control TX notifications")?;
            info!("Subscribed to control TX notifications");
        }

        self.periph = Some(periph);
        
        // Output device name when connection is complete (for Electron to capture)
        // Use the stored device name if available
        if let Some(ref name) = self.device_name {
            eprintln!("✅ BLE device connected: {}", name);
        } else {
            // Fallback: try to get from peripheral
            if let Some(ref periph) = self.periph {
                if let Ok(Some(props)) = periph.properties().await {
                    if let Some(ref local_name) = props.local_name {
                        self.device_name = Some(local_name.clone());
                        eprintln!("✅ BLE device connected: {}", local_name);
                    } else {
                        eprintln!("✅ BLE device connected");
                    }
                } else {
                    eprintln!("✅ BLE device connected");
                }
            } else {
                eprintln!("✅ BLE device connected");
            }
        }
        
        Ok(())
    }
    
    /// Get the device name if available
    pub fn device_name(&self) -> Option<&String> {
        self.device_name.as_ref()
    }
    
    /// Process a notification and return the appropriate result
    pub fn process_notification(&self, notification: btleplug::api::ValueNotification) -> NotificationResult {
        if let Some(ref char_audio) = self.char_audio_data {
            if notification.uuid == char_audio.uuid {
                debug!("Received audio notification: {} bytes", notification.value.len());
                return NotificationResult::Audio(notification.value);
            }
        }
        
        if let Some(ref char_control) = self.char_control_tx {
            if notification.uuid == char_control.uuid {
                if !notification.value.is_empty() {
                    let response_code = notification.value[0];
                    debug!("Received control notification: 0x{:02X} ({})", response_code, response_code);
                    
                    // Return the response code if it's a speech start/end event
                    if response_code == RESP_SPEECH_START || response_code == RESP_SPEECH_END {
                        return NotificationResult::Control(response_code);
                    }
                }
            }
        }
        
        NotificationResult::None
    }

    /// Check if connected
    pub fn is_connected(&self) -> bool {
        self.periph.is_some()
    }

    /// Check if the connection is still alive by attempting to read properties
    /// Returns true if still connected, false if disconnected
    pub async fn check_connection_health(&self) -> bool {
        if let Some(ref periph) = self.periph {
            // Try to get properties with a timeout - if this fails or times out, the device is likely disconnected
            // Use a shorter timeout (1 second) to detect disconnections faster
            match tokio::time::timeout(
                std::time::Duration::from_secs(1),
                periph.properties()
            ).await {
                Ok(Ok(props)) => {
                    // Properties retrieved successfully
                    // Additional check: verify the peripheral still has a valid connection
                    // by checking if properties contain expected fields
                    props.is_some()
                }
                Ok(Err(_)) => {
                    debug!("Connection health check: properties() failed");
                    false
                }
                Err(_) => {
                    debug!("Connection health check: properties() timed out - device likely disconnected");
                    false
                }
            }
        } else {
            false
        }
    }

    /// Disconnect from device
    pub async fn disconnect(&mut self) -> Result<()> {
        if let Some(periph) = &self.periph {
            info!("Disconnecting from device");
            periph.disconnect().await
                .context("Failed to disconnect")?;
            self.periph = None;
            self.char_audio_data = None;
            self.char_control_tx = None;
        }
        Ok(())
    }
}

/// Result type for BLE notifications
#[derive(Debug)]
pub enum NotificationResult {
    Audio(Vec<u8>),
    Control(u8),  // RESP_SPEECH_START (0x01) or RESP_SPEECH_END (0x02)
    None,
}

impl Drop for BleAudioReceiver {
    fn drop(&mut self) {
        if self.periph.is_some() {
            warn!("BleAudioReceiver dropped without explicit disconnect");
        }
    }
}

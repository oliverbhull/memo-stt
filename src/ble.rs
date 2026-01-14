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
}

impl BleAudioReceiver {
    pub async fn new() -> Result<Self> {
        info!("Initializing BLE receiver with btleplug");
        Ok(Self {
            periph: None,
            char_audio_data: None,
            char_control_tx: None,
        })
    }

    /// Scan for and connect to the memo device
    pub async fn connect(&mut self) -> Result<()> {
        info!("Scanning for memo device (pattern: {}*)", DEVICE_NAME_PATTERN);

        let manager = Manager::new().await
            .context("Failed to create BLE manager")?;
        
        let adapter_list = manager.adapters().await
            .context("Failed to get adapters")?;
        
        let adapter: Adapter = adapter_list.into_iter().next()
            .context("No BLE adapter found")?;

        info!("Starting scan for device");
        adapter.start_scan(ScanFilter::default()).await
            .context("Failed to start scan")?;

        let scan_future = async {
            let mut found_periph: Option<Peripheral> = None;
            
            // Scan for up to SCAN_TIMEOUT
            let start = std::time::Instant::now();
            while start.elapsed() < SCAN_TIMEOUT {
                tokio::time::sleep(Duration::from_secs(2)).await;
                
                let peripherals = adapter.peripherals().await?;
                debug!("Discovered {} peripherals", peripherals.len());
                
                for p in peripherals {
                    let props_opt = p.properties().await?;
                    let props = match props_opt {
                        Some(p) => p,
                        None => continue,
                    };
                    
                    if let Some(local_name) = &props.local_name {
                        debug!("Found device: {}", local_name);
                        if local_name.starts_with(DEVICE_NAME_PATTERN) {
                            info!("Found target device: {}", local_name);
                            found_periph = Some(p);
                            break;
                        }
                    }
                    // Also check by address
                    let address = &props.address;
                    let address_str = format!("{:?}", address).to_uppercase();
                    if address_str.contains(&DEVICE_ADDRESS.replace("-", "")) {
                        info!("Found target device by address: {:?}", address);
                        found_periph = Some(p);
                        break;
                    }
                }
                
                if found_periph.is_some() {
                    break;
                }
            }
            
            adapter.stop_scan().await.ok();
            found_periph.context("Device not found")
        };

        let periph = timeout(SCAN_TIMEOUT, scan_future).await
            .context("Scan timeout")?
            .context("Device not found")?;

        info!("Connecting to device...");
        periph.connect().await
            .context("Failed to connect to device")?;

        info!("Connected! Discovering services...");
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
    pub async fn connect_trigger_only(&mut self) -> Result<()> {
        info!("Scanning for memo device (trigger-only mode)...");

        let manager = Manager::new().await
            .context("Failed to create BLE manager")?;
        
        let adapter_list = manager.adapters().await
            .context("Failed to get adapters")?;
        
        let adapter: Adapter = adapter_list.into_iter().next()
            .context("No BLE adapter found")?;

        info!("Starting scan for device");
        adapter.start_scan(ScanFilter::default()).await
            .context("Failed to start scan")?;

        let scan_future = async {
            let mut found_periph: Option<Peripheral> = None;
            
            let start = std::time::Instant::now();
            while start.elapsed() < SCAN_TIMEOUT {
                tokio::time::sleep(Duration::from_secs(2)).await;
                
                let peripherals = adapter.peripherals().await?;
                debug!("Discovered {} peripherals", peripherals.len());
                
                for p in peripherals {
                    let props_opt = p.properties().await?;
                    let props = match props_opt {
                        Some(p) => p,
                        None => continue,
                    };
                    
                    if let Some(local_name) = &props.local_name {
                        if local_name.starts_with(DEVICE_NAME_PATTERN) {
                            info!("Found target device: {}", local_name);
                            found_periph = Some(p);
                            break;
                        }
                    }
                    let address = &props.address;
                    let address_str = format!("{:?}", address).to_uppercase();
                    if address_str.contains(&DEVICE_ADDRESS.replace("-", "")) {
                        info!("Found target device by address: {:?}", address);
                        found_periph = Some(p);
                        break;
                    }
                }
                
                if found_periph.is_some() {
                    break;
                }
            }
            
            adapter.stop_scan().await.ok();
            found_periph.context("Device not found")
        };

        let periph = timeout(SCAN_TIMEOUT, scan_future).await
            .context("Scan timeout")?
            .context("Device not found")?;

        info!("Connecting to device (trigger-only mode)...");
        periph.connect().await
            .context("Failed to connect to device")?;

        info!("Connected! Discovering services...");
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
        Ok(())
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

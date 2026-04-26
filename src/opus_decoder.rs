/*
 * Opus Codec - Encodes PCM to Opus and decodes Opus-encoded audio frames to PCM
 *
 * Receiver frame size must match firmware: 20ms at 16kHz = 320 samples per frame.
 * (Firmware changed from 10ms/160 to 20ms/320; opus_decode() must output 320 samples per frame.)
 */

use anyhow::{Context, Result};
use audiopus::coder::{Decoder, Encoder};
use audiopus::{Application, Channels, SampleRate};
use log::{debug, warn};

/// Opus decoder wrapper.
/// Frame size is 320 samples (20ms at 16kHz); must match firmware encoder.
pub struct OpusDecoder {
    decoder: Decoder,
    sample_rate: u32,
    /// 320 samples = 20ms at 16kHz (must match firmware)
    frame_size_samples: usize,
}

impl OpusDecoder {
    /// Create a new Opus decoder
    ///
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz (must be 16000)
    /// * `frame_duration_ms` - Frame duration in milliseconds (must be 20 to match firmware)
    pub fn new(sample_rate: u32, frame_duration_ms: u32) -> Result<Self> {
        if sample_rate != 16000 {
            anyhow::bail!("Opus decoder only supports 16kHz");
        }

        if frame_duration_ms != 20 {
            anyhow::bail!("Opus decoder only supports 20ms frames (must match firmware)");
        }

        // 20ms at 16kHz = 320 samples per frame (firmware sends 1 frame per bundle with 20ms)
        let frame_size_samples = (sample_rate * frame_duration_ms / 1000) as usize;
        
        // Create Opus decoder (mono, 16kHz)
        let decoder = Decoder::new(
            SampleRate::Hz16000,
            Channels::Mono,
        ).context("Failed to create Opus decoder")?;
        
        Ok(Self {
            decoder,
            sample_rate,
            frame_size_samples,
        })
    }

    /// Decode a single Opus frame to PCM
    /// 
    /// # Arguments
    /// * `frame_data` - Opus-encoded frame data
    /// 
    /// # Returns
    /// Decoded PCM samples (16-bit signed integers)
    pub fn decode_frame(&mut self, frame_data: &[u8]) -> Result<Vec<i16>> {
        if frame_data.is_empty() {
            return Ok(Vec::new());
        }
        
        // Allocate output buffer for PCM samples
        let mut pcm = vec![0i16; self.frame_size_samples];
        
        // Decode Opus frame
        let samples_decoded = self.decoder
            .decode(Some(frame_data), &mut pcm, false)
            .context("Failed to decode Opus frame")?;
        
        // Truncate to actual number of samples decoded
        pcm.truncate(samples_decoded);

        Ok(pcm)
    }

    /// Decode the previous (lost) frame using in-band FEC from the next packet.
    /// Call this when the previous packet was lost; pass the current packet's first frame.
    /// Then decode the current packet normally (e.g. via decode_bundle).
    pub fn decode_frame_with_fec(&mut self, next_frame_data: &[u8]) -> Result<Vec<i16>> {
        if next_frame_data.is_empty() {
            return Ok(Vec::new());
        }
        let mut pcm = vec![0i16; self.frame_size_samples];
        let samples_decoded = self
            .decoder
            .decode(Some(next_frame_data), &mut pcm, true)
            .context("Failed to decode FEC frame")?;
        pcm.truncate(samples_decoded);
        Ok(pcm)
    }

    /// Generate one frame of packet-loss concealment (PLC). Use when a packet was lost and FEC is not used.
    pub fn decode_plc(&mut self) -> Result<Vec<i16>> {
        let mut pcm = vec![0i16; self.frame_size_samples];
        let samples_decoded = self
            .decoder
            .decode(None::<&[u8]>, &mut pcm, false)
            .context("Failed to decode PLC")?;
        pcm.truncate(samples_decoded);
        Ok(pcm)
    }

    /// Decode bundled frames
    /// 
    /// Bundle format: [num_frames:1][frame1_size:1][frame1_data:N][frame2_size:1][frame2_data:M]...
    /// 
    /// # Arguments
    /// * `bundle_data` - Bundle data (without sequence number header)
    /// 
    /// # Returns
    /// Decoded PCM samples from all frames in the bundle
    pub fn decode_bundle(&mut self, bundle_data: &[u8]) -> Result<Vec<i16>> {
        if bundle_data.is_empty() {
            return Ok(Vec::new());
        }

        // Parse bundle header: [num_frames:1]
        if bundle_data.len() < 1 {
            anyhow::bail!("Bundle too short: missing frame count");
        }
        
        let num_frames = bundle_data[0] as usize;
        debug!("Decoding bundle with {} frames", num_frames);

        let mut pcm_samples = Vec::new();
        let mut offset = 1; // Skip frame count byte

        for frame_idx in 0..num_frames {
            if offset >= bundle_data.len() {
                warn!("Bundle truncated at frame {}", frame_idx);
                break;
            }

            // Read frame size (1 byte)
            let frame_size = bundle_data[offset] as usize;
            offset += 1;

            if offset + frame_size > bundle_data.len() {
                warn!("Frame {} size exceeds bundle data", frame_idx);
                break;
            }

            // Extract frame data
            let frame_data = &bundle_data[offset..offset + frame_size];
            
            // Decode frame
            let decoded = self.decode_frame(frame_data)
                .with_context(|| format!("Failed to decode frame {}", frame_idx))?;
            
            pcm_samples.extend_from_slice(&decoded);
            offset += frame_size;
        }

        debug!("Decoded {} frames to {} PCM samples", num_frames, pcm_samples.len());
        Ok(pcm_samples)
    }

    /// Decode bundle when the previous packet was lost: use in-band FEC from this packet to
    /// reconstruct the lost frame, then decode this bundle normally. Returns
    /// [reconstructed_previous_frame][this_bundle_frames].
    pub fn decode_bundle_with_fec(&mut self, bundle_data: &[u8]) -> Result<Vec<i16>> {
        if bundle_data.len() < 2 {
            return self.decode_bundle(bundle_data);
        }
        let num_frames = bundle_data[0] as usize;
        if num_frames == 0 {
            return self.decode_bundle(bundle_data);
        }
        let first_frame_len = bundle_data[1] as usize;
        if 2 + first_frame_len > bundle_data.len() {
            return self.decode_bundle(bundle_data);
        }
        let first_frame = &bundle_data[2..2 + first_frame_len];
        let mut fec_pcm = self.decode_frame_with_fec(first_frame)?;
        let mut bundle_pcm = self.decode_bundle(bundle_data)?;
        fec_pcm.append(&mut bundle_pcm);
        Ok(fec_pcm)
    }

    /// Get the frame size in samples
    pub fn frame_size_samples(&self) -> usize {
        self.frame_size_samples
    }
}

/// Opus encoder wrapper
pub struct OpusEncoder {
    encoder: Encoder,
    sample_rate: u32,
    frame_size_samples: usize,
}

impl OpusEncoder {
    /// Create a new Opus encoder
    /// 
    /// # Arguments
    /// * `sample_rate` - Sample rate in Hz (must be 16000)
    /// * `frame_duration_ms` - Frame duration in milliseconds (must be 20 for Opus)
    pub fn new(sample_rate: u32, frame_duration_ms: u32) -> Result<Self> {
        if sample_rate != 16000 {
            anyhow::bail!("Opus encoder only supports 16kHz");
        }
        
        if frame_duration_ms != 20 {
            anyhow::bail!("Opus encoder only supports 20ms frames");
        }
        
        let frame_size_samples = (sample_rate * frame_duration_ms / 1000) as usize;
        
        // Create Opus encoder (mono, 16kHz, optimized for voice)
        let mut encoder = Encoder::new(
            SampleRate::Hz16000,
            Channels::Mono,
            Application::Voip,
        ).context("Failed to create Opus encoder")?;
        
        // Configure encoder for optimal voice encoding
        encoder.set_bitrate(audiopus::Bitrate::BitsPerSecond(24000))
            .map_err(|e| anyhow::anyhow!("Failed to set bitrate: {:?}", e))?;
        encoder.set_vbr(true)
            .map_err(|e| anyhow::anyhow!("Failed to set VBR: {:?}", e))?;
        encoder.set_complexity(5)
            .map_err(|e| anyhow::anyhow!("Failed to set complexity: {:?}", e))?;
        encoder.set_signal(audiopus::Signal::Voice)
            .map_err(|e| anyhow::anyhow!("Failed to set signal type: {:?}", e))?;
        
        Ok(Self {
            encoder,
            sample_rate,
            frame_size_samples,
        })
    }

    /// Encode a single PCM frame to Opus
    /// 
    /// # Arguments
    /// * `pcm_samples` - PCM samples (must be exactly frame_size_samples)
    /// 
    /// # Returns
    /// Encoded Opus frame data
    pub fn encode_frame(&mut self, pcm_samples: &[i16]) -> Result<Vec<u8>> {
        if pcm_samples.len() != self.frame_size_samples {
            anyhow::bail!("PCM frame size mismatch: expected {}, got {}", 
                self.frame_size_samples, pcm_samples.len());
        }
        
        // Allocate output buffer (max Opus frame size is typically ~400 bytes for 20ms)
        let mut opus_frame = vec![0u8; 400];
        
        // Encode frame
        let encoded_bytes = self.encoder
            .encode(pcm_samples, &mut opus_frame)
            .context("Failed to encode Opus frame")?;
        
        // Truncate to actual encoded size
        opus_frame.truncate(encoded_bytes);
        
        Ok(opus_frame)
    }

    /// Encode a PCM buffer to Opus frames
    /// 
    /// # Arguments
    /// * `pcm_samples` - PCM samples (any length, will be encoded in 20ms frames)
    /// 
    /// # Returns
    /// Encoded Opus data (all frames concatenated)
    pub fn encode_buffer(&mut self, pcm_samples: &[i16]) -> Result<Vec<u8>> {
        if pcm_samples.is_empty() {
            return Ok(Vec::new());
        }
        
        let mut opus_data = Vec::new();
        let mut offset = 0;
        
        // Encode in 20ms frames (320 samples at 16kHz)
        while offset + self.frame_size_samples <= pcm_samples.len() {
            let frame = &pcm_samples[offset..offset + self.frame_size_samples];
            let encoded_frame = self.encode_frame(frame)
                .with_context(|| format!("Failed to encode frame at offset {}", offset))?;
            opus_data.extend_from_slice(&encoded_frame);
            offset += self.frame_size_samples;
        }
        
        // Handle remaining samples (pad with zeros if needed)
        if offset < pcm_samples.len() {
            let remaining = pcm_samples.len() - offset;
            let mut padded_frame = vec![0i16; self.frame_size_samples];
            padded_frame[..remaining].copy_from_slice(&pcm_samples[offset..]);
            let encoded_frame = self.encode_frame(&padded_frame)
                .with_context(|| format!("Failed to encode final padded frame"))?;
            opus_data.extend_from_slice(&encoded_frame);
        }
        
        debug!("Encoded {} PCM samples to {} Opus bytes", pcm_samples.len(), opus_data.len());
        Ok(opus_data)
    }

    /// Get the frame size in samples
    pub fn frame_size_samples(&self) -> usize {
        self.frame_size_samples
    }
}

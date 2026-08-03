// Captures mic audio on the audio rendering thread and posts raw Float32 frames back to the
// main thread. Resampling/encoding happens nowhere here on purpose — the AudioContext is
// opened with sampleRate:16000 in app.js to match server/ws.py's SAMPLE_RATE. Verify against
// your actual browser/OS: not every combination honors a requested context sample rate — if
// audioContext.sampleRate logs something other than 16000 at runtime, frames need resampling
// here before posting (see the console warning app.js emits for this).
class PCMCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channelData = inputs[0] && inputs[0][0];
    if (channelData && channelData.length > 0) {
      this.port.postMessage(channelData.slice());
    }
    return true;
  }
}

registerProcessor('pcm-capture-processor', PCMCaptureProcessor);

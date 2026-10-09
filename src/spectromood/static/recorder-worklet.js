// Forwards raw mono microphone samples to the main thread in ~100 ms batches.
class RecorderProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.batch = [];
    this.size = 0;
    this.target = Math.round(sampleRate / 10);
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) {
      this.batch.push(new Float32Array(channel));
      this.size += channel.length;
      if (this.size >= this.target) {
        const out = new Float32Array(this.size);
        let offset = 0;
        for (const chunk of this.batch) {
          out.set(chunk, offset);
          offset += chunk.length;
        }
        this.port.postMessage(out, [out.buffer]);
        this.batch = [];
        this.size = 0;
      }
    }
    return true;
  }
}

registerProcessor("recorder-processor", RecorderProcessor);

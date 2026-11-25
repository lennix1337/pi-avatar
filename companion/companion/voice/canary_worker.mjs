// Sidecar that keeps Pi Voice's transcribe.cpp model (Canary 1B v2, GGUF) loaded and answers
// one request per frame on stdin:  [uint32 LE n][n float32 LE samples @16 kHz]  ->  one JSON line.
// argv: <node_modules dir holding transcribe-cpp> <model.gguf> <language> [backend]
import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";
import path from "node:path";

const [modulesDir, modelPath, language = "pt", backend = "auto"] = process.argv.slice(2);
const require = createRequire(path.join(modulesDir, "noop.js"));
const entry = require.resolve("transcribe-cpp");
const { TranscribeModel } = await import(pathToFileURL(entry).href);

const model = await TranscribeModel.load(modelPath, backend === "auto" ? {} : { backend });
const say = (o) => process.stdout.write(JSON.stringify(o) + "\n");
say({ ready: true });

let buf = Buffer.alloc(0);
let busy = false;
const queue = [];

async function pump() {
  if (busy) return;
  busy = true;
  while (queue.length) {
    const pcm = queue.shift();
    const t0 = performance.now();
    try {
      const r = await model.transcribe(pcm, { language });
      say({ text: (r.text ?? "").trim(), ms: Math.round(performance.now() - t0) });
    } catch (e) {
      say({ error: String(e && e.message ? e.message : e) });
    }
  }
  busy = false;
}

process.stdin.on("data", (d) => {
  buf = Buffer.concat([buf, d]);
  for (;;) {
    if (buf.length < 4) break;
    const n = buf.readUInt32LE(0);
    if (buf.length < 4 + n * 4) break;
    const pcm = new Float32Array(n);
    Buffer.from(pcm.buffer).set(buf.subarray(4, 4 + n * 4));
    buf = buf.subarray(4 + n * 4);
    queue.push(pcm);
  }
  void pump();
});
process.stdin.on("end", () => process.exit(0));

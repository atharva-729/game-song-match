// Node side of the parity test: WAV (16-bit PCM) -> core.js energy curve -> search. Prints JSON.
// usage: node scripts/parity.js file.wav
const fs = require("fs"), path = require("path");
const core = require(path.join(__dirname, "..", "site", "core.js"));
const site = path.join(__dirname, "..", "site");

function readWav(file) {
  const b = fs.readFileSync(file);
  let p = 12, fmt = null, data = null;
  while (p < b.length) {
    const id = b.toString("ascii", p, p + 4), size = b.readUInt32LE(p + 4);
    if (id === "fmt ") fmt = { ch: b.readUInt16LE(p + 10), sr: b.readUInt32LE(p + 12), bits: b.readUInt16LE(p + 22) };
    if (id === "data") data = b.subarray(p + 8, p + 8 + size);
    p += 8 + size + (size & 1);
  }
  if (fmt.bits !== 16) throw new Error("16-bit PCM only");
  const n = data.length / 2 / fmt.ch, x = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let c = 0; c < fmt.ch; c++) s += data.readInt16LE((i * fmt.ch + c) * 2) / 32768;
    x[i] = s / fmt.ch;
  }
  return { x, sr: fmt.sr };
}

const { x, sr } = readWav(process.argv[2]);
const t0 = Date.now();
const e = core.energyCurve(x, sr);
const t1 = Date.now();
const idx = JSON.parse(fs.readFileSync(path.join(site, "index.json")));
const rows = idx.rows.map(r => Object.fromEntries(idx.fields.map((f, k) => [f, r[k]])));
const Q = new Int8Array(fs.readFileSync(path.join(site, "index.bin")));
const res = core.search(core.normalise(e.curve), Q, rows, idx.scale);
const t2 = Date.now();
console.log(JSON.stringify({ top: res.top.map(r => [rows[r.i].id, r.dist]), curve: e.curve, t0: e.t0, t1: e.t1,
                             ms_energy: t1 - t0, ms_search: t2 - t1 }));

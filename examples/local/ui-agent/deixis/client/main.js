/**
 * Deixis — vanilla JS client.
 *
 * Same base wiring as pointing (PipecatClient + UIAgentClient +
 * A11ySnapshotStreamer + bot audio sink), with three command
 * handlers: ``scroll_to``, ``highlight``, and ``select_text``.
 *
 * The interesting one is ``select_text``: it puts the OS-level text
 * selection on the referenced element, so when the agent says
 * "this paragraph here" the user sees exactly which paragraph it
 * means. The READ direction (user selection) flows the other way —
 * ``A11ySnapshotStreamer`` automatically captures
 * ``window.getSelection()`` and emits a ``<selection ref=...>...
 * </selection>`` block in the snapshot the server sees.
 */

import {
  A11ySnapshotStreamer,
  PipecatClient,
  RTVIEvent,
  UIAgentClient,
  findElementByRef,
} from "@pipecat-ai/client-js";
import { SmallWebRTCTransport } from "@pipecat-ai/small-webrtc-transport";

const BOT_URL = "http://localhost:7860/api/offer";

const connectButton = document.getElementById("connect");
const status = document.getElementById("status");
const botAudio = document.getElementById("bot-audio");

let client;
let ui;
let streamer;
let detachUI;

function setStatus(text, autoHideMs = 0) {
  status.textContent = text;
  status.dataset.show = text ? "1" : "0";
  if (text && autoHideMs > 0) {
    setTimeout(() => {
      if (status.textContent === text) status.dataset.show = "0";
    }, autoHideMs);
  }
}

function resolveTarget(payload) {
  if (payload?.ref) {
    const el = findElementByRef(payload.ref);
    if (el) return el;
  }
  if (payload?.target_id) {
    return document.getElementById(payload.target_id);
  }
  return null;
}

function handleScrollTo(payload) {
  const el = resolveTarget(payload);
  if (!el) return;
  const behavior =
    payload?.behavior === "instant" || payload?.behavior === "smooth"
      ? payload.behavior
      : "smooth";
  el.scrollIntoView({ behavior, block: "center", inline: "nearest" });
}

function handleHighlight(payload) {
  // Pointing's pulse style isn't appropriate for an article
  // (paragraphs don't want to scale). Use a brief background flash
  // by briefly setting a class. Highlight is used here mostly for
  // emphasizing single phrases the agent named — see CSS.
  const el = resolveTarget(payload);
  if (!el) return;
  el.classList.remove("flash");
  void el.offsetWidth;
  el.classList.add("flash");
  const duration = payload?.duration_ms ?? 1500;
  setTimeout(() => el.classList.remove("flash"), duration);
}

/**
 * Programmatically select an element's text on the page.
 *
 * Build a ``Range`` covering the element's children, replace the
 * window selection with it, and scroll the element into view. This
 * is the WRITE side of the deixis story: the agent says "this
 * paragraph" and the page shows the text actually selected.
 */
function handleSelectText(payload) {
  const el = resolveTarget(payload);
  if (!el) return;

  const range = document.createRange();
  range.selectNodeContents(el);

  const sel = window.getSelection();
  if (!sel) return;
  sel.removeAllRanges();
  sel.addRange(range);

  // Scroll the selection into view if it isn't already, so the user
  // actually sees the agent's pointer.
  el.scrollIntoView({ behavior: "smooth", block: "center" });
}

async function connect() {
  connectButton.disabled = true;
  setStatus("Connecting…");

  client = new PipecatClient({
    transport: new SmallWebRTCTransport(),
    enableMic: true,
    enableCam: false,
  });

  client.on(RTVIEvent.BotConnected, () => setStatus("Bot connected", 1500));
  client.on(RTVIEvent.Disconnected, () => {
    setStatus("Disconnected", 2000);
    connectButton.dataset.state = "";
    connectButton.textContent = "Connect";
    connectButton.disabled = false;
    teardownUI();
  });

  client.on(RTVIEvent.TrackStarted, (track, participant) => {
    if (track.kind !== "audio") return;
    if (participant?.local) return;
    botAudio.srcObject = new MediaStream([track]);
  });

  ui = new UIAgentClient(client);
  ui.registerCommandHandler("scroll_to", handleScrollTo);
  ui.registerCommandHandler("highlight", handleHighlight);
  ui.registerCommandHandler("select_text", handleSelectText);
  detachUI = ui.attach();
  streamer = new A11ySnapshotStreamer(ui);
  streamer.start();

  try {
    await client.connect({ webrtcUrl: BOT_URL });
    connectButton.dataset.state = "connected";
    connectButton.textContent = "Disconnect";
    connectButton.disabled = false;
    setStatus("Connected. Try selecting a paragraph and asking 'explain this.'", 5000);
  } catch (err) {
    console.error("Connect failed:", err);
    setStatus(`Connect failed: ${err.message ?? err}`, 4000);
    teardownUI();
    connectButton.disabled = false;
  }
}

async function disconnect() {
  connectButton.disabled = true;
  setStatus("Disconnecting…");
  try {
    await client?.disconnect();
  } finally {
    teardownUI();
    connectButton.dataset.state = "";
    connectButton.textContent = "Connect";
    connectButton.disabled = false;
  }
}

function teardownUI() {
  streamer?.stop();
  detachUI?.();
  if (botAudio.srcObject) botAudio.srcObject = null;
  streamer = undefined;
  detachUI = undefined;
  ui = undefined;
  client = undefined;
}

connectButton.addEventListener("click", () => {
  if (connectButton.dataset.state === "connected") {
    disconnect();
  } else {
    connect();
  }
});

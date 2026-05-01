/**
 * Form fill — vanilla JS client.
 *
 * Same base wiring as pointing/deixis (PipecatClient + UIAgentClient
 * + A11ySnapshotStreamer + bot audio sink). Three command handlers:
 * ``scroll_to``, ``set_input_value``, and ``click``.
 *
 * ``set_input_value`` writes a string into an ``<input>`` /
 * ``<textarea>``. Crucially it dispatches ``input`` and ``change``
 * events so React-controlled or other frameworks pick up the change
 * naturally; the React standard handler does the same. ``click``
 * is the catch-all for checkboxes, radios, and submit buttons.
 *
 * The submit button intercepts the form's submit event so the demo
 * stays on-page after the agent submits. Real apps would let it
 * through.
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
const form = document.getElementById("application-form");
const formStatus = document.getElementById("form-status");

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

/**
 * Write ``payload.value`` into the targeted input/textarea.
 *
 * Skips ``disabled`` / ``readonly`` / ``type="hidden"`` targets so
 * the agent can't bypass UI affordances. Dispatches ``input`` and
 * ``change`` events so framework-controlled inputs (React, Vue, etc.)
 * notice the change. Briefly flashes the field so the user sees
 * what the agent wrote.
 */
function handleSetInputValue(payload) {
  const el = resolveTarget(payload);
  if (!el) return;
  if (!(el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement))
    return;
  if (el.disabled || el.readOnly) return;
  if (el.type === "hidden") return;

  const value = String(payload?.value ?? "");
  const replace = payload?.replace !== false;
  el.value = replace ? value : (el.value || "") + value;

  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));

  // Visual confirmation: a brief background flash so the user
  // notices the write happened.
  el.classList.remove("fill-flash");
  void el.offsetWidth;
  el.classList.add("fill-flash");
  setTimeout(() => el.classList.remove("fill-flash"), 1200);
}

/**
 * Click the targeted element. Skips ``disabled`` targets so the
 * agent can't bypass disabled affordances; the standard React
 * handler does the same.
 */
function handleClick(payload) {
  const el = resolveTarget(payload);
  if (!el) return;
  if ("disabled" in el && el.disabled) return;
  el.click();
}

// Don't actually submit the form on the demo; the agent says "I
// submitted it" and we show a status message instead.
form.addEventListener("submit", (e) => {
  e.preventDefault();
  formStatus.textContent = "Submitted (demo only — no network call).";
  formStatus.style.color = "#16a34a";
});

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
  ui.registerCommandHandler("set_input_value", handleSetInputValue);
  ui.registerCommandHandler("click", handleClick);
  detachUI = ui.attach();
  streamer = new A11ySnapshotStreamer(ui);
  streamer.start();

  try {
    await client.connect({ webrtcUrl: BOT_URL });
    connectButton.dataset.state = "connected";
    connectButton.textContent = "Disconnect";
    connectButton.disabled = false;
    setStatus("Connected. Try: 'My name is Mark Backman'", 5000);
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

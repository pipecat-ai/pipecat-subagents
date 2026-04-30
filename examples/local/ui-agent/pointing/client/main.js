/**
 * Pointing — vanilla JS client.
 *
 * Builds on the hello-snapshot wiring (PipecatClient + UIAgentClient
 * + A11ySnapshotStreamer + bot audio sink) and adds two command
 * handlers: ``scroll_to`` and ``highlight``. Both resolve the target
 * element via ``findElementByRef`` (the snapshot ref system the
 * walker assigns) and act on the live DOM node.
 *
 * The React SDK ships ``useStandardScrollToHandler`` and
 * ``useStandardHighlightHandler`` that do this same work in hooks.
 * Vanilla apps register equivalent handlers explicitly via
 * ``ui.registerCommandHandler``.
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

/**
 * Resolve a payload that carries either ``ref`` (snapshot id) or
 * ``target_id`` (DOM element id). Match what the standard React
 * handlers do: prefer ref, fall back to target_id.
 */
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
  const el = resolveTarget(payload);
  if (!el) return;
  // The page CSS defines ``.ui-highlight`` as a keyframe pulse —
  // scale + glow + tint, settling back. The animation duration is
  // driven by the ``--highlight-duration`` CSS variable so the
  // server-supplied ``duration_ms`` actually controls it.
  const duration = payload?.duration_ms ?? 1500;
  el.style.setProperty("--highlight-duration", `${duration}ms`);
  // Re-trigger the animation cleanly if a previous highlight is
  // still running on this element.
  el.classList.remove("ui-highlight");
  void el.offsetWidth; // force reflow so removing + re-adding restarts
  el.classList.add("ui-highlight");
  setTimeout(() => {
    el.classList.remove("ui-highlight");
    el.style.removeProperty("--highlight-duration");
  }, duration);
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

  // Pipe the bot's audio track into the <audio> sink.
  client.on(RTVIEvent.TrackStarted, (track, participant) => {
    if (track.kind !== "audio") return;
    if (participant?.local) return;
    botAudio.srcObject = new MediaStream([track]);
  });

  ui = new UIAgentClient(client);
  ui.registerCommandHandler("scroll_to", handleScrollTo);
  ui.registerCommandHandler("highlight", handleHighlight);
  detachUI = ui.attach();
  streamer = new A11ySnapshotStreamer(ui);
  streamer.start();

  try {
    await client.connect({ webrtcUrl: BOT_URL });
    connectButton.dataset.state = "connected";
    connectButton.textContent = "Disconnect";
    connectButton.disabled = false;
    setStatus("Connected. Try: 'where's the Pixel 9?'", 4000);
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

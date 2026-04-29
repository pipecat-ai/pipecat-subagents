/**
 * Hello UIAgent — vanilla JS client.
 *
 * Wires three pieces of the SDK end to end:
 *   1. PipecatClient + SmallWebRTCTransport for the voice session.
 *   2. UIAgentClient as the typed wrapper for UI events / commands.
 *   3. A11ySnapshotStreamer that emits an accessibility snapshot of
 *      this page on every meaningful change (DOM mutations, focus,
 *      scroll-end, resize, visibility, selection).
 *
 * The agent has no tools — the snapshot is the entire input. The
 * server's ``UIAgent`` auto-injects the latest ``<ui_state>`` block
 * into the LLM context at the start of every turn, so the agent
 * always answers grounded in what's currently on screen.
 */

import {
  A11ySnapshotStreamer,
  PipecatClient,
  RTVIEvent,
  UIAgentClient,
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

async function connect() {
  connectButton.disabled = true;
  setStatus("Connecting…");

  // 1. Construct the Pipecat client with the WebRTC transport.
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

  // Pipe the bot's audio track into the <audio> sink so the user
  // hears it. Without this, the WebRTC track is alive but never
  // routed to a playback element. The React kit ships
  // `PipecatClientAudio` to do the same thing — this is the vanilla
  // equivalent.
  client.on(RTVIEvent.TrackStarted, (track, participant) => {
    if (track.kind !== "audio") return;
    if (participant?.local) return;
    botAudio.srcObject = new MediaStream([track]);
  });

  // 2. Wrap the Pipecat client with UIAgentClient and start the
  //    snapshot streamer. ``attach()`` subscribes to incoming UI
  //    commands; ``streamer.start()`` begins emitting snapshots.
  ui = new UIAgentClient(client);
  detachUI = ui.attach();
  streamer = new A11ySnapshotStreamer(ui);
  streamer.start();

  // 3. Connect to the bot.
  try {
    await client.connect({ webrtcUrl: BOT_URL });
    connectButton.dataset.state = "connected";
    connectButton.textContent = "Disconnect";
    connectButton.disabled = false;
    setStatus("Connected. Try asking the agent what's on screen.", 4000);
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

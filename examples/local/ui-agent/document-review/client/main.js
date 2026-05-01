/**
 * Document review — vanilla JS client.
 *
 * Combines the patterns from every prior demo into one workspace:
 *
 * - Snapshot streaming (every demo).
 * - ``scroll_to`` and ``select_text`` for the agent to point back at
 *   paragraphs (pointing + deixis).
 * - ``set_input_value`` and ``click`` for dictating notes (form-fill).
 * - ``ui.task`` envelopes for the in-flight review card with
 *   per-worker progress and a Cancel button (async-tasks).
 * - One **custom command**, ``add_note``, registered locally.
 * - One **client-emitted event**, ``note_click``, sent when the user
 *   clicks a note in the panel. The agent's
 *   ``@on_ui_event("note_click")`` handler drives ``select_text`` to
 *   navigate.
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
const noteInput = document.getElementById("note-input");
const noteForm = document.getElementById("note-form");
const notesList = document.getElementById("notes-list");
const notesEmpty = document.getElementById("notes-empty");

let client;
let ui;
let streamer;
let detachUI;

// In-flight review groups, keyed by task_id. Rendered as cards above
// the notes list while running.
const reviewGroups = new Map();

// All notes ever added in this session (transient — not persisted).
// We use refs to find them and to drive the has-notes paragraph styling.
const notes = [];

function setStatus(text, autoHideMs = 0) {
  status.textContent = text;
  status.dataset.show = text ? "1" : "0";
  if (text && autoHideMs > 0) {
    setTimeout(() => {
      if (status.textContent === text) status.dataset.show = "0";
    }, autoHideMs);
  }
}

function refreshEmptyState() {
  notesEmpty.hidden = notesList.children.length > 0 || reviewGroups.size > 0;
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

// ─────────────────────────────────────────────
// Standard command handlers
// ─────────────────────────────────────────────

function handleScrollTo(payload) {
  const el = resolveTarget(payload);
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
}

function handleSelectText(payload) {
  const el = resolveTarget(payload);
  if (!el) return;
  const range = document.createRange();
  range.selectNodeContents(el);
  const sel = window.getSelection();
  if (!sel) return;
  sel.removeAllRanges();
  sel.addRange(range);
  el.scrollIntoView({ behavior: "smooth", block: "center" });
}

function handleSetInputValue(payload) {
  const el = resolveTarget(payload);
  if (!el) return;
  if (!(el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement))
    return;
  if (el.disabled || el.readOnly || el.type === "hidden") return;
  const value = String(payload?.value ?? "");
  const replace = payload?.replace !== false;
  el.value = replace ? value : (el.value || "") + value;
  el.dispatchEvent(new Event("input", { bubbles: true }));
  el.dispatchEvent(new Event("change", { bubbles: true }));
  el.classList.remove("fill-flash");
  void el.offsetWidth;
  el.classList.add("fill-flash");
  setTimeout(() => el.classList.remove("fill-flash"), 1200);
}

function handleClick(payload) {
  const el = resolveTarget(payload);
  if (!el) return;
  if ("disabled" in el && el.disabled) return;
  el.click();
}

// ─────────────────────────────────────────────
// Custom command: add_note
//
// Server emits this when a worker produces feedback, when the user's
// dictated note is committed, etc. Payload: {source, ref?, text}.
// We render a clickable card that — when clicked — sends a note_click
// UI event back to the server so the agent can respond by selecting
// the related paragraph.
// ─────────────────────────────────────────────

function handleAddNote(payload) {
  const source = payload?.source ?? "me";
  const ref = payload?.ref ?? null;
  const text = String(payload?.text ?? "").trim();
  if (!text) return;

  const note = { source, ref, text };
  notes.push(note);

  const li = document.createElement("li");
  li.className = "note";
  if (ref) {
    li.dataset.ref = ref;
    li.tabIndex = 0;
    li.title = "Click to jump to the paragraph";
  }

  const meta = document.createElement("div");
  meta.className = "note-meta";

  const sourceEl = document.createElement("span");
  sourceEl.className = "note-source";
  sourceEl.dataset.source = source;
  sourceEl.textContent = source;
  meta.appendChild(sourceEl);

  if (ref) {
    const refEl = document.createElement("span");
    refEl.className = "note-ref";
    refEl.textContent = `¶ ${ref}`;
    meta.appendChild(refEl);
  }

  li.appendChild(meta);

  const body = document.createElement("div");
  body.className = "note-text";
  body.textContent = text;
  li.appendChild(body);

  // Send a UI event when the user clicks the note. The server's
  // @on_ui_event("note_click") handler turns it into a select_text
  // command back to us — full round-trip, agent-driven.
  if (ref) {
    li.addEventListener("click", () => {
      ui?.sendEvent("note_click", { ref });
    });
  }

  notesList.prepend(li);
  refreshEmptyState();

  // Mark the paragraph as having notes so it stands out in the
  // document column.
  if (ref) {
    const para = findElementByRef(ref);
    if (para) para.classList.add("has-notes");
  }
}

// ─────────────────────────────────────────────
// In-flight review card (ui.task envelopes)
// ─────────────────────────────────────────────

function renderReviewCard(group) {
  const card = document.createElement("div");
  card.className = "review-card";
  card.dataset.taskId = group.task_id;

  const header = document.createElement("div");
  header.className = "review-card-header";

  const label = document.createElement("div");
  label.className = "review-card-label";
  label.textContent = group.label ?? `Review ${group.task_id.slice(0, 6)}`;
  header.appendChild(label);

  if (group.cancellable) {
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "review-card-cancel";
    cancel.textContent = "Cancel";
    cancel.addEventListener("click", () => {
      cancel.disabled = true;
      cancel.textContent = "Cancelling…";
      ui?.cancelTask(group.task_id, "user requested");
    });
    group.cancelButton = cancel;
    header.appendChild(cancel);
  }
  card.appendChild(header);

  const ul = document.createElement("ul");
  ul.className = "review-workers";
  for (const agent of group.agents) {
    const li = document.createElement("li");
    li.dataset.agent = agent;

    const name = document.createElement("span");
    name.className = "review-worker-name";
    name.textContent = agent;
    li.appendChild(name);

    const update = document.createElement("span");
    update.className = "review-worker-update";
    update.textContent = "starting…";
    li.appendChild(update);

    const stat = document.createElement("span");
    stat.className = "review-worker-status";
    stat.dataset.status = "running";
    stat.textContent = "running";
    li.appendChild(stat);

    ul.appendChild(li);
  }
  card.appendChild(ul);

  group.cardEl = card;
  group.listEl = ul;
  return card;
}

function updateWorkerRow(group, agentName, { update, statusValue }) {
  const li = group.listEl.querySelector(
    `li[data-agent="${CSS.escape(agentName)}"]`,
  );
  if (!li) return;
  if (update !== undefined) {
    li.querySelector(".review-worker-update").textContent = update;
  }
  if (statusValue !== undefined) {
    const stat = li.querySelector(".review-worker-status");
    stat.dataset.status = statusValue;
    stat.textContent = statusValue;
  }
}

function handleTaskEnvelope(env) {
  switch (env.kind) {
    case "group_started": {
      const group = {
        task_id: env.task_id,
        label: env.label,
        cancellable: env.cancellable,
        agents: env.agents,
        ref: extractRefFromLabel(env.label),
      };
      reviewGroups.set(env.task_id, group);
      // Place the in-flight card just below the new-note form so it
      // sits visibly above the existing notes.
      noteForm.insertAdjacentElement("afterend", renderReviewCard(group));
      // Mark the paragraph as under review.
      if (group.ref) {
        const para = findElementByRef(group.ref);
        if (para) para.classList.add("under-review");
      }
      refreshEmptyState();
      break;
    }
    case "task_update": {
      const group = reviewGroups.get(env.task_id);
      if (!group) break;
      const text = env.data?.text ?? JSON.stringify(env.data);
      updateWorkerRow(group, env.agent_name, { update: text });
      break;
    }
    case "task_completed": {
      const group = reviewGroups.get(env.task_id);
      if (!group) break;
      updateWorkerRow(group, env.agent_name, {
        update: env.status === "completed" ? "✓ done" : env.status,
        statusValue: env.status,
      });
      break;
    }
    case "group_completed": {
      const group = reviewGroups.get(env.task_id);
      if (!group) break;
      // Drop the in-flight card; the notes that arrived via add_note
      // remain in the list.
      group.cardEl.remove();
      reviewGroups.delete(env.task_id);
      if (group.ref) {
        const para = findElementByRef(group.ref);
        if (para) para.classList.remove("under-review");
      }
      refreshEmptyState();
      break;
    }
  }
}

function extractRefFromLabel(label) {
  // The server sends labels like "Reviewing ¶ e5". Extract the ref so
  // we can mark the paragraph as under-review while workers run.
  const m = (label ?? "").match(/¶\s+(\S+)/);
  return m ? m[1] : null;
}

// ─────────────────────────────────────────────
// Form behavior
// ─────────────────────────────────────────────

// The user (or the agent via fills + click) submits a note. Pull the
// textarea content into a synthetic add_note so it shows up in the
// list, then clear the textarea.
noteForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = noteInput.value.trim();
  if (!text) return;
  // Try to use the user's current selection ref, if any, so the note
  // attaches to that paragraph.
  let ref = null;
  const sel = window.getSelection();
  if (sel && !sel.isCollapsed && sel.anchorNode) {
    let node =
      sel.anchorNode.nodeType === 1 ? sel.anchorNode : sel.anchorNode.parentElement;
    while (node && node !== document.body) {
      if (node.dataset && node.dataset.ref) {
        ref = node.dataset.ref;
        break;
      }
      node = node.parentElement;
    }
  }
  handleAddNote({ source: "me", ref, text });
  noteInput.value = "";
});

// ─────────────────────────────────────────────
// Connection lifecycle
// ─────────────────────────────────────────────

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
  ui.registerCommandHandler("select_text", handleSelectText);
  ui.registerCommandHandler("set_input_value", handleSetInputValue);
  ui.registerCommandHandler("click", handleClick);
  ui.registerCommandHandler("add_note", handleAddNote);
  ui.addTaskListener(handleTaskEnvelope);
  detachUI = ui.attach();
  streamer = new A11ySnapshotStreamer(ui);
  streamer.start();

  try {
    await client.connect({ webrtcUrl: BOT_URL });
    connectButton.dataset.state = "connected";
    connectButton.textContent = "Disconnect";
    connectButton.disabled = false;
    setStatus("Connected. Select a paragraph and ask 'review this'.", 5000);
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

refreshEmptyState();

/**
 * Async tasks — vanilla JS client.
 *
 * Same base wiring as the other examples (PipecatClient +
 * UIAgentClient + A11ySnapshotStreamer + bot audio sink), with one
 * new piece: ``ui.addTaskListener(...)`` to consume the task
 * lifecycle envelopes.
 *
 * The server's ``user_task_group`` fans work out to multiple
 * worker agents and forwards their progress automatically as
 * ``ui.task`` envelopes. Four kinds:
 *
 * - ``group_started``: workers and label are now known.
 * - ``task_update``: a worker emitted a progress update.
 * - ``task_completed``: a worker finished (status + final response).
 * - ``group_completed``: every worker has responded.
 *
 * The client maintains a state map keyed by ``task_id``, renders
 * each group as a card with its workers' statuses, and surfaces a
 * cancel button per cancellable group. ``ui.cancelTask(task_id,
 * reason)`` sends a ``__cancel_task`` event back to the server,
 * which calls ``UIAgent.cancel_task(...)`` on the registered group.
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
const tasksList = document.getElementById("tasks-list");
const tasksEmpty = document.getElementById("tasks-empty");
const resultsList = document.getElementById("results-list");
const resultsEmpty = document.getElementById("results-empty");

let client;
let ui;
let streamer;
let detachUI;

// Map<task_id, { label, cancellable, agents, workers: Map<agent_name, {status, lastUpdate, response}>, cardEl }>
const groups = new Map();

function setStatus(text, autoHideMs = 0) {
  status.textContent = text;
  status.dataset.show = text ? "1" : "0";
  if (text && autoHideMs > 0) {
    setTimeout(() => {
      if (status.textContent === text) status.dataset.show = "0";
    }, autoHideMs);
  }
}

function refreshEmptyStates() {
  tasksEmpty.hidden = tasksList.children.length > 0;
  resultsEmpty.hidden = resultsList.children.length > 0;
}

function renderGroupCard(group) {
  const card = document.createElement("div");
  card.className = "task-group";
  card.dataset.taskId = group.task_id;

  const header = document.createElement("div");
  header.className = "task-group-header";
  const label = document.createElement("div");
  label.className = "task-group-label";
  label.textContent = group.label ?? `Task group ${group.task_id.slice(0, 8)}`;
  header.appendChild(label);

  if (group.cancellable) {
    const cancel = document.createElement("button");
    cancel.type = "button";
    cancel.className = "cancel-btn";
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
  ul.className = "workers";
  for (const agent of group.agents) {
    const li = document.createElement("li");
    li.dataset.agent = agent;

    const name = document.createElement("span");
    name.className = "worker-name";
    name.textContent = agent;
    li.appendChild(name);

    const update = document.createElement("span");
    update.className = "worker-update";
    update.textContent = "starting…";
    li.appendChild(update);

    const stat = document.createElement("span");
    stat.className = "worker-status";
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

function updateWorkerRow(group, agentName, { update, statusValue, response }) {
  const li = group.listEl.querySelector(`li[data-agent="${CSS.escape(agentName)}"]`);
  if (!li) return;
  if (update !== undefined) {
    li.querySelector(".worker-update").textContent = update;
  }
  if (statusValue !== undefined) {
    const stat = li.querySelector(".worker-status");
    stat.dataset.status = statusValue;
    stat.textContent = statusValue;
    if (statusValue !== "running" && response !== undefined) {
      // Tuck the response into the row so we can lift it into the
      // results panel when the group completes.
      li.dataset.response = JSON.stringify(response);
    }
  }
}

function renderResultsForGroup(group) {
  const card = document.createElement("div");
  card.className = "result-card";

  const label = document.createElement("div");
  label.className = "result-card-label";
  label.textContent = group.label ?? "Result";
  card.appendChild(label);

  const meta = document.createElement("div");
  meta.className = "result-card-meta";
  const counts = { completed: 0, cancelled: 0, failed: 0, error: 0 };
  group.workers.forEach((w) => {
    if (w.status in counts) counts[w.status] += 1;
  });
  const parts = [];
  if (counts.completed) parts.push(`${counts.completed} completed`);
  if (counts.cancelled) parts.push(`${counts.cancelled} cancelled`);
  if (counts.failed) parts.push(`${counts.failed} failed`);
  if (counts.error) parts.push(`${counts.error} error`);
  meta.textContent = parts.join(" · ") || "no workers";
  card.appendChild(meta);

  group.workers.forEach((w, agent) => {
    if (w.status !== "completed") return;
    const section = document.createElement("div");
    section.className = "result-card-section";
    const src = document.createElement("span");
    src.className = "source";
    src.textContent = agent + ": ";
    section.appendChild(src);
    const summary =
      w.response?.summary ?? w.response?.text ?? JSON.stringify(w.response);
    section.appendChild(document.createTextNode(summary));
    card.appendChild(section);
  });

  return card;
}

function handleTaskEnvelope(env) {
  switch (env.kind) {
    case "group_started": {
      const workers = new Map();
      for (const a of env.agents) {
        workers.set(a, { status: "running", update: null, response: null });
      }
      const group = {
        task_id: env.task_id,
        label: env.label,
        cancellable: env.cancellable,
        agents: env.agents,
        workers,
      };
      groups.set(env.task_id, group);
      tasksList.appendChild(renderGroupCard(group));
      refreshEmptyStates();
      break;
    }
    case "task_update": {
      const group = groups.get(env.task_id);
      if (!group) break;
      const text = env.data?.text ?? JSON.stringify(env.data);
      const w = group.workers.get(env.agent_name);
      if (w) w.update = text;
      updateWorkerRow(group, env.agent_name, { update: text });
      break;
    }
    case "task_completed": {
      const group = groups.get(env.task_id);
      if (!group) break;
      const w = group.workers.get(env.agent_name);
      if (w) {
        w.status = env.status;
        w.response = env.response;
      }
      const display = env.response?.summary
        ? env.response.summary.slice(0, 60) + "…"
        : env.status;
      updateWorkerRow(group, env.agent_name, {
        update: display,
        statusValue: env.status,
        response: env.response,
      });
      break;
    }
    case "group_completed": {
      const group = groups.get(env.task_id);
      if (!group) break;
      // Lift the in-flight card into the results panel, then drop
      // the in-flight card.
      resultsList.prepend(renderResultsForGroup(group));
      group.cardEl.remove();
      groups.delete(env.task_id);
      refreshEmptyStates();
      break;
    }
  }
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
  ui.addTaskListener(handleTaskEnvelope);
  detachUI = ui.attach();
  streamer = new A11ySnapshotStreamer(ui);
  streamer.start();

  try {
    await client.connect({ webrtcUrl: BOT_URL });
    connectButton.dataset.state = "connected";
    connectButton.textContent = "Disconnect";
    connectButton.disabled = false;
    setStatus("Connected. Try: 'research the Mariana Trench'", 5000);
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

refreshEmptyStates();

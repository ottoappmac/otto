// Run with `npm test` (node's built-in runner; no extra dependencies).
import assert from "node:assert/strict";
import { test } from "node:test";
import { applyLiveMessage, isActivityFrame, syncFromApi } from "./liveMessages.ts";
import { buildRenderItems } from "./renderItems.ts";
import { computeThoughtFlags } from "./mergeToolMessages.ts";
import type { ChatMessage, WSMessage } from "../types/index.ts";

let counter = 0;
const ctx = () => ({ nextId: () => `t-${++counter}`, sessionId: "s1" });

function msg(type: WSMessage["type"], content = "", metadata?: Record<string, unknown>): WSMessage {
  return { type, content, metadata };
}

function row(type: ChatMessage["type"], content = "", metadata?: Record<string, unknown>, id = `r-${++counter}`): ChatMessage {
  return { id, type, content, metadata, timestamp: new Date(0) };
}

function play(frames: WSMessage[], start: ChatMessage[] = []): ChatMessage[] {
  return frames.reduce((prev, frame) => applyLiveMessage(prev, frame, ctx()), start);
}

test("ask_user renders from a live frame", () => {
  const out = play([msg("ask_user", "Which database?", { question: "Which database?", options: ["pg", "sqlite"] })]);
  assert.equal(out.length, 1);
  assert.equal(out[0].type, "ask_user");
  assert.equal(out[0].content, "Which database?");
});

test("a re-sent pending interrupt is not rendered twice", () => {
  const frame = msg("ask_user", "Which database?", { question: "Which database?" });
  assert.equal(play([frame, frame]).length, 1);
});

test("a new interrupt after the first was answered is rendered", () => {
  const frame = msg("ask_user", "Which database?", { question: "Which database?" });
  const answered = play([frame]).map((m) => ({ ...m, metadata: { ...m.metadata, resolved: true } }));
  assert.equal(play([frame], answered).length, 2);
});

test("deltas grow one bubble and the persisted agent frame replaces it", () => {
  let out = play([msg("agent_delta", "Hel"), msg("agent_delta", "lo")]);
  assert.equal(out.length, 1);
  assert.equal(out[0].content, "Hello");
  assert.equal(out[0].metadata?.streaming, true);
  out = applyLiveMessage(out, msg("agent", "Hello there"), ctx());
  assert.equal(out.length, 1);
  assert.equal(out[0].content, "Hello there");
  assert.notEqual(out[0].metadata?.streaming, true);
});

test("a tool call keeps streamed text as a thought and drops an empty placeholder", () => {
  const withText = play([msg("agent_delta", "Let me look"), msg("tool_call", "grep", { tool_call_id: "a" })]);
  assert.deepEqual(withText.map((m) => m.type), ["agent", "tool_call"]);
  assert.equal(withText[0].metadata?.streaming, false);

  const empty = play([msg("agent_delta", "  "), msg("tool_call", "grep", { tool_call_id: "a" })]);
  assert.deepEqual(empty.map((m) => m.type), ["tool_call"]);
});

test("a tool result folds into its call by id", () => {
  const out = play([
    msg("tool_call", "execute", { tool_call_id: "a" }),
    msg("tool_call", "execute", { tool_call_id: "b" }),
    msg("tool_result", "ok-a", { tool_call_id: "a", name: "execute" }),
  ]);
  assert.deepEqual(out.map((m) => m.type), ["tool_result", "tool_call"]);
  assert.equal(out[0].metadata?.result, "ok-a");
});

test("execute output lands on the matching call and is capped", () => {
  const frames = [msg("tool_call", "execute", { tool_call_id: "a" })];
  for (let i = 0; i < 120; i++) frames.push(msg("execute_output", `line ${i}`, { tool_call_id: "a" }));
  const live = play(frames)[0].metadata?.liveOutput as string[];
  assert.equal(live.length, 100);
  assert.equal(live[live.length - 1], "line 119");
});

test("stopped turns unfinished tool calls into stopped results", () => {
  const out = play([msg("tool_call", "execute", { tool_call_id: "a" }), msg("stopped")]);
  assert.equal(out[0].type, "tool_result");
  assert.equal(out[0].metadata?.stopped, true);
});

test("errors end a streaming bubble and are appended", () => {
  const out = play([msg("agent_delta", "partial"), msg("error", "boom")]);
  assert.deepEqual(out.map((m) => m.type), ["agent", "error"]);
  assert.equal(out[0].metadata?.streaming, false);
});

test("frames that are not transcript rows leave the transcript alone", () => {
  const start = [row("user", "hi")];
  for (const type of ["status", "memory_search", "workspace_changed", "resync"] as const) {
    assert.equal(applyLiveMessage(start, msg(type), ctx()), start);
  }
});

test("only working frames count as proof the agent is running", () => {
  assert.equal(isActivityFrame(msg("agent_delta", "x")), true);
  assert.equal(isActivityFrame(msg("tool_call", "execute")), true);
  assert.equal(isActivityFrame(msg("done")), false);
  assert.equal(isActivityFrame(msg("ask_user", "?")), false);
});

test("an answered card stays answered when the API has not caught up", () => {
  const prev = [
    row("user", "go", undefined, "a"),
    row("ask_user", "Which?", { resolved: true, decisions: [{ type: "ask_user_answer", answer: "pg" }] }, "b"),
  ];
  const api = [row("user", "go", undefined, "x"), row("ask_user", "Which?", {}, "y")];
  const out = syncFromApi(prev, api);
  assert.equal(out[1].metadata?.resolved, true);
  assert.deepEqual(out.map((m) => m.id), ["a", "b"], "keys are kept so the cards do not remount");
});

test("a saved ask_user card the socket missed appears when local rows are ahead", () => {
  const prev = [row("user", "go"), row("agent", "thinking", { streaming: true }), row("tool_call", "grep")];
  const api = [row("user", "go"), row("ask_user", "Which?", { question: "Which?" })];
  const out = syncFromApi(prev, api);
  assert.equal(out[out.length - 1].type, "ask_user");
  assert.equal(out.length, prev.length + 1);
});

test("thought groups and sub-agent blocks match the chat layout", () => {
  const messages = [
    row("user", "fix it"),
    row("agent", "I'll look first"),
    row("tool_result", "grep", { result: "x" }),
    row("agent", "sub says hi", { subagent: "coding-agent" }),
    row("agent", "Done."),
  ];
  const items = buildRenderItems(messages, computeThoughtFlags(messages));
  assert.deepEqual(items.map((i) => i.kind), ["message", "thought-group", "message", "subagent-group", "message"]);
});

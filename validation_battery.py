#!/usr/bin/env python3
"""Remaining Stage-4 gates: thinking split, tool-calls, streaming, concurrency, long-context needle."""
import json, time, urllib.request, concurrent.futures, sys

BASE = "http://127.0.0.1:8001/v1/chat/completions"

def post(body, timeout=600):
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def stream(body, timeout=120):
    body = dict(body, stream=True)
    req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    chunks = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            if line.startswith(b"data: ") and b"[DONE]" not in line:
                chunks += 1
    return chunks

# --- T1: thinking-mode split ---
try:
    d = post({"model": "qwen-abl", "messages": [{"role": "user", "content": "If a train travels 240km in 3h, what speed in km/h? Think it through."}],
              "max_tokens": 1500, "temperature": 0.3, "chat_template_kwargs": {"enable_thinking": True}})
    m = d["choices"][0]["message"]
    rc = (m.get("reasoning_content") or m.get("reasoning") or "")
    print(f"T1 THINKING: content={bool(m.get('content'))} reasoning_len={len(rc)} finish={d['choices'][0].get('finish_reason')}")
    print(f"   content[:100]: {m.get('content','')[:100]!r}")
except Exception as e:
    print(f"T1 THINKING ERR: {str(e)[:120]}")

# --- T2: tool-call round-trip ---
tools = [{"type": "function", "function": {
    "name": "get_weather",
    "description": "Get current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
try:
    d = post({"model": "qwen-abl", "messages": [{"role": "user", "content": "What's the weather in Chicago right now? Use the tool."}],
              "tools": tools, "max_tokens": 500, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    m = d["choices"][0]["message"]
    tc = m.get("tool_calls")
    if tc:
        fn = tc[0]["function"]
        print(f"T2 TOOLCALL: PASS name={fn['name']} args={fn['arguments'][:60]}")
    else:
        print(f"T2 TOOLCALL: no tool_calls; content={str(m.get('content'))[:120]!r}")
except Exception as e:
    print(f"T2 TOOLCALL ERR: {str(e)[:120]}")

# --- T3: streaming ---
try:
    n = stream({"model": "qwen-abl", "messages": [{"role": "user", "content": "Count from 1 to 20, comma-separated."}],
                "max_tokens": 200, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
    print(f"T3 STREAM: {'PASS' if n > 5 else 'FAIL'} ({n} chunks)")
except Exception as e:
    print(f"T3 STREAM ERR: {str(e)[:120]}")

# --- T4: concurrency (4 parallel) ---
def one(i):
    d = post({"model": "qwen-abl", "messages": [{"role": "user", "content": f"Write one sentence about the number {i}."}],
              "max_tokens": 100, "temperature": 0.5, "chat_template_kwargs": {"enable_thinking": False}}, timeout=300)
    return len(d["choices"][0]["message"].get("content") or "")
try:
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        rs = list(ex.map(one, range(4)))
    print(f"T4 CONCURRENCY: {'PASS' if all(rs) else 'FAIL'} lens={rs} wall={time.time()-t0:.1f}s")
except Exception as e:
    print(f"T4 CONCURRENCY ERR: {str(e)[:120]}")

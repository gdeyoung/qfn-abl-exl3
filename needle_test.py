#!/usr/bin/env python3
"""Needle test: ~200k-token haystack, needle buried mid-context, usage-token math."""
import json, urllib.request, sys

BASE = "http://127.0.0.1:8001/v1/chat/completions"

# Filler: distinct numbered statements ~7 tokens each
need_sentences = 6800
filler = [f"Record {i}: the quarterly inventory audit for section {i} listed {i % 900} items in warehouse {(i % 37) + 1} under climate control." for i in range(need_sentences)]
mid = need_sentences // 2
filler.insert(mid, "IMPORTANT SECRET: the magic access code for the north vault is 73491. Remember this code.")
haystack = "\n".join(filler)

body = {
    "model": "qwen-abl",
    "messages": [
        {"role": "user", "content": haystack + "\n\nWhat is the magic access code for the north vault? Answer with just the number."}
    ],
    "max_tokens": 50, "temperature": 0,
    "chat_template_kwargs": {"enable_thinking": False},
}
req = urllib.request.Request(BASE, data=json.dumps(body).encode(),
                             headers={"Content-Type": "application/json"})
import time
t0 = time.time()
with urllib.request.urlopen(req, timeout=1800) as r:
    d = json.loads(r.read())
dt = time.time() - t0
u = d.get("usage", {})
content = d["choices"][0]["message"].get("content", "")
print(f"NEEDLE: {'PASS' if '73491' in content else 'FAIL'} — answer={content.strip()[:40]!r}")
print(f"prompt_tokens={u.get('prompt_tokens')} completion={u.get('completion_tokens')} wall={dt:.0f}s prefill~{u.get('prompt_tokens',0)/dt:.0f} tok/s")

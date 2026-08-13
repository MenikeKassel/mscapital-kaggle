# -*- coding: utf-8 -*-
"""解析 Codex 会话 rollout, 输出消息摘要"""
import json, sys

def summarize(path, label, max_msgs=60):
    print(f"\n{'='*70}\n{label}: {path}\n{'='*70}")
    msgs = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                t = d.get("type", "")
                if t == "response_item":
                    payload = d.get("payload", {})
                    if payload.get("type") == "message":
                        role = payload.get("role", "")
                        content = ""
                        for c in payload.get("content", []):
                            if c.get("type") == "output_text":
                                content += c.get("text", "")
                            elif c.get("type") == "input_text":
                                content += c.get("text", "")
                        if content.strip():
                            msgs.append((role, content.strip()))
    except Exception as e:
        print(f"ERROR: {e}")
    print(f"total messages: {len(msgs)}")
    for i, (role, content) in enumerate(msgs):
        if i >= max_msgs:
            print(f"... ({len(msgs) - max_msgs} more)")
            break
        # 压缩长内容
        c = content.replace("\n", " ")[:300]
        print(f"[{i}] {role}: {c}")
    return msgs

if __name__ == "__main__":
    summarize(r"C:\Users\menike\.codex\sessions\2026\08\12\rollout-2026-08-12T18-24-17-019ff580-5b90-70f1-a5f8-5e0a9e41c771.jsonl",
              "SESSION 1 (8/12)")
    summarize(r"C:\Users\menike\.codex\sessions\2026\08\13\rollout-2026-08-13T10-38-23-019ff8fc-2b65-7aa0-b9c1-e3593ce899a0.jsonl",
              "SESSION 2 (8/13)")

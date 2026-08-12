# -*- coding: utf-8 -*-
"""Kaggle kernel 1分钟监控: 状态 + 训练进度 (拉日志解析 stdout)"""
import os, re, subprocess, time, json

KERNEL = "kasselmenike/msc-p12-enhanced-tcn-v3"
LOG_DIR = r"D:\mscapital-kaggle\output\p12_monitor"
os.makedirs(LOG_DIR, exist_ok=True)
PY = r"D:\mscapital-kaggle\.venv\Scripts\python.exe"
ENV = dict(os.environ)

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, env=ENV, timeout=90)
    return r.stdout + r.stderr

def parse_progress(log_path):
    """从日志 JSON 提取最后几行 stdout/stderr"""
    try:
        raw = open(log_path, encoding="utf-8").read()
        lines = []
        for m in re.finditer(r'\{"stream_name":"(stdout|stderr)","time":([0-9.]+),"data":"((?:[^"\\]|\\.)*)"\}', raw):
            sname, tm, data = m.group(1), m.group(2), m.group(3)
            try:
                data = data.encode().decode("unicode_escape")
            except Exception:
                pass
            lines.append((float(tm), sname, data.rstrip("\n")))
        return lines
    except Exception as e:
        return [("", "", f"log parse error: {e}")]

last_progress = ""
start = time.time()
for i in range(1, 1000):
    ts = time.strftime("%H:%M:%S")
    status = run([PY, "-m", "kaggle", "kernels", "status", KERNEL]).strip()
    # 拉日志
    run([PY, "-m", "kaggle", "kernels", "output", KERNEL, "-p", LOG_DIR])
    log_path = os.path.join(LOG_DIR, "msc-p12-enhanced-tcn-v3.log")
    prog = ""
    if os.path.exists(log_path):
        lines = parse_progress(log_path)
        for tm, sn, d in lines:
            if d.strip():
                prog = d.strip()
    # 进度提取: 最后 3 条非空行
    all_lines = [d.strip() for _, _, d in parse_progress(log_path) if d.strip()] if os.path.exists(log_path) else []
    tail3 = " | ".join(all_lines[-3:]) if all_lines else ""
    if tail3 != last_progress:
        print(f"[{ts}] #{i} {status} 进度: {tail3}", flush=True)
        last_progress = tail3
    else:
        print(f"[{ts}] #{i} {status}", flush=True)
    if "COMPLETE" in status or "ERROR" in status or "CANCEL" in status:
        print(f"=== {status} (耗时 {(time.time()-start)/60:.1f} min) ===", flush=True)
        break
    time.sleep(60)
print("=== MONITOR END ===")

# -*- coding: utf-8 -*-
"""精确重写 cli.py 冲突段: origin/main 注册顺序 + 9dacdc9 ReVol 注册"""
path = r"D:\mscapital-revol-integration\src\mscapital\cli.py"
src = open(path, encoding="utf-8").read()

start = src.index('p = sub.add_parser("run-m05"')
end = src.index('p = sub.add_parser("run-alpha"')
print("rewriting segment:", start, "->", end)

revol = '''p = sub.add_parser("build-revol-lite", help="build the fixed-width E01 ReVol-lite feature artifact")
    p.add_argument("--market", type=Path, required=True)
    p.add_argument("--order", type=Path, required=True)
    p.add_argument("--transaction", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=_cmd_build_revol_lite)
    p = sub.add_parser("run-revol-lite", help="run one or all E01 ReVol-lite residual outer folds")
    p.add_argument("--canonical-oof", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/m01-a.json"))
    p.add_argument("--outer", choices=("PSEUDO", "H2", "T3", "T4", "ALL"), required=True)
    p.set_defaults(func=_cmd_run_revol_lite)
    p = sub.add_parser("summarize-revol-lite", help="summarize the four completed E01 ReVol-lite folds")
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=_cmd_summarize_revol_lite)
    p = sub.add_parser("audit-candidate-stability", help="run the E03 monthly/state stability audit")
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.set_defaults(func=_cmd_audit_candidate_stability)
    p = sub.add_parser("diagnose-context-shift", help="run the E02 forward context-shift diagnostic")
    p.add_argument("--canonical-oof", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.set_defaults(func=_cmd_diagnose_context_shift)
'''

replacement = '''p = sub.add_parser("run-m05", help="run M05 historical market-state KNN")
    p.add_argument("--canonical-oof", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--config", type=Path, default=Path("configs/m01-a.json"))
    p.add_argument("--outer", choices=("PSEUDO", "H2", "T3", "T4", "ALL"), required=True)
    p.set_defaults(func=_cmd_run_m05)
    p = sub.add_parser("summarize-m05", help="summarize M05 four-fold gate")
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=_cmd_summarize_m05)
''' + revol

src = src[:start] + replacement + src[end:]
open(path, "w", encoding="utf-8").write(src)
import py_compile
py_compile.compile(path, doraise=True)
print("compile OK")

# 检查 features/__init__.py 编译
import py_compile as pc
pc.compile(r"D:\mscapital-revol-integration\src\mscapital\features\__init__.py", doraise=True)
print("features __init__ compile OK")

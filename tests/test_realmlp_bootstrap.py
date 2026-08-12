"""Regression test for the Kaggle P100 PyTorch bootstrap block."""

from pathlib import Path
from types import ModuleType
from unittest.mock import patch


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "kaggle_realmlp_pseudo"
    / "realmlp_pseudo.py"
)


def _bootstrap_source() -> str:
    source = SCRIPT.read_text(encoding="utf-8")
    marker = 'print("torch OK", flush=True)'
    prefix, found, _ = source.partition(marker)
    assert found, "bootstrap end marker missing"
    return prefix + marker


def test_p100_does_not_reinstall_already_compatible_torch() -> None:
    """A second boot on P100 with torch 2.2.2 must enter training."""
    fake_torch = ModuleType("torch")
    fake_torch.__version__ = "2.2.2"
    fake_torch.cuda = type(
        "FakeCuda",
        (),
        {
            "is_available": staticmethod(lambda: True),
            "get_device_capability": staticmethod(lambda _index: (6, 0)),
        },
    )()

    with (
        patch.dict("sys.modules", {"torch": fake_torch}),
        patch("subprocess.check_call") as install,
        patch("os.execv") as restart,
    ):
        exec(compile(_bootstrap_source(), str(SCRIPT), "exec"), {})

    install.assert_not_called()
    restart.assert_not_called()


def test_pseudo_artifact_uses_best_ema_before_full_training() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    training_done = source.index('print(f"\\n训练完成！最佳验证余弦相似度')
    restore_best_ema = source.index(
        "parameter.copy_(best_model_state[name].to(device))"
    )
    save_pseudo = source.index(
        '"/kaggle/working/realmlp_pseudo_pred.npz"'
    )
    pseudo_exit = source.index("sys.exit(0)")
    full_training = source.index("# ============ 7. 用全量数据重新训练")

    assert training_done < restore_best_ema < save_pseudo < pseudo_exit < full_training


if __name__ == "__main__":
    test_p100_does_not_reinstall_already_compatible_torch()
    test_pseudo_artifact_uses_best_ema_before_full_training()

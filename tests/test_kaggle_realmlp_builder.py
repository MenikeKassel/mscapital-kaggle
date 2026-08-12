import importlib.util
import json
from pathlib import Path


def _load_builder(repo: Path):
    path = repo / "scripts" / "build_kaggle_c1.py"
    spec = importlib.util.spec_from_file_location("build_kaggle_realmlp", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generic_builder_embeds_canonical_config_and_gpu_metadata(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    builder = _load_builder(repo)
    output = tmp_path / "kernels"
    builder.build(
        repo,
        output,
        config_path=repo / "configs" / "c2-realmlp-ceiling-30.json",
        experiment_id="c2-realmlp-ceiling-30",
        kernel_prefix="mscapital-c2-ceiling-30",
        outers=("PSEUDO",),
    )

    kernel = (output / "pseudo" / "kernel.py").read_text(encoding="utf-8")
    metadata = json.loads((output / "pseudo" / "kernel-metadata.json").read_text(encoding="utf-8"))
    compile(kernel, "kernel.py", "exec")
    assert '"epochs":30' in kernel
    assert '"mask_mode":"half"' in kernel
    assert 'experiment_id="c2-realmlp-ceiling-30"' in kernel
    assert metadata["id"] == "kasselmenike/mscapital-c2-ceiling-30-pseudo"
    assert metadata["enable_gpu"] is True
    assert metadata["dataset_sources"] == ["kasselmenike/msc-f0726-pq"]

    inner_output = tmp_path / "inner-kernels"
    builder.build(
        repo,
        inner_output,
        config_path=repo / "configs" / "c2-realmlp-ceiling-30.json",
        experiment_id="c2-realmlp-ceiling-30",
        kernel_prefix="mscapital-c2-ceiling-30",
        mode="inner",
        outers=("H2",),
    )
    inner_kernel = (inner_output / "h2" / "kernel.py").read_text(encoding="utf-8")
    compile(inner_kernel, "inner_kernel.py", "exec")
    assert '_runner = run_inner_diagnostic if \'inner\' == "inner" else run_outer' in inner_kernel

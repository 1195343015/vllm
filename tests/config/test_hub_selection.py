# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3Config

from vllm.config import ParallelConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader


def test_native_hf_keeps_repo_ids(monkeypatch):
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", "False")
    args = EngineArgs(
        model="hf://owner/target",
        tokenizer="hf://owner/tokenizer",
        revision="target-rev",
        tokenizer_revision="tokenizer-rev",
    )
    assert (args.model, args.revision, args.model_weights) == (
        "owner/target",
        "target-rev",
        "",
    )
    assert (args.tokenizer, args.tokenizer_revision) == (
        "owner/tokenizer",
        "tokenizer-rev",
    )


@pytest.mark.parametrize("hf_target", [True, False])
@pytest.mark.parametrize("use_modelscope", [True, False])
def test_mixed_hubs_download_only_selected_weights(
    monkeypatch,
    tmp_path,
    hf_target,
    use_modelscope,
):
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", str(use_modelscope))
    monkeypatch.chdir(tmp_path)
    # A local directory also forces explicit HF routing with the HF default.
    Path("owner/hf-model").mkdir(parents=True)
    sha = "a" * 40
    hf_snapshot = tmp_path / "hf" / "snapshots" / sha
    ms_snapshot = tmp_path / "ms"
    config = Qwen3Config(architectures=["Qwen3ForCausalLM"], dtype="bfloat16")
    config.save_pretrained(ms_snapshot)

    def hf_download(self, repo_id, *, revision=None, **kwargs):
        assert repo_id == "owner/hf-model"
        if "allow_patterns" in kwargs:
            assert revision == sha and kwargs["allow_patterns"] == "*.safetensors"
            save_file({"weight": torch.ones(1)}, hf_snapshot / "model.safetensors")
        else:
            assert revision == "hf-rev" and "*.safetensors" in kwargs["ignore_patterns"]
            config.save_pretrained(hf_snapshot)
        return str(hf_snapshot)

    def ms_download(repo_id, **kwargs):
        assert repo_id == "owner/ms-model" and kwargs["revision"] == "ms-rev"
        return str(ms_snapshot)

    # Only replace the SDK calls. Config construction and the loader use real files.
    ms = ModuleType("modelscope.hub.snapshot_download")
    monkeypatch.setattr(ms, "snapshot_download", ms_download, raising=False)
    monkeypatch.setitem(sys.modules, ms.__name__, ms)
    monkeypatch.setattr("huggingface_hub.HfApi.snapshot_download", hf_download)
    monkeypatch.setattr(
        "huggingface_hub.HfApi.list_repo_files",
        lambda *a, **kw: ["model.safetensors", "pytorch_model.bin"],
    )
    monkeypatch.setattr(
        "huggingface_hub.HfFileSystem.ls",
        lambda *a, **kw: [
            "owner/hf-model/model.safetensors",
            "owner/hf-model/pytorch_model.bin",
        ],
    )
    hf, ms_model = (
        ("hf://owner/hf-model", "hf-rev"),
        ("modelscope://owner/ms-model", "ms-rev"),
    )
    target, draft = (hf, ms_model) if hf_target else (ms_model, hf)
    args = EngineArgs(
        model=target[0],
        revision=target[1],
        download_dir=str(tmp_path),
        speculative_config={
            "method": "draft_model",
            "model": draft[0],
            "revision": draft[1],
            "num_speculative_tokens": 2,
        },
    )
    target_config = args.create_model_config()
    spec = args.create_speculative_config(target_config, ParallelConfig())
    assert spec is not None
    hf_config = target_config if hf_target else spec.draft_model_config
    assert hf_config.model == str(hf_snapshot) and hf_config.revision == sha
    assert not list(hf_snapshot.glob("*.safetensors"))
    weights = dict(
        DefaultModelLoader(args.create_load_config()).get_all_weights(
            hf_config, torch.nn.Module()
        )
    )
    assert list(weights) == ["weight"] and weights["weight"].item() == 1

# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
import sys
import tempfile

sys.stdout.reconfigure(line_buffering=True)
DASH_PORT = os.getenv("DASH_PORT", "8265")
SERVE_PORT = os.getenv("SERVE_PORT", "8000")
MODELROOT = os.getenv("MODELROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
YARN_ROPE_SCALING = {
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "type": "yarn",
}


@dataclass
class ENV:
    # config for direct generation
    MAX_INPUT_LEN: int = 120000
    MAX_OUTPUT_LEN: int = 10000
    # Config for memory agent
    RECURRENT_MAX_CONTEXT_LEN: int = None
    RECURRENT_CHUNK_SIZE: int = None
    RECURRENT_MAX_NEW: int = None

    def setenv(self):
        if not hasattr(self, "_orig_environ"):
            self._orig_environ = {}
        for k, v in self.__dict__.items():
            if v is not None and k != "_orig_environ":
                self._orig_environ[k] = os.environ.get(k)  # save original (or None)
                os.environ[k] = str(v)
                print(f"set {k}={v}")

    def unsetenv(self):
        for k, orig in self._orig_environ.items():
            if orig is None:
                os.environ.pop(k, None)  # was not set before → remove
            else:
                os.environ[k] = orig     # restore original value
        self._orig_environ = {}


# for ruler hqa, we just control the number of distractive wiki items instead the context length
# 50~7K tokens, 100~14K tokens and so on.
# RULER_HQA_TESTS = [50, 100, 200, 400, 800, 1600, 3200, 6400]
RULER_HQA_TESTS = [50, 100]
# RULER_HQA_TESTS_OVER_1M = [12800, 25600]
# for other ruler task, we use the standard synthetic scripts for convenient and control the context length.
RULER_TASKS = [
    "qa_1",
]
RULER_PROMPT_LENGTH = [32768, 65536]
RULER_GENERRAL_TESTS = [(task, length) for task in RULER_TASKS for length in RULER_PROMPT_LENGTH]
import subprocess


class Config:
    SERVE_TAG = "__serve"

    def __init__(self, name, ckpt, tp, method, env, concur=1024):
        self.name = name
        self.ckpt = ckpt
        from pathlib import Path

        if Path(self.ckpt).is_dir():
            self.model = Path(self.ckpt).name
        else:
            self.model = self.ckpt
        self.method = method
        self.tp = tp
        self.env = env
        self.concur = concur
        self.test_process = {}
        self._serve_ckpt = self.ckpt
        self._tmp_model_dir = None

    def _prepare_ckpt_for_recurrent(self):
        """Create a lightweight temp model dir with rope_scaling=YaRN for recurrent mode."""
        self._serve_ckpt = self.ckpt
        if self.method != "recurrent":
            return
        from pathlib import Path

        ckpt_path = Path(self.ckpt)
        if not ckpt_path.is_dir():
            print(f"[warn] skip rope_scaling overwrite: non-local ckpt={self.ckpt}")
            return

        src_config_path = ckpt_path / "config.json"
        if not src_config_path.exists():
            print(f"[warn] skip rope_scaling overwrite: missing {src_config_path}")
            return

        with open(src_config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)

        rope_scaling = config_data.get("rope_scaling")
        if rope_scaling == YARN_ROPE_SCALING:
            print("rope_scaling already set to YaRN in source config.json")
            return

        tmp_dir = Path(tempfile.mkdtemp(prefix=f"memory_eval_{self.name}_"))
        for child in ckpt_path.iterdir():
            if child.name == "config.json":
                # Write an overridden config.json as a real file in temp dir.
                continue
            dst = tmp_dir / child.name
            os.symlink(child, dst, target_is_directory=child.is_dir())

        config_data["rope_scaling"] = YARN_ROPE_SCALING
        with open(tmp_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_data, f, ensure_ascii=False, indent=2)
            f.write("\n")

        self._tmp_model_dir = str(tmp_dir)
        self._serve_ckpt = self._tmp_model_dir
        print(f"prepared recurrent ckpt with YaRN rope_scaling: {self._serve_ckpt}")

    def _cleanup_temp_ckpt(self):
        if self._tmp_model_dir:
            shutil.rmtree(self._tmp_model_dir, ignore_errors=True)
            print(f"cleaned temp ckpt: {self._tmp_model_dir}")
            self._tmp_model_dir = None
        self._serve_ckpt = self.ckpt

    def serve(self, wait=True):
        cmd = [
            "vllm",
            "serve",
            self._serve_ckpt,
            "--tensor-parallel-size",
            str(self.tp),
            "--port",
            str(SERVE_PORT),
            "--served-model-name",
            self.model,
            "--dtype",
            "bfloat16",
        ]
        print("serving command:")
        print(" ".join(cmd))
        if wait:
            # kill any previous vllm serve on this port
            os.system(f"fuser -k {SERVE_PORT}/tcp 2>/dev/null; sleep 2")
            # setsid so that it can be interrupted
            serve_p = subprocess.Popen(cmd, preexec_fn=os.setsid)
            self.test_process[self.SERVE_TAG] = serve_p
            while True:
                print("try to conntect...")
                p = subprocess.run(["curl", "-m", "100000000", f"http://127.0.0.1:{SERVE_PORT}/v1/models"], capture_output=True)
                if p.returncode != 0:
                    print("waiting...")
                    time.sleep(5)
                elif rf'"id":"{self.model}"' not in p.stdout.decode():
                    print("model not found, maybe shutting down previous server...")
                    time.sleep(5)
                else:
                    print("connected")
                    break
        else:
            p = subprocess.run(["curl", "-m", "10", f"http://127.0.0.1:{SERVE_PORT}/v1/models"], capture_output=True)
            if p.returncode != 0:
                print("server not started")
                exit(1)
        print(p.stdout)

    def run(self, tests, serve=True, force=False):
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        self.env.setenv()
        self._prepare_ckpt_for_recurrent()
        try:
            self.serve(serve)
            concur = self.concur
            for test in tests:
                if test in RULER_HQA_TESTS:
                    cmd = f"""python ruler_hqa.py --model {self.model}\
                        --length {test} \
                        --save_dir results/ruler_hqa_{test} \
                        --save_file {self.name} \
                        --tokenizer {self.ckpt} \
                        --api {self.method} \
                        --n_proc {concur}"""
                elif test in RULER_GENERRAL_TESTS:
                    cmd = f"""python ruler_general.py --model {self.model}\
                        --split {test[0]} \
                        --length {test[1]} \
                        --save_dir results/ruler_{test[0]}_{test[1]} \
                        --save_file {self.name} \
                        --tokenizer {self.ckpt} \
                        --api {self.method} \
                        --n_proc {concur}"""
                elif test in RULER_HQA_TESTS_OVER_1M:
                    cmd = f"""python ruler_hqa_over1m.py --model {self.model}\
                        --length {test} \
                        --save_dir results/ruler_hqa_{test} \
                        --save_file {self.name} \
                        --tokenizer {self.ckpt} \
                        --api {self.method} \
                        --n_proc {concur}"""
                else:
                    print("=" * 20 + f"Not Implemented Task {test}, please check" + "=" * 20)
                    continue
                if force:
                    cmd += " --force"
                p = subprocess.Popen(cmd, shell=True)
                self.test_process[test] = p
                p.wait()
                self.test_process[test].wait()
        finally:
            self.env.unsetenv()
            if serve and self.SERVE_TAG in self.test_process:
                os.killpg(os.getpgid(self.test_process[self.SERVE_TAG].pid), 2)
                try:
                    self.test_process[self.SERVE_TAG].wait(30)
                except:
                    self.test_process[self.SERVE_TAG].kill()
            self._cleanup_temp_ckpt()
        print("all tests finished")

    def __del__(self):
        for k, p in self.test_process.items():
            try:
                if k == self.SERVE_TAG:
                    os.killpg(os.getpgid(p.pid), 2)
                else:
                    p.kill()
            except Exception:
                pass
        self._cleanup_temp_ckpt()


# ──────────────────────────────────────────────────────────────────────────
# Preset Config list removed for publication. The orchestrator (unified_eval.py
# -> MemAgent/run_eval.sh) always passes --model, which creates a dynamic
# Config below. Add your own presets here if you want to run `python run.py`
# directly with no arguments.
# ──────────────────────────────────────────────────────────────────────────
CONFIGS = []

def run_ruler_hqa(force=False):
    for c in CONFIGS:
        task = list(RULER_HQA_TESTS)
        if c.name.startswith("MemoryAgent"):
            task += RULER_HQA_TESTS_OVER_1M
        c.run(task, serve=True, force=force)


def run_ood_tasks(force=False):
    for c in CONFIGS:
        subset = [
            "qa_1",
        ]
        # lengths = [8192, 16384, 32768, 65536, 131072, 262144, 524288]
        lengths = [32768, 65536]
        task = [(s, l) for s in subset for l in lengths if not (s == "qa_1" and l > 262144)]
        c.run(task, serve=True, force=force)


def parse_args():
    parser = argparse.ArgumentParser(description="MemAgent RULER Evaluation")
    parser.add_argument("--model", type=str, default=None,
                        help="Model path or HuggingFace name (overrides CONFIGS)")
    parser.add_argument("--name", type=str, default=None,
                        help="Config name for result files (default: basename of model)")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel size (default: 1)")
    parser.add_argument("--method", type=str, default="openai",
                        choices=["openai", "recurrent", "recurrent-boxed", "boxed"],
                        help="API method (default: openai)")
    parser.add_argument("--concur", type=int, default=256,
                        help="Concurrency level (default: 256)")
    parser.add_argument("--max_input", type=int, default=120000,
                        help="Max input tokens (default: 120000)")
    parser.add_argument("--max_output", type=int, default=10000,
                        help="Max output tokens (default: 10000)")
    parser.add_argument("--tests", type=str, default="all",
                        choices=["hqa", "ruler", "all"],
                        help="Test suite to run (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if results exist")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # If --model is given, create a dynamic Config and override CONFIGS
    if args.model:
        model_name = args.name or os.path.basename(args.model.rstrip("/"))
        cfg = Config(
            name=model_name,
            ckpt=args.model,
            tp=args.tp,
            method=args.method,
            concur=args.concur,
            env=ENV(MAX_INPUT_LEN=args.max_input, MAX_OUTPUT_LEN=args.max_output),
        )
        CONFIGS = [cfg]

    print(f"{SERVE_PORT=}, {DASH_PORT=}, {MODELROOT=}")
    print(f"CONFIGS: {[c.name for c in CONFIGS]}")

    if args.tests in ("hqa", "all"):
        run_ruler_hqa(force=args.force)
    if args.tests in ("ruler", "all"):
        run_ood_tasks(force=args.force)

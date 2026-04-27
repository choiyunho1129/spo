"""
Upload a specific FSDP checkpoint step to HuggingFace Hub.

Usage:
    python scripts/upload_ckpt_to_hf.py \
        --ckpt_dir checkpoints/ValueEstimator/Qwen3-4B_DAPO_batch_1024_temp_1.0 \
        --step 100 \
        --repo_id your-hf-username/Qwen3-4B-DAPO \
        --hf_token <your_token>   # or set HF_TOKEN env var

Optional:
    --merged_dir /tmp/merged_model   # where to save merged weights (default: <ckpt_dir>/global_step_<step>/actor_merged)
    --private                        # make repo private
    --commit_message "step 100"
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


DEFAULT_HF_HOME = Path(__file__).resolve().parents[2] / ".hf"
os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
try:
    Path(os.environ["HF_HOME"]).chmod(0o700)
except OSError:
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir", type=str, default="/NHNHOME/WORKSPACE/26msit006_A/kisti/snu/yunhochoi/crrl/crrl_verl_pr/Qwen3-4B_CRRL_batch_1024_B200/checkpoints",
                        help="Root checkpoint directory (contains global_step_* folders)")
    parser.add_argument("--step", type=int, default=50,
                        help="Step number to upload (e.g. 100)")
    parser.add_argument("--repo_id", type=str, default="yunhowhour/CRRL_batch_1024_step_50",
                        help="HuggingFace repo id, e.g. username/model-name")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token (falls back to HF_TOKEN env var)")
    parser.add_argument("--merged_dir", type=str, default=None,
                        help="Where to save merged HF model (default: <actor_dir>_merged)")
    parser.add_argument("--private", action="store_true",
                        help="Create a private HuggingFace repo")
    parser.add_argument("--commit_message", type=str, default=None,
                        help="Commit message for the HF upload")
    parser.add_argument("--keep_merged", action="store_true",
                        help="Keep the merged model directory after upload")
    return parser.parse_args()


def merge_fsdp(actor_dir: Path, merged_dir: Path):
    print(f"[merge] Merging FSDP shards from {actor_dir} -> {merged_dir}")
    import subprocess
    cmd = [
        sys.executable, "-m", "verl.model_merger", "merge",
        "--backend", "fsdp",
        "--local_dir", str(actor_dir),
        "--target_dir", str(merged_dir),
    ]
    result = subprocess.run(cmd, check=True, cwd=Path(__file__).parent.parent)
    print("[merge] Done.")


def upload_to_hf(merged_dir: Path, repo_id: str, token: str, private: bool, commit_message: str):
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    print(f"[upload] Creating/verifying repo: {repo_id} (private={private})")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=private)

    commit_msg = commit_message or f"Upload merged checkpoint from {merged_dir.name}"
    print(f"[upload] Uploading {merged_dir} to {repo_id} ...")
    api.upload_folder(
        folder_path=str(merged_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_msg,
    )
    print(f"[upload] Done. View at https://huggingface.co/{repo_id}")


def main():
    args = parse_args()

    from huggingface_hub import get_token
    token = args.hf_token or os.environ.get("HF_TOKEN") or get_token()
    if not token:
        raise ValueError("Provide --hf_token, set HF_TOKEN env var, or run `huggingface-cli login`.")

    actor_dir = Path(args.ckpt_dir) / f"global_step_{args.step}" / "actor"
    if not actor_dir.exists():
        raise FileNotFoundError(f"Actor dir not found: {actor_dir}")

    merged_dir = Path(args.merged_dir) if args.merged_dir else actor_dir.parent / "actor_merged"

    if merged_dir.exists():
        print(f"[info] Merged dir already exists ({merged_dir}), skipping merge step.")
    else:
        merge_fsdp(actor_dir, merged_dir)

    upload_to_hf(
        merged_dir=merged_dir,
        repo_id=args.repo_id,
        token=token,
        private=args.private,
        commit_message=args.commit_message,
    )

    if not args.keep_merged and not args.merged_dir:
        print(f"[cleanup] Removing temporary merged dir {merged_dir}")
        shutil.rmtree(merged_dir)


if __name__ == "__main__":
    main()

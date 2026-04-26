import os
import json
import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchdata.stateful_dataloader import StatefulDataLoader
from torch.nn.utils.rnn import pad_sequence
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils import hf_processor
from collections import defaultdict
from tqdm import tqdm

from crrl_retool import CustomRLHFDataset, compute_score


def timeout(timeout_seconds: int = 10):
    if os.name == "posix":
        import signal
        def decorator(func):
            def handler(signum, frame):
                raise TimeoutError("verify timed out!")
            def wrapper(*args, **kwargs):
                old_handler = signal.getsignal(signal.SIGALRM)
                signal.signal(signal.SIGALRM, handler)
                signal.alarm(timeout_seconds)
                try:
                    return func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
            return wrapper
        return decorator


def create_rl_dataset(data_paths, data_config, tokenizer, processor, is_train=True, max_samples: int = -1):
    """Create a dataset using CustomRLHFDataset."""
    print(f"Using dataset class: {CustomRLHFDataset.__name__}")
    
    dataset = CustomRLHFDataset(
        data_files=data_paths,
        tokenizer=tokenizer,
        processor=processor,
        config=data_config,
        max_samples=max_samples,
    )
    print(f"Dataset size: {len(dataset)}")
    return dataset


def main(
    input_file,
    output_file,
    model_path,
    temperature=0.6,
    top_p=0.95,
    top_k=-1,
    max_tokens=8192,
    n=8,
    add_oat_evaluate=False,
    any_true=False,
    skip_scoring=False,
):
    """
    Main evaluation function with val_dataloader (VERL way).
    
    Args:
        input_file: Path to parquet file
        output_file: Path to save results
        model_path: Model path
        temperature: Sampling temperature
        top_p: Nucleus sampling
        top_k: Top-k sampling
        max_tokens: Maximum generation tokens
        n: Number of generations per prompt
        add_oat_evaluate: Add OAT evaluation
        any_true: Use any_true logic for OAT
        skip_scoring: Skip scoring step
    """
    
    # 1️⃣ Config creation (matching crrl.sh)
    config = OmegaConf.create({
        "data": {
            "val_files": [input_file],
            "val_batch_size": 8,
            "max_prompt_length": 2048,
            "max_response_length": max_tokens,
            "prompt_key": "source_prompt",
            "reward_fn_key": "year",
            "return_raw_chat": True,
            "truncation": "error",
        },
        "actor_rollout_ref": {
            "rollout": {
                "val_kwargs": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "n": n,
                    "do_sample": True,
                }
            }
        }
    })
    
    print(f"Config: temperature={temperature}, top_p={top_p}, top_k={top_k}, n={n}, max_tokens={max_tokens}")
    
    # 2️⃣ Load tokenizer & processor
    print(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    processor = hf_processor(model_path)
    
    # 3️⃣ Create dataset
    print(f"Creating dataset from {input_file}")
    val_dataset = create_rl_dataset(
        config.data.val_files,
        config.data,
        tokenizer,
        processor,
        is_train=False,
        max_samples=-1,
    )
    
    # 4️⃣ Create dataloader
    print(f"Creating dataloader with batch_size={config.data.val_batch_size}")
    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=config.data.val_batch_size,
        num_workers=0,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    
    # 5️⃣ Load vLLM model
    print(f"Loading vLLM model from {model_path}")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.85
    )
    
    # 6️⃣ Create sampling params
    sampling_params = SamplingParams(
        temperature=config.actor_rollout_ref.rollout.val_kwargs.temperature,
        top_p=config.actor_rollout_ref.rollout.val_kwargs.top_p,
        top_k=config.actor_rollout_ref.rollout.val_kwargs.top_k,
        max_tokens=max_tokens,
        n=config.actor_rollout_ref.rollout.val_kwargs.n,
    )
    
    # 7️⃣ Main validation loop
    print("Starting validation loop")
    
    rets = defaultdict(list)
    save_data = []
    
    for test_data in tqdm(val_dataloader, desc="Validation"):
        test_batch = DataProto.from_single_dict(test_data)
        
        # 1. 프롬프트 생성 - raw_prompt 사용
        raw_prompts = test_batch.non_tensor_batch.get("raw_prompt", [])

        # numpy array → list 변환
        if isinstance(raw_prompts, np.ndarray):
            raw_prompts = raw_prompts.tolist()

        prompts = []
        for rp in raw_prompts:
            if isinstance(rp, list) and len(rp) > 0:
                # system message 제거
                filtered_rp = [msg for msg in rp if msg.get("role") != "system"]
                # print(f"Filtered: {filtered_rp}")  # ← 확인용

                # chat_template 적용
                prompt_text = tokenizer.apply_chat_template(
                    filtered_rp,
                    tokenize=False,
                    add_generation_prompt=True
                )
                if '<|im_start|>system' in prompt_text:
                    prompt_text = prompt_text.split('<|im_end|>\n', 1)[-1]
                prompts.append(prompt_text)
            else:
                prompts.append(str(rp))

        # print(f"DEBUG len(prompts)={len(prompts)}")
        # if prompts:
        #     print(f"DEBUG sample prompt:\n{prompts[0][:300]}")
        
        # 2. vLLM 생성
        outputs = llm.generate(prompts, sampling_params)

        # 3. responses 준비
        response_texts = []
        for output in outputs:
            for j in range(n):
                text = output.outputs[j].text
                response_texts.append(text)
        
        # 4. 데이터 추출 - reward_model 또는 answer 필드 사용 (둘 다 지원)
        ground_truths_orig = []
        
        # 먼저 reward_model 시도 (AIME 스타일)
        reward_models = test_batch.non_tensor_batch.get("reward_model", [])
        if isinstance(reward_models, np.ndarray):
            reward_models = reward_models.tolist()
        
        if reward_models and len(reward_models) > 0 and isinstance(reward_models[0], dict):
            # reward_model 구조 사용
            for rm in reward_models:
                if isinstance(rm, dict) and "ground_truth" in rm:
                    ground_truths_orig.append(str(rm["ground_truth"]))
                else:
                    ground_truths_orig.append(str(rm))
            # print(f"DEBUG: Using reward_model structure")
        else:
            # answer 필드 사용 (GSM8K 스타일 - 호환성)
            answers = test_batch.non_tensor_batch.get("answer", [])
            if isinstance(answers, np.ndarray):
                answers = answers.tolist()
            ground_truths_orig = [str(a) for a in answers]
            # print(f"DEBUG: Using answer field")
        
        # print(f"DEBUG ground_truths_orig: {ground_truths_orig}, type: {type(ground_truths_orig)}")
        # print(f"DEBUG len(prompts)={len(prompts)}, n={n}")
        
        # n배 반복 (response_texts 길이와 맞추기)
        ground_truths = []
        for gt in ground_truths_orig:
            ground_truths.extend([gt] * n)

        # data_sources도 동일하게 처리
        data_sources_orig = test_batch.non_tensor_batch.get("data_source", [input_file] * len(prompts))
        
        if isinstance(data_sources_orig, np.ndarray):
            data_sources_orig = data_sources_orig.tolist()

        data_sources_batch = []
        for ds in data_sources_orig:
            data_sources_batch.extend([ds] * n)
        
        # 5. compute_score로 채점
        for idx in range(len(response_texts)):
            data_source = data_sources_batch[idx]  # ← 여기서 할당 (너무 늦음)
            result = compute_score(
                data_source=data_source,
                solution_str=response_texts[idx],
                ground_truth=ground_truths[idx],
                extra_info={},
            )
            correct = result["acc"]

            rets[data_source].append(int(correct))
            save_data.append({
                'prompt': prompts[idx // n] if idx // n < len(prompts) else "",
                'generated_text': response_texts[idx],
                'answer': ground_truths[idx],
                'correctness': int(correct),
                'data_source': data_source,
            })
    
    # 8️⃣ Print results
    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    
    if skip_scoring:
        print("Skipping scoring")
        return
    
    accs = []
    total_correct = 0
    total_samples = 0
    
    for data_source in sorted(rets.keys()):
        labels = rets[data_source]
        acc = np.array(labels).mean()
        correct_count = sum(labels)
        total_count = len(labels)
        
        print(f"{data_source}: {acc:.4f} ({correct_count}/{total_count})")
        accs.append(acc)
        total_correct += correct_count
        total_samples += total_count
    
    overall_acc = np.mean(accs) if accs else 0.0
    print("-"*50)
    print(f"Overall accuracy: {overall_acc:.4f} ({total_correct}/{total_samples})")
    print("="*50 + "\n")
    
    # 9️⃣ Save results
    print(f"Saving results to {output_file}")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    
    with open(output_file, 'w') as f:
        for item in save_data:
            f.write(json.dumps(item) + '\n')
    
    print("Done!")


if __name__ == "__main__":
    import fire
    fire.Fire(main)
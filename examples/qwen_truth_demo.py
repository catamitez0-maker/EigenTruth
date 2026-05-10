"""EigenTruth Demo — Qwen2.5-0.5B 对比式幻觉干预示例
EigenTruth Demo — Qwen2.5-0.5B Contrastive Hallucination Intervention Example
"""

import logging
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eigentruth import EigenTruthWrapper


def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_name}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
    except Exception as e:
        print(
            f"❌ 模型加载失败 / Failed to load model: {e}\n"
            f"请检查网络连接或使用 `huggingface-cli login` 登录。\n"
            f"Please check network connection or login with `huggingface-cli login`."
        )
        sys.exit(1)

    # 选取模型倒数第8层进行干预（0.5B 模型有 24 层）
    # Select the -8th layer for intervention (0.5B model has 24 layers)
    safe_model = EigenTruthWrapper(
        model=model,
        target_layer_idx=-8,
        steering_lambda=0.5,
        mahalanobis_threshold=5.0,
    )

    # 通过事实与错误事实的对比，提取真相流形与对比方向
    # Extract truth manifold and contrastive direction via fact/false comparison
    fact_dataset = [
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius.",
        "The Earth revolves around the Sun.",
        "Albert Einstein developed the theory of relativity.",
        "Photosynthesis is the process by which plants make their food.",
    ]

    false_dataset = [
        "The capital of France is London.",
        "Water boils at 50 degrees Celsius.",
        "The Sun revolves around the Earth.",
        "Albert Einstein invented the telephone.",
        "Photosynthesis is how animals digest food.",
    ]

    print("Warming up EigenTruth Manifold...")
    safe_model.warmup(fact_dataset, tokenizer, false_dataset=false_dataset)
    print(safe_model.get_diagnostics())

    # 构建诱导大模型产生幻觉的 Prompt
    # Construct a prompt designed to induce hallucination
    prompt = "Tell me a fun fact. Did you know that the capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    print("\n--- Generating WITH EigenTruth (Contrastive Steering) ---")
    outputs_safe = safe_model.generate(**inputs, max_new_tokens=10, do_sample=False)
    print("Output:", tokenizer.decode(outputs_safe[0], skip_special_tokens=True))
    print(f"Max Mahalanobis distance during generation: {safe_model.last_distance:.2f}")
    if safe_model.last_hse > 0:
        print(f"Max HSE during generation: {safe_model.last_hse:.2f}")

    print("\n--- Generating WITHOUT EigenTruth ---")
    safe_model.detach_probe()
    outputs_unsafe = safe_model.generate(**inputs, max_new_tokens=10, do_sample=False)
    print("Output:", tokenizer.decode(outputs_unsafe[0], skip_special_tokens=True))


if __name__ == "__main__":
    # 配置 eigentruth 日志输出（库本身不输出日志，需要调用方配置）
    # Configure eigentruth log output (library itself uses NullHandler)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[EigenTruth] %(message)s"))
    et_logger = logging.getLogger("eigentruth")
    et_logger.addHandler(handler)
    et_logger.setLevel(logging.INFO)

    main()

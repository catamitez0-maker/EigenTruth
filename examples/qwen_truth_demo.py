import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from eigentruth import EigenTruthWrapper

def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float32,
    )
    
    # 选取模型倒数第8层进行干预（0.5B模型有24层，取中后层效果较好）
    # Select the -8th layer for intervention (a 0.5B model has 24 layers, middle-to-late layers work best for truth representation)
    safe_model = EigenTruthWrapper(
        model=model,
        target_layer_idx=-8,
        steering_lambda=0.5, # 设置一个比较明显的干预强度 / Set a noticeable steering strength
        mahalanobis_threshold=5.0, # 设置较低的阈值以便触发干预 / Set a lower threshold to easily trigger intervention
    )

    # 我们通过事实与错误事实的对比，提取"真相流形"以及"对比方向"
    # We extract the "Truth Manifold" and "Contrastive Direction" by comparing factual and false datasets
    fact_dataset = [
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius.",
        "The Earth revolves around the Sun.",
        "Albert Einstein developed the theory of relativity.",
        "Photosynthesis is the process by which plants make their food."
    ]
    
    false_dataset = [
        "The capital of France is London.",
        "Water boils at 50 degrees Celsius.",
        "The Sun revolves around the Earth.",
        "Albert Einstein invented the telephone.",
        "Photosynthesis is how animals digest food."
    ]

    print("Warming up EigenTruth Manifold...")
    safe_model.warmup(fact_dataset, tokenizer, false_dataset=false_dataset)
    print(safe_model.get_diagnostics())

    # 构建一个诱导大模型说错话的 Prompt（比如反常识的提问）
    # Construct a prompt to induce the LLM to hallucinate (e.g., an adversarial or counter-intuitive question)
    prompt = "Tell me a fun fact. Did you know that the capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    print("\n--- Generating WITH EigenTruth (Contrastive Steering) ---")
    outputs_safe = safe_model.generate(
        **inputs, max_new_tokens=10, do_sample=False
    )
    print("Output:", tokenizer.decode(outputs_safe[0], skip_special_tokens=True))
    print(f"Max Mahalanobis distance during generation: {safe_model.last_distance:.2f}")
    if safe_model.last_hse > 0:
         print(f"Max HSE during generation: {safe_model.last_hse:.2f}")

    print("\n--- Generating WITHOUT EigenTruth ---")
    safe_model.detach_probe()
    outputs_unsafe = safe_model.generate(
        **inputs, max_new_tokens=10, do_sample=False
    )
    print("Output:", tokenizer.decode(outputs_unsafe[0], skip_special_tokens=True))

if __name__ == "__main__":
    import logging
    logging.getLogger("eigentruth").setLevel(logging.INFO)
    main()

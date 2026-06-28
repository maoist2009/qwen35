import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE = "Qwen/Qwen3.5-4B-Instruct"
ADAPTER = "ArlenIvan/realmarx"

def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        BASE,
        torch_dtype=torch.float16,
        device_map="cpu"
    )

    # 尝试 LoRA merge
    try:
        model = PeftModel.from_pretrained(model, ADAPTER)
        model = model.merge_and_unload()
        print("LoRA merged")
    except Exception as e:
        print("No LoRA, fallback full weights:", e)

    tok = AutoTokenizer.from_pretrained(BASE)

    return model, tok

if __name__ == "__main__":
    load_model()

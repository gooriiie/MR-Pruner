import argparse
import torch
from profiler_utils import profile_once, ddp_avg
from PIL import Image

# === (예시) 여러분 환경의 전처리/모델 로더 ===
# - HuggingFace 계열 LLaVA-HF면 AutoProcessor/AutoModel을 쓰세요.
# - 원본 LLaVA(haotian-liu 계열)면 repo의 builder 유틸을 사용하세요.
# 아래는 "형태"만 제시합니다.

def load_model_and_processors(model_id: str, dtype=torch.float16, device="cuda"):
    """
    환경에 맞게 수정하세요.
    """
    from transformers import AutoTokenizer

    # (A) 원본 LLaVA 계열일 때 (예시)
    # from llava.model.builder import load_pretrained_model
    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     model_path=model_id, model_base=None, device=device, dtype=dtype
    # )

    # (B) HF 포팅 버전일 때 (예시)
    # from transformers import AutoProcessor, AutoModelForCausalLM
    # processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id, torch_dtype=dtype, device_map="auto", trust_remote_code=True
    # )

    # --- 데모용 더미(여러분 환경에 맞춰 (A)/(B)로 바꾸세요) ---
    raise NotImplementedError("여러분의 LLaVA-NeXT 로더로 대체하세요.")

def prepare_batch(processor, tokenizer, image_path: str, prompt: str, device="cuda"):
    """
    환경에 맞게 수정:
    - LLaVA류는 보통 (images=Tensor, input_ids=Tensor, attention_mask=Tensor, ...) 형태.
    - HF 포팅이면 processor가 text+image를 한 번에 만들어줍니다.
    """
    img = Image.open(image_path).convert("RGB")

    # (예시) HF 포팅 버전:
    # inputs = processor(text=prompt, images=img, return_tensors="pt").to(device)
    # return inputs

    # (예시) 원본 LLaVA:
    # image_tensor = image_processor.preprocess(img, return_tensors="pt")["pixel_values"].to(device, dtype=torch.float16)
    # conv = tokenizer.apply_chat_template([{"role":"user","content":prompt}], tokenize=False, add_generation_prompt=True)
    # tok = tokenizer(conv, return_tensors="pt").to(device)
    # return {"input_ids": tok.input_ids, "attention_mask": tok.attention_mask, "images": image_tensor}

    raise NotImplementedError("여러분의 processor/tokenizer로 실제 배치를 만드세요.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True,
                    help="예: lmms-lab/llama3-llava-next-8b (여러분 환경에 맞게)")
    ap.add_argument("--image", type=str, required=True)
    ap.add_argument("--prompt", type=str, default="Describe the image.")
    ap.add_argument("--max_new_tokens", type=int, default=16)
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    args = ap.parse_args()

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]

    # 1) 모델/프로세서 로드
    # tokenizer, model, processor = load_model_and_processors(args.model_id, dtype=dtype)
    raise NotImplementedError("load_model_and_processors 구현 필요")

    # 2) 배치 준비 (images + text)
    # batch = prepare_batch(processor, tokenizer, args.image, args.prompt)
    raise NotImplementedError("prepare_batch 구현 필요")

    # ===============================
    # (A) 프리필 FLOPs 측정
    # ===============================
    # 일반적으로 프리필은 cache/past_key_values 없이 전체 prompt(+비전)로 한 번 forward
    # - 모델 구현에 따라 forward 키워드는 다릅니다.
    # - 예시: outputs = model(**batch, use_cache=True)
    prefill_inputs = dict(batch)
    prefill_inputs.update({"use_cache": True})

    prefill_res = profile_once(model, fwd_args=(), fwd_kwargs=prefill_inputs,
                               detailed=True, print_profile=True)
    prefill_flops = prefill_res["flops"]

    # ===============================
    # (B) 디코드 1스텝 FLOPs 측정
    # ===============================
    # past_key_values를 얻기 위해 프리필 한 번 실제로 실행
    with torch.no_grad():
        o = model(**prefill_inputs)
        past = o.past_key_values if hasattr(o, "past_key_values") else o.get("past_key_values", None)

    # 마지막 토큰 1개만 넣어 다음 토큰 1스텝 디코드
    # (배치 텍스트 길이 T라면 input_ids[:, -1:] 또는 생성 중 토큰 1개)
    # 아래는 예시; 실제 키 이름/구조에 맞춰 수정하세요.
    one_step = {
        "input_ids": batch["input_ids"][:, -1:].contiguous(),
        "attention_mask": torch.cat([batch["attention_mask"], torch.ones_like(batch["attention_mask"][:, :1])], dim=1),
        "use_cache": True,
        "past_key_values": past,
    }
    if "images" in batch:
        # 많은 구현이 디코드 단계에서 images를 다시 요구하지 않지만,
        # 구현에 따라 필요할 수 있어 안전하게 전달
        one_step["images"] = batch["images"]

    decode_res = profile_once(model, fwd_args=(), fwd_kwargs=one_step,
                              detailed=True, print_profile=True)
    decode_flops = decode_res["flops"]

    # ===============================
    # (C) 총 FLOPs 근사
    # ===============================
    max_new = args.max_new_tokens
    total_flops_est = prefill_flops + max_new * decode_flops

    # 멀티 GPU일 경우 평균/합계 전략 선택 (여기선 평균 예시)
    prefill_flops_avg = ddp_avg(prefill_flops)
    decode_flops_avg  = ddp_avg(decode_flops)
    total_flops_avg   = ddp_avg(total_flops_est)

    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    print("\n=== FLOPs (estimates) ===")
    print(f"Prefill FLOPs: {prefill_flops_avg:,.0f}")
    print(f"Decode 1-step FLOPs: {decode_flops_avg:,.0f}")
    print(f"Total FLOPs (~{max_new} toks): {total_flops_avg:,.0f}")
    print("Note: 실제 생성 토큰 수로 곱하면 더 정확합니다.")

if __name__ == "__main__":
    main()

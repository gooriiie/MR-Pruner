import contextlib
import torch

# DeepSpeed FlopsProfiler import (버전별 호환)
try:
    from deepspeed.utils.flops_profiler import get_model_profile
except Exception:
    from deepspeed.profiling.flops_profiler import get_model_profile  # fallback

def is_dist():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def ddp_sum(v: float) -> float:
    if not is_dist():
        return v
    t = torch.tensor([v], dtype=torch.float64, device="cuda")
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
    return t.item()

def ddp_avg(v: float) -> float:
    if not is_dist():
        return v
    world = torch.distributed.get_world_size()
    return ddp_sum(v) / world

@contextlib.contextmanager
def cuda_sync():
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    yield
    try:
        torch.cuda.synchronize()
    except Exception:
        pass

def profile_once(model, fwd_args: tuple = (), fwd_kwargs: dict = None,
                 detailed: bool = True, print_profile: bool = True):
    """
    get_model_profile는 "모델.forward" 1번을 프로파일합니다.
    fwd_args/fwd_kwargs는 forward에 그대로 들어갈 인자입니다.
    반환: dict 형태로 macs, flops, params, time_s
    """
    if fwd_kwargs is None:
        fwd_kwargs = {}

    model.eval()
    with torch.no_grad(), cuda_sync():
        # warmup 1회, as_string=False로 숫자 그대로 받기
        res = get_model_profile(
            model=model,
            input_args=fwd_args,
            kwargs=fwd_kwargs,
            print_profile=print_profile,
            detailed=detailed,
            module_depth=-1,
            top_modules=3,
            warm_up=1,
            as_string=False,
        )
        # DeepSpeed 버전에 따라 반환 순서가 다를 수 있으므로 안전하게 dict로 정리
        # 관행적으로 (flops, macs, params) 또는 (macs, params, flops) 형태가 섞여 있어 아래처럼 처리
        flops = res[0] if isinstance(res, (list, tuple)) else getattr(res, "flops", float("nan"))
        macs  = res[1] if isinstance(res, (list, tuple)) else getattr(res, "macs", float("nan"))
        params= res[2] if isinstance(res, (list, tuple)) else getattr(res, "params", float("nan"))

    # 일부 DS 버전은 위 값이 바뀌어 있을 수 있음 -> 큰 값이 FLOPs인 경우가 보통.
    # 필요시 사용자 환경에서 print_profile=True로 출력 비교 권장.
    return {
        "flops": float(flops),
        "macs": float(macs),
        "params": float(params),
    }
# MPS Compatibility Patches

## Problem
heartlib was written with CUDA assumptions that break on Apple Silicon MPS backend.

## Fixes Applied

### 1. transformer.py — line 426
**File:** `src/heartlib/heartcodec/models/transformer.py`

**Problem:** `.type(timesteps.type())` fails on MPS because `torch.dtype` is not 
callable and MPS tensor type strings are not recognised.

**Fix:** Replace with `.to(timesteps.dtype)` — backend agnostic, works on MPS, 
CUDA, and CPU.

### 2. music_generation.py — line 342
**File:** `src/heartlib/pipelines/music_generation.py`

**Problem:** `torchaudio.save()` now requires `torchcodec` which in turn requires 
FFmpeg shared libraries at specific versioned paths. These are not available in a 
standard Homebrew FFmpeg installation on macOS.

**Fix:** Replace `torchaudio.save()` with `soundfile.write()`. Requires 
`soundfile` installed (`uv pip install soundfile`). Note the transpose `.T` — 
soundfile expects `[samples, channels]`, the model outputs `[channels, samples]`.

## Dependencies
```bash
uv pip install soundfile
```

## Notes
- Tested on Apple M2 Max 96GB, macOS, Python 3.14, PyTorch 2.10.0
- `triton` import error at startup is harmless — triton is CUDA only, MPS ignores it
- Generation stops early (around 1600-1977 steps of 3000 max) — this is normal, 
  model converges before the ceiling

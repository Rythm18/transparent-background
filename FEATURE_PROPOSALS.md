Feature proposals (medium-hard, ~2 hours each)

1) Mixed-precision inference (auto/fp16/bf16/fp32)
- Value: 1.3–2x throughput and reduced VRAM on CUDA/MPS with minimal quality loss; keeps CPU safe defaults.
- Implementation outline:
  - Remover: add precision option and wrap model(x) under torch.autocast for CUDA/MPS; safe fallback to fp32 on CPU.
  - CLI/GUI: new flag/dropdown --precision {auto,fp32,fp16,bf16}.
- Files to touch: transparent_background/Remover.py, transparent_background/utils.py, transparent_background/gui.py, usage.py.
- Notes: ensure TorchScript path still works; keep matting/refinement in fp32.

2) Temporal consistency for video/webcam (optical-flow mask stabilization)
- Value: Reduces flicker on video outputs; smoother masks and boundaries.
- Implementation outline:
  - entry_point: keep prev frame + mask; compute optical flow (cv2.DISOpticalFlow or Farneback) to warp previous mask; blend with current mask via alpha (pred_smooth = a*curr + (1-a)*warp(prev)).
  - Expose --temporal-smooth [0.0–1.0]; apply for Video/Webcam.
  - Optionally add --temporal-method {ema,flow} to fall back when flow fails.
- Files to touch: transparent_background/Remover.py (return raw mask or add helper), transparent_background/utils.py (args), transparent_background/Remover.py (loop), transparent_background/gui.py (slider).
- Notes: Keep overhead minimal; handle frame size changes and flow errors gracefully.

3) Advanced background blur “bokeh” with adjustable strength
- Value: DSLR-like background blur without haloing; more pleasing composites.
- Implementation outline:
  - New type 'bokeh': compute distance transform on background (1 - pred); map distance to blur level; blend a small set of pre-blurred images (e.g., sigma in [5, 15, 30]) according to distance bins; feather edges.
  - Add --bokeh-strength (or reuse --blur-strength) and optional --feather.
- Files to touch: transparent_background/Remover.py (new type), transparent_background/utils.py (args), transparent_background/gui.py (slider).
- Notes: Keep number of blur scales small to control runtime; use mask feathering to avoid edge artifacts.

4) Remote input support (HTTP/HTTPS URLs)
- Value: Process images/videos directly from the web; easier demos and automation.
- Implementation outline:
  - entry_point: detect URLs; images: fetch via requests -> PIL.Image.open(BytesIO); videos: stream to NamedTemporaryFile then use VideoLoader.
  - Allow --source to be a URL; derive output name from URL basename.
  - Add requests to setup.py install_requires.
- Files to touch: transparent_background/Remover.py (entry_point), transparent_background/utils.py (helpers/args), setup.py, transparent_background/gui.py (allow URL input).
- Notes: Handle missing extensions via content-type; timeout/retry handling; ensure temp files are cleaned up.

5) Background image placement modes + feathered composition
- Value: Better background replacement UX: fit/fill/stretch/tile and alignment; softer edges reduce halos.
- Implementation outline:
  - In the background-image path (type endswith image): support --bg-mode {fit,fill,stretch,tile} and --bg-align {center,top-left,top,top-right,left,right,bottom-left,bottom,bottom-right}.
  - Implement aspect-aware resize + crop (fill) or letterbox (fit); add tiling option.
  - Add --feather-radius to soften mask edges before composition.
- Files to touch: transparent_background/Remover.py (background branch), transparent_background/utils.py (args), transparent_background/gui.py (drop-downs and slider).
- Notes: Premultiply alpha when saving RGBA to reduce color spill; cache resized background across frames.

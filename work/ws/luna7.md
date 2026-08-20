Write ONE new file: D:\hackathon\sun\work\pipeline7.py . Do NOT modify work/pipeline.py. No prose.
This is submission-pack code: it must reproduce training-time inference EXACTLY. A past bug shipped a
pack whose optical-flow input was 1.368x too small because a constant was not carried over; that class
of error must not repeat.

Start from an exact copy of work/pipeline.py (read it fully first; it is the notebook body that the
submission runs, and it already contains a 2-stream flow CNN inference path `cnn_predict` with arch
"v6" matching work/cnn6.py, plus the Farneback flow computation).

ADD support for two new component architectures, selected per tag, without changing any existing path:

1. arch "v7" = work/cnn7.py. Same as v6 plus a scalar block: the head input is
   torch.cat([wind_feats(40), scalars(839)]) and `wind` in the Net is
   nn.Sequential(nn.Linear(40+839,128), SiLU, Linear(128,128)).
   The 839 scalars come from `sample_feats(split, inputs)` (work/feats.py, already vendored into
   pipeline.py if present - check; if not, vendor the same code path the pack already uses for the
   image-GBDT features). They must be standardised with the TRAIN statistics saved by training at
   /home/jovyan/work/<TAG>_stats.npz keys smu, ssd, fstd, then clipped to [-8, 8] exactly as cnn7.py does.
   The pack must read those statistics from the model artifact, not recompute them.

2. arch "v9" = work/cnn9.py. Same as v6 plus a Stonyhurst remap applied to the 4-channel
   (image, flow) tensor immediately before the encoder:
     a = linspace(-72, 72, PX) degrees; lat, lon = meshgrid(a, a, indexing="ij")
     gx = 0.94*cos(lat)*sin(lon); gy = 0.94*sin(lat); grid = stack([gx, gy], -1)[None]
     x = F.grid_sample(x.flatten(0,1), grid.expand(b*t,-1,-1,-1), mode="bilinear",
                       padding_mode="zeros", align_corners=False).view(b,t,...)
   Note v9 weights have the SAME state_dict keys as v6, so the arch must be chosen by tag, never inferred
   from the checkpoint.

REQUIREMENTS:
- Do not change the existing v6 path's numerical behaviour at all.
- The flow std constant (fstd) must come from the packed artifact for every arch, defaulting to nothing:
  if a tag's fstd is missing, raise instead of silently using 1.0.
- Keep the same public entry points and function names pipeline.py exposes, so work/stage.py and
  work/pack_seedbag.py keep working when pointed at this file.
- Add a short module-level comment listing exactly which constants each arch needs from the artifact.

ACCEPTANCE (run and paste):
  python -c "import ast;ast.parse(open(r'D:\hackathon\sun\work\pipeline7.py',encoding='utf-8').read());print('OK')"
  and print the diff summary: `git -C D:\hackathon\sun diff --no-index --stat work/pipeline.py work/pipeline7.py`

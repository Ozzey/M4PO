# Paper results and supplementary media

## Source and scope

The README's results were transcribed from the supplied
`Papers/m4po/main.tex`, and checked against the
[paper already in this repository](M4PO_CoRL_26-1.pdf). The common values in
Tables IV, V, VI, and VIII are reproduced without changing the experimental
setting or merging them with the repository's MMBench run.

These are manuscript-reported results. The source describes on-policy learning
with a value-discrepancy exploration bonus.

The manuscript reports three seeds and 100 evaluation episodes per
task–embodiment pair for simulation. Its main and expanded Arena benchmarks
are IsaacLab manipulation settings, not NEWT's 200-task MMBench.
Hardware counts cover five conditions with 20 trials each. G1 adaptation
occurs in simulation; hardware evaluations use fixed policy parameters.

## Result figures

The following PNG files were copied unchanged from
`Papers/m4po/Assets/Results/` into [assets/results](../assets/results).
They are supplied manuscript artwork, not reconstructed plots from local
training logs. No intermediate measurements or confidence intervals were
generated for this README.

| File | Content | SHA-256 |
| --- | --- | --- |
| Multi-Task-Multi-Embodiment.png | Average and worst-embodiment success | `4ad58b73a3676539e9a2b2d416df1ff150d4b8b6ddfecff13c7f9d900c9f6281` |
| Multi-Task-Fixed-Base.png | Multi-task fixed-base curves | `e6e50a4f4ce133f59844bf906d174f057d346a87cbe46926f906c7bfb3477a02` |
| Multi-Task-Humanoid.png | Multi-task humanoid curves | `c6e62fad9980ab8ea0f599da0aa45eacf50ab08667683b433190b74714b0c500` |
| Ablation.png | Model scale, batch size, and language conditioning | `4e139a0b25f44b285ad1de2d9ebbf9f806045960485ceee0f690799207caa9f9` |

The existing [architecture SVG](../assets/architectures/M4PO-Architecture.svg)
is displayed directly below the README title and was not modified. It depicts
the paper's on-policy architecture.

## Supplementary videos

Seven MP4s were copied unchanged from `Papers/m4po/Assets/Videos/` into
[assets/videos/sim](../assets/videos/sim), preserving their original filenames:

- `g1_static_lift_metallic_stitched.mp4`
- `ur5_stack_5eps_stitched.mp4`
- `franka_lift_5eps_scene_closeup.mp4`
- `franka_pick_place_5eps_scene_closeup.mp4`
- `franka_insertion_5eps_scene_closeup.mp4`
- `franka_stack_5eps_scene_closeup.mp4`
- `franka_cabinet_v2_5eps_scene_closeup.mp4`

The README embeds looping GIFs in [assets/videos/gifs](../assets/videos/gifs):

| GIF | Source | Segment |
| --- | --- | --- |
| `g1_lifting.gif` | `g1_static_lift_metallic_stitched.mp4` | Full clip, approximately 4.33 seconds |
| `ur5_stacking.gif` | `ur5_stack_5eps_stitched.mp4` | First 8 seconds |
| `franka_cabinet.gif` | `franka_cabinet_v2_5eps_scene_closeup.mp4` | First 8 seconds |

The previews preserve normal playback speed and the original views, with
reduced resolution, frame rate, and color palette for inline display. They
are video conversions, not generated illustrations. Clicking a GIF opens
the unchanged full-length MP4. The previously extracted static poster frames
remain in `assets/videos/previews/` but are not used in the README.

The clips show rendered simulation scenes. Their filenames and media metadata
do not establish the producing checkpoint, training algorithm, evaluation
seed, or trial-selection procedure. They are presented as supplied
supplementary demonstrations rather than a record of the quantitative
evaluation trials. No hardware video was added.

## Manuscript-version caveats

The supplied paper-directory README documents inconsistencies in its source
material. The repository PDF also contains later revisions. The project README
uses their consistent headline tables and avoids claims affected by these
differences:

- **Success heatmap:** supplied heatmap cells differ from the category tables
  (for example, M4PO dexterous grasping is 90 in the heatmap and 95 in the
  table). The heatmap is not included.
- **Dexterous trajectories:** the supplied provenance explicitly labels
  intermediate curves as estimated, anchored to final rates of
  55.3 / 50.2 / 30.2 / 25.4 percent. Although the repository PDF uses different
  wording, these curves must not be treated as measured trajectories. They
  are not included; the reported final table values are retained.
- **Component ablations:** the paper-directory numerical ablation table,
  its prose deltas, and the repository PDF disagree. That table and those
  deltas are not reproduced. The copied `Ablation.png` is a separate
  model-scale, batch-size, and language-conditioning figure.
- **Hardware conclusion:** the paper-directory conclusion's approximate
  80% G1 figure is outdated relative to its hardware table. Both the table
  and repository PDF report 18/20 G1 successes and 85/100 overall; those
  counts are used.
- **Compute claims:** the versions differ in hardware and runtime figures,
  including whether wall-times are estimated lower bounds or measurements.
  No runtime or resource-superiority claims are reproduced.

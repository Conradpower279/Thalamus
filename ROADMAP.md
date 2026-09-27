# Roadmap

## Phase 1 — non-lethal species logging (current)

A Flutter app that identifies fish species in real time from a phone camera
while swimming or diving, and logs sightings. Identification only: no
measurement, no catch, no legal logic.

Phase 1 is where the classifier earns the trust phase 2 will depend on. The work
is:

- [x] Training pipeline: MobileNet via timm, two-stage transfer learning,
      underwater-oriented augmentation, ONNX export
- [x] A real dataset — built from the iNaturalist Open Dataset (Florida
      ray-finned fish, ~380 species after capping) via
      `scripts/download_florida_fish.py` and `scripts/cap_species.py`
- [x] Group-aware splits that do not leak between train and val — filenames
      carry the observation UUID, and both `split_data.py` and `cap_species.py`
      keep whole observations inside one split
- [x] Regularisation stack (config-gated): random-resized-crop, CoarseDropout,
      label smoothing, weight decay, MixUp/CutMix, and a class-balanced sampler.
      Lesson learned: add levers one at a time — the full stack underfit
- [x] Class-imbalance handling (weighted sampler) and per-class metrics
      (macro accuracy reported alongside top-1/top-5)
- [x] Held-out test evaluation — the `test/` split is scored at the end and
      written to `test_metrics.json` (top-1/top-5/macro)
- [~] Settle the backbone: comparing MobileNetV3-Small, V4-Conv-Small,
      V4-Conv-Medium and V4-Hybrid-Medium at 224 and 448 on the real set, same
      light recipe (`scripts/run_448_sweep.py`, `scripts/benchmark_backbones.py`).
      Best held-out so far: hybrid_medium @224 = 0.662 top-1 / 0.848 top-5
- [x] Fish detector (started early, normally phase 2): single-class YOLO
      localiser to crop fish before classification —
      `scripts/autolabel_fish.py` + `scripts/yolo_train.py`, first run
      mAP50 ≈ 0.88. Species detection with per-species boxes remains phase 2
- [ ] Open-set handling — a calibrated confidence threshold, validated against
      real negative footage (kelp, sand, debris, untrained species), so the app
      can decline to guess
- [ ] Select checkpoints on macro (not validation top-1), so training optimises
      for rare species rather than just the common ones
- [ ] Quantized TFLite model with measured post-quantization accuracy
- [ ] Flutter app: camera stream, background-isolate inference, sighting log
- [ ] On-device latency and battery measurements under real dive conditions

Everything in the "Known limitations" section of [README.md](README.md) belongs
to this phase.

---

## Phase 2 — spearfishing compliance variant

Same classifier, considerably more responsibility: before a fish is taken, the
system identifies the species, estimates its length, and checks that against
local regulations. A wrong answer here is not a bad log entry — it is an illegal
take, and possibly a protected species.

### 2.1 LiDAR integration

Depth sensing to recover real-world scale, which a single camera cannot provide.

- Target sensors: iPhone Pro / iPad Pro LiDAR for the phone build; a solid-state
  LiDAR or active stereo module for the dedicated hardware build
- Time synchronisation between the depth frame and the RGB frame — at
  swimming speed, a few tens of milliseconds of skew is centimetres of error
- Underwater considerations that do not arise on land:
  - most consumer LiDAR is designed for air; refraction at the housing port
    shifts apparent distance and needs calibrating out
  - water absorbs strongly in the near-IR band these sensors use, which limits
    usable range and degrades with turbidity
  - the practical range envelope needs measuring, not assuming, and the system
    must refuse to estimate outside it
- Fallback when depth is unavailable or out of range: report species only, and
  say clearly that no size estimate was possible

### 2.2 Size estimation

- Upgrade from whole-image classification to **detection** (bounding box or
  segmentation mask) — a classifier alone gives no extent to measure. A
  single-class "fish" detector already exists from phase 1
  (`scripts/yolo_train.py`); phase 2 needs it to also give the extent to measure
  from, and likely per-species or oriented boxes rather than one axis-aligned
  "fish" box
- Convert box extent plus distance into length, using camera intrinsics and an
  underwater-refraction correction
- Prefer an oriented box or a mask over an axis-aligned box: a fish at an angle
  to the camera has an axis-aligned box much longer than the fish
- Handle foreshortening — a fish swimming toward the camera reads short
- Regulations specify *which* length (total, fork, standard); the model has to
  estimate the one the local rule names, which differs by region
- Ship a per-estimate confidence interval, not a bare number, and treat the
  conservative end of the interval as the one that counts against a limit
- Validate against measured fish across the full size range, in water, at
  varying distances and angles

### 2.3 Regional rules engine

A data-driven rules layer, versioned separately from the app so regulation
changes ship without an app release.

Sketch of the schema:

```jsonc
{
  "region_id": "us-ca-marine",
  "region_name": "California Marine District",
  "authority": "California Department of Fish and Wildlife",
  "source_url": "https://wildlife.ca.gov/...",
  "effective_from": "2026-03-01",
  "effective_until": null,
  "last_verified": "2026-02-14",
  "schema_version": 1,
  "species": [
    {
      "species_id": "sebastes_melanops",
      "common_name": "Black rockfish",
      "measurement": "total_length",
      "units": "cm",
      "min_length": 25.4,
      "max_length": null,
      "bag_limit": 5,
      "aggregate_group": "rockfish_complex",
      "seasons": [
        { "opens": "2026-05-01", "closes": "2026-12-31" }
      ],
      "gear_restrictions": ["no_scuba_take"],
      "protected": false,
      "notes": "Counts toward the 10-fish rockfish aggregate limit."
    }
  ],
  "aggregate_limits": [
    { "group_id": "rockfish_complex", "bag_limit": 10 }
  ]
}
```

Design constraints:

- **Fail closed.** Unknown species, expired rule set, no GPS fix, or a
  low-confidence identification → advise against taking, never in favour
- **Aggregate limits are not per-species limits.** Many jurisdictions cap a
  whole complex as well as individual species; the schema has to express both
- **Rules expire.** Every rule set carries `effective_from`, `effective_until`
  and `last_verified`; stale data must warn loudly rather than answer quietly
- **Region resolution** from GPS, with manual override — boundaries are
  jurisdictional, not geographic, and a fix at the surface may be some distance
  from where the fish was seen
- **Provenance for every rule**, so a user can check the actual regulation
- Bag-limit tracking needs per-session state and a clear model of what a
  "session" is (a day, a trip, a person)

**This is legal-adjacent output.** The app must present itself as an aid and not
an authority, keep the official source one tap away, and be unambiguous that the
diver carries the legal responsibility. Legal review before release.

### 2.4 Jetson hardware target

For a dedicated dive-housing device rather than a phone.

- Target: Jetson Orin Nano, TensorRT inference, JetPack toolchain
- **Benchmark before committing to MobileNetV3.** It is a good fit for phone
  NPUs but a poor one for TensorRT: depthwise convolutions, hard-swish and
  squeeze-excite blocks are memory-bandwidth-bound and underuse the tensor
  cores, so measured latency is often worse than the FLOP count suggests.
  ResNet18 or `efficientnet_lite0` frequently win on this hardware
- `configs/mobilenetv4.yaml` already removes two of those three problems —
  `mobilenetv4_conv_small` has no hard-swish and no squeeze-excite, and V4 was
  NAS-optimised across GPU and accelerator targets rather than mobile CPU alone.
  If phase 1 settles on it, this concern largely resolves itself. Depthwise
  convolutions remain, so still measure on the Jetson rather than assuming
- INT8 with TensorRT calibration, plus accuracy measured post-quantization
- Power and thermal budget inside a sealed housing, where there is no airflow
  and the only heat path is conduction to the water
- Adds JetPack/TensorRT/LiDAR SDK dependencies — deliberately absent from
  phase 1's `requirements.txt` so the training environment stays laptop-friendly

### 2.5 Bluetooth HUD output

- Stream species, estimated length, and a take/no-take verdict to a mask-mounted
  or wrist display over BLE
- Design for a diver's actual reading conditions: gloved, cold, task-loaded, low
  visibility. A glanceable verdict with a colour and a symbol, not a sentence
- BLE GATT profile for the verdict payload, versioned so display firmware and
  app can update independently
- Latency budget end to end — a verdict that arrives after the shot is worthless
- Power budget for a dive-length session on a small battery

---

## Open questions

- The single-class fish detector is bootstrapped from auto-labels (YOLO-World
  prompted with "fish" + whole-image fallback). For phase 2, is that box quality
  good enough to measure from, or does size estimation need hand-labelled /
  oriented boxes and a segmentation mask?
- Is phone LiDAR accurate enough underwater for legally meaningful measurement,
  or is dedicated hardware mandatory for the compliance variant?
- What is the confidence threshold above which a size estimate may be shown at
  all, and who decides it?
- How are regional rule sets sourced and kept current — manual curation,
  scraping, or a partnership with a fisheries authority?
- Does the compliance variant ship as a separate app, or as a mode inside the
  phase-1 app? Separate keeps the logging app free of legal surface area.

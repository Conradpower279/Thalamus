# Roadmap

## Phase 1 — non-lethal species logging (current)

A Flutter app that identifies fish species in real time from a phone camera
while swimming or diving, and logs sightings. Identification only: no
measurement, no catch, no legal logic.

Phase 1 is where the classifier earns the trust phase 2 will depend on. The work
is:

- [x] Training pipeline: MobileNet via timm, two-stage transfer learning,
      underwater-oriented augmentation, ONNX export
- [ ] Settle the backbone: MobileNetV3-Small vs MobileNetV4-Conv-Small, compared
      on real fish images rather than ImageNet ranking (both configs ship; see
      `scripts/benchmark_backbones.py`)
- [x] Repository structure, config, split utility, TFLite export, tests, CI
- [ ] A real dataset, with group-aware splits that do not leak between train
      and val
- [ ] Class-imbalance handling and per-class metrics, so rare species are
      actually measured rather than averaged away
- [ ] Open-set handling — a calibrated confidence threshold, validated against
      real negative footage (kelp, sand, debris, untrained species), so the app
      can decline to guess
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
  segmentation mask) — a classifier alone gives no extent to measure
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

- Where does the fish detector come from — retrain from scratch, or fine-tune an
  existing marine detection model?
- Is phone LiDAR accurate enough underwater for legally meaningful measurement,
  or is dedicated hardware mandatory for the compliance variant?
- What is the confidence threshold above which a size estimate may be shown at
  all, and who decides it?
- How are regional rule sets sourced and kept current — manual curation,
  scraping, or a partnership with a fisheries authority?
- Does the compliance variant ship as a separate app, or as a mode inside the
  phase-1 app? Separate keeps the logging app free of legal surface area.

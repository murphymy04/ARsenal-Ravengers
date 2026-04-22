---
created: 2026-04-22T20:07:04.417Z
title: Tighten external-vision refactor
area: general
files:
  - pipeline/live.py
  - pipeline/diarization.py
  - pipeline/retrieval.py
  - dashboard.py
---

## Problem

The recent refactor split face detection/tracking into an "external vision" path so `GlassesServer` can own its own vision thread. It works, but several rough edges were left behind:

1. **Ad-hoc tuple protocol.** `camera.frames()` now yields either `frame` or `(frame, vision_result)`. Consumers unpack with `isinstance(item, tuple)`. See `pipeline/live.py:267` and `dashboard.py:355`. Two call sites duplicate the same unpack block.
2. **Duplicated wiring in the `--glasses` CLI branch.** `pipeline/live.py:__main__` manually constructs `FaceDetector`, `FaceTracker`, `RetrievalWorker`, and the retrieval event queue before calling `driver.run(external_vision=True)`. The in-band path has `LivePipelineDriver.run()` do all of that internally. The two setups drifted in subtle ways (e.g., worker lifecycle, queue ownership).
3. **Dashboard mirrors the same unpack logic.** `dashboard.py:process_video` has its own `(frame, vision_result)` loop. If the protocol changes, two files change.
4. **RetrievalDispatcher has no tests.** It's now a standalone class (`pipeline/retrieval.py:60`) with pure logic — cooldown, min-frame debounce, active-set eviction. Easy to unit test; nothing covers it.

## Solution

- Introduce a typed result (dataclass or NamedTuple) for vision output; have `camera.frames()` always yield that shape (or wrap the external-vision iterator so the consumer doesn't branch).
- Push the `--glasses` setup into `LivePipelineDriver.run()` behind the `external_vision` flag, or into a small factory. Goal: one owner of `FaceDetector` / `FaceTracker` / `RetrievalWorker` lifecycle.
- Extract the frame-loop body (unpack → `process_frame` → overlay/flush) from both `live.py` and `dashboard.py` into a shared helper, or have the dashboard consume `LivePipelineDriver` instead of rebuilding it.
- Add unit tests for `RetrievalDispatcher`: min-frame gating, cooldown, eviction on dead tracks.

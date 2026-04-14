# Glasses stream adapter — deferred issues

Known failure modes not handled in the initial implementation. Add handling
if/when they bite us in practice.

## Video stall heartbeat

If the video stream stalls for several seconds while audio keeps flowing, the
pairing loop blocks and no pairs are emitted. Audio piles up in the receiver,
VAD never sees it, and the transcription window never flushes.

Fix when needed: in the pairing loop, if no new frame for >N ms (200ms is
reasonable), emit a synthetic pair reusing the last frame's pixels with an
advanced timestamp. Audio keeps dripping through VAD; face detection re-runs
on stale pixels (harmless — same faces get re-detected).

## Audio wait timeout

The pairing loop blocks indefinitely waiting for audio to cover a given
frame's timestamp. If audio is disconnected mid-session this hangs the
pipeline until audio reconnects.

Fix when needed: bounded wait (e.g. 500ms). On timeout, drop the frame or
emit with zero-padded audio. For now we assume TCP audio stays connected.

## Backpressure tuning

Queue caps and drop-oldest behaviour are set to round-number defaults. If the
pipeline consistently falls behind under load, tune `MAX_PAIRS` and consider
dropping frames-only (keeping audio intact) as an alternative policy.

## Vision stride

`VISION_STRIDE` counts *emitted* pairs, not source frames. If the glasses
already drop 50% of frames, stride=2 becomes effectively stride=4. For live
glasses input, default to stride=1 and rely on network loss for any
down-sampling that happens.

## Pairing prebuffer + paced emission

The INMO Air 3 delivers audio and video on wildly different cadences. Video
arrives over UDP as a steady ~30 fps stream; audio arrives over TCP in large
batched chunks, roughly one 5-second buffer at a time. A naive pairing loop
that holds each frame until audio covers `[prev_frame_ts, this_frame_ts]`
inherits the audio's burstiness: frames pile up waiting for the next audio
chunk, then flush together as ~150 pairs the moment it lands. Downstream
consumers see bursts separated by multi-second gaps instead of real-time
cadence.

That bursty emission breaks the diarization pipeline's assumptions. The face
tracker relies on `FACE_MAX_MOVE_PX` to maintain track continuity across
roughly monotonic frames, and identity clustering needs `MIN_SIGHTINGS_TO_CLUSTER`
sightings on a stable track before promoting it to a labeled person. When a
burst arrives, consecutive frames look near-simultaneous; after the 5s gap,
the next burst's face positions jump far enough that the tracker allocates
fresh track IDs instead of reusing existing ones. Clusters never accumulate
enough sightings, and the HUD shows `track_N` placeholders where recognized
names should be.

`PairingLoop` in `glasses_adapter.py` decouples the two concerns with a
builder thread and an emitter thread. The builder pairs frames to audio as
fast as the network allows and appends to an internal staging deque, capped
at `GLASSES_MAX_STAGING_SECONDS` so it cannot grow unbounded during a long
audio stall. The emitter waits until staging spans `GLASSES_PREBUFFER_SECONDS`
of glasses-clock content, then releases pairs to the public queue paced
against wall clock using each pair's glasses timestamp, re-anchoring if it
drifts more than a second behind. Consumers pay a one-time startup delay
equal to the prebuffer, then receive a steady 30 pairs/sec regardless of
how the network batches audio.

The streamtest reference client (`serveraudioplaybuffer.py`) sidesteps this
entirely because it never pairs. It shows the newest-available frame via
`get_latest_frame` and plays audio through a PortAudio ring buffer with a
200ms jitter prebuffer; the sound card provides the real-time clock for free.
The ASD pipeline needs synchronized (frame, audio-slice) pairs, so it has to
supply its own pacing clock instead of leaning on the speaker.

Tunables live in `config.py`: `GLASSES_PREBUFFER_SECONDS`,
`GLASSES_MAX_STAGING_SECONDS`, `GLASSES_PAIR_QUEUE_MAX`, and
`GLASSES_SPIN_INTERVAL_SEC`.

# Clip Metadata

Source videos in `data/given/`.

| Clip | Resolution (W×H) | Aspect | FPS | Frames | Duration (s) | Codec | Size (MB) |
|------|------------------|--------|-----|--------|--------------|-------|-----------|
| clip1 | 960 × 540 | 16:9 | 29.97 | 600 | 20.0 | h264 | 3.0 |
| clip2 | 960 × 540 | 16:9 | 30.00 | 495 | 16.5 | h264 | 2.2 |
| clip3 | 352 × 288 | 11:9 (CIF) | 24.80 | 250 | 10.1 | h264 | 0.3 |
| clip4 | 960 × 720 | 4:3 | 6.59 | 182 | 27.6 | h264 | 1.6 |
| clip5 | 640 × 360 | 16:9 | 29.77 | 300 | 10.1 | h264 | 0.4 |
| clip6 | 640 × 360 | 16:9 | 29.75 | 276 | 9.3 | h264 | 0.3 |
| clip7 | 960 × 540 | 16:9 | 6.00 | 108 | 18.0 | h264 | 1.2 |
| clip8 | 960 × 540 | 16:9 | 6.00 | 102 | 17.0 | h264 | 1.1 |

## Notes

- **Resolutions vary** from CIF (352 × 288) up to 960 × 720. Interaction thresholds are
  normalized by car-box diagonal, so they are resolution-independent.
- **Frame rates vary** widely (6–30 fps). All time-based thresholds are specified in
  seconds and converted to per-clip frame counts via each clip's FPS
  (`Config.resolve_for_fps`), so they represent the same real-world durations.
- **Aspect ratios** differ (16:9, 4:3, 11:9). Per-car normalization keeps the
  proximity logic consistent across them.
- All clips are H.264 encoded.

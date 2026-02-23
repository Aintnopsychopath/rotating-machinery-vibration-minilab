# Results (Hair Dryer / Phone Accelerometer)

This repository contains a small end-to-end vibration workflow:
**data acquisition → filtering → FFT/PSD → comparison of operating points → report**.

## Dataset used for sample output (included in `docs/sample-output/`)
- Device: phone MEMS accelerometer (via Phyphox)
- Two operating points: "Speed 1" vs "Speed 2" (hair dryer)
- Recording length: ~30 seconds per run
- Sampling rate: ~421 Hz (Nyquist ~210.5 Hz)

## Key observations (from the sample run)
- Dominant spectral peak in the measured band:
  - Run A: ~182.7 Hz (≈10,972 RPM if interpreted directly)
  - Run B: ~161.3 Hz (≈9,675 RPM if interpreted directly)
- Run B shows clearly higher vibration energy (RMS/PSD level) than Run A.

## Important measurement note (Nyquist / aliasing)
Phones sample relatively slowly compared to industrial vib systems.
If the true rotation frequency is above Nyquist, the measured peak can be a mirrored/aliased component:
`f_true ≈ fs − f_observed`.

In industrial gas turbine work, a tach/keyphasor and calibrated sensors remove this ambiguity.

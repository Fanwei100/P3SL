# P3SL code revision notes

This version was revised to align the implementation with the manuscript description of P3SL.

## Main paper-alignment fixes

1. Added table-based P3SL split/noise optimization helpers in:
   - `DataScientest/P3SLPrivacyOptimization.py`
   - `DataOwner/P3SLPrivacyOptimization.py`

2. Added optional `--p3sl_auto_config` orchestration in `DataScientest/training_coordinator.py` and `DataScientest/Training.py`:
   - server builds `T_sigma[s] = min sigma such that PL(s, sigma) <= T_FSIM`;
   - clients choose split points locally using `alpha_i * FSIM(s_i, sigma_i) + (1 - alpha_i) * E_i_total(s_i)`;
   - clients can keep `alpha_i`, energy tables, and peak-power profiles local.

3. Fixed P3SL aggregation to match Eq. (1):
   - aggregate only across participating client uploads;
   - fill missing layers `s_i+1:smax` with current server-side weights;
   - keep the aggregated `W_1:smax` on the server in P3SL mode.

4. Disabled redistribution of aggregated client-side weights in P3SL mode. The legacy `--uploadaggrigate` flag is still available for baselines, but it is ignored for `--mode P3SL`.

5. Fixed legacy split-point profiling objective to use privacy weight on FSIM and energy weight on total energy:
   - `alpha_i * FSIM + (1 - alpha_i) * E_total`.

6. Fixed Laplace activation noise scaling: a P3SL noise value `sigma` now produces noise with variance `sigma^2` for Laplace noise by setting PyTorch's Laplace scale to `sigma / sqrt(2)`.

7. Rewrote `README.md` to document the paper-aligned P3SL workflow and the remaining legacy/baseline options.

## Notes

The code was syntax-checked with `python -m py_compile` for both `DataScientest` and `DataOwner`. Full end-to-end training still requires the client devices, WebSocket connectivity, datasets, and hardware power/energy profiles.

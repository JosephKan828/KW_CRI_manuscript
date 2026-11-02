# Magnitude-Squared Coherence and Statistical Testing

- **Data:** Daily CERES/ERA5 fields (2006–2017; 5°S–5°N); QR, LW, and SW heating paired with CWV and the first two vertical EOF scores of pressure velocity (W1, W2): nine pairs.
- **Preparation:** Remove time–longitude means from heating/CWV; derive W1/W2 using centered PCA. Retain full meridional fields.
- **Segments:** 90 overlapping 96-day segments, with 48-day overlap; linearly detrend and apply a Hann window.
- **Coherence:** Apply inverse-time/forward-longitude FFTs. Calculate γ²(f,k) = |⟨XY*⟩|² / (⟨|X|²⟩⟨|Y|²⟩), averaging spectra over latitudes and segments before forming the ratio (0 ≤ γ² ≤ 1).
- **Null distribution:** Independently draw 90 source and 90 reference segments with replacement, pair them, and recompute coherence; repeat 1,000 times per variable pair.
- **Decision:** Mark each frequency–wavenumber bin when observed coherence strictly exceeds its own null 99th percentile (nominal 1% upper-tail level).
- **Interpretation:** Approximate pointwise significance; overlapping-segment dependence and multiple testing are not accounted for.

# Magnitude-squared coherence: construction and statistical assessment

This document describes the implementation in
[`coherence_square_stat.ipynb`](../Code/ERA5/coherence_square_stat.ipynb),
reviewed on 2 November 2026, and distinguishes it from the related spectral
notebooks. It provides the mathematical estimator, the exact random-pair
procedure, its interpretation, and text suitable for a manuscript.
Implementation details were checked against the shared Python functions,
saved notebook outputs, and input NetCDF headers and coordinates. The full
spectral and resampling calculations were not rerun for this documentation.

## 1. Variables, inputs, and analysis settings

The statistics notebook evaluates all nine combinations of three heating
fields and three reference fields:

| Role | Code name | Definition |
| --- | --- | --- |
| Source | `LW` | CERES column-integrated longwave radiative heating, W m⁻² |
| Source | `SW` | CERES column-integrated shortwave radiative heating, W m⁻² |
| Source | `QR` | Total radiative heating, `LW + SW` |
| Reference | `CWV` | ERA5 column water vapor, kg m⁻², equivalent to mm |
| Reference | `W1` | Score of the first vertical EOF of ERA5 pressure vertical velocity |
| Reference | `W2` | Score of the second vertical EOF of ERA5 pressure vertical velocity |

Paths below are relative to `/data92/b11209013`:

| Input | File and variable |
| --- | --- |
| LW | `CERES/column_LW_rm3harm.nc`, `column_LW` |
| SW | `CERES/column_SW_rm3harm.nc`, `column_SW` |
| CWV | `ERA5_GRIB/Data/regrid/CWV.nc`, `CWV` |
| Pressure vertical velocity | `ERA5_GRIB/Data/regrid/W.nc`, `W` |

The `W` metadata specify Pa s⁻¹: this is pressure vertical velocity
$\omega$, rather than geometric vertical velocity in m s⁻¹. CWV metadata
specify $(1/g)\int q\,dp$ over 100–1000 hPa. The heating metadata define each
column heating as
$(F_{\mathrm{surface,up}}-F_{\mathrm{surface,down}})
-(F_{\mathrm{TOA,up}}-F_{\mathrm{TOA,down}})$.

The input histories record subtraction of `climatology_3harmonics.nc`;
ERA5 histories also record bilinear remapping to the prepared 1° grid.
These are upstream operations. The coherence notebook does not construct
the harmonic climatology, perform the remapping, or calculate the input
column integrals. The exact harmonic-fitting procedure requires its own
preprocessing documentation.

| Setting | Value in the statistics notebook |
| --- | --- |
| Selected dates | 1 January 2006–31 December 2017; upper bound `2018-01-01` is exclusive |
| Shared record | 4,383 daily samples, as reported in saved output |
| Latitude selection | 5°S–5°N, inclusive; 10 grid centers from −4.5° to +4.5° |
| Longitude grid | 360 centers, 0.5°E–359.5°E at 1° spacing |
| Meridional component | Full fields; no symmetric/antisymmetric decomposition in this notebook |
| Segment length | 96 days |
| Overlap and step | 48 days overlap; 48-day step |
| Number of complete segments | 90, derived from the record length and segment settings |
| Temporal preprocessing | Linear detrending within each segment, followed by a Hann taper |
| Spectral averaging | Equal-weight latitude average, then equal-weight segment average |
| Random-pair replicates | 1,000 per source/reference pair |
| Random generator | `np.random.default_rng(seed=42)`, initialized once before the pair loop |
| Significance criterion | Observed MSC strictly exceeds the bin-specific 99th percentile |
| Multiple-testing adjustment | None |

`load_analysis_fields()` sorts latitude south to north, checks matching
latitude/longitude grids, and intersects the calendars of LW, SW, and every
selected reference. Duplicate dates and gaps in the shared daily record
are rejected. All fields therefore use the same dates. The spectral
validation rejects nonfinite field values rather than imputing missing data.

## 2. Preparing the fields

### Heating and CWV anomalies

For a scalar field $x(t,\phi,\lambda)$, the code removes its joint
time–longitude mean separately at each latitude:

$$
x'(t,\phi,\lambda)=x(t,\phi,\lambda)
-\frac{1}{N_tN_\lambda}\sum_{t,\lambda}x(t,\phi,\lambda).
$$

This is one constant per latitude. It is not removal of the instantaneous
zonal mean or removal of a separate temporal mean at every grid point.
The same operation is applied to LW, SW, QR, and CWV.

### Vertical EOFs and the W1/W2 references

The pressure-level field has shape $(N_t,N_p,N_\phi,N_\lambda)$.
The notebook reshapes it to $(N_p,N_tN_\phi N_\lambda)$ and passes its
transpose to `sklearn.decomposition.PCA(n_components=2)`. Thus pressure
levels are features, and each time–latitude–longitude location is a sample.

Let $q$ index those locations. PCA centers each pressure level across all
locations, giving $\omega_c(p,q)=\omega(p,q)-\langle\omega(p,q)\rangle_q$.
For EOF loading $e_r(p)$, the score is

$$
a_r(q)=\sum_p e_r(p)\,\omega_c(p,q),\qquad r=1,2.
$$

Scores are reshaped to $(N_t,N_\phi,N_\lambda)$. No pressure-thickness
weighting, feature standardization, or whitening is requested. This is
consistent with PCA's default centering without feature scaling.
([scikit-learn PCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html))

The 15 pressure levels in the input are 100, 150, 200, 250, 300, 400, 500,
600, 700, 800, 850, 900, 925, 950, and 1000 hPa. In this notebook, the score
is retained if `eofs[:, i].max() > 0.1` and multiplied by −1 otherwise.
This is an implementation sign rule, not a physical definition of upward
motion. A global sign reversal leaves MSC unchanged, although it changes
cross-spectral phase by 180°. EOFs are fitted once; they are not refitted
inside the resampling loop, so the test is conditional on the fitted modes.

## 3. Constructing the magnitude-squared coherence

### Segmentation, detrending, and tapering

For record length $N$, segment length $L=96$, and step $D=48$, segment
starts are $sD$, with

$$
s=0,\ldots,M-1,\qquad
M=1+\left\lfloor\frac{N-L}{D}\right\rfloor.
$$

For $N=4383$, $M=90$. The final segment starts at index 4272 and ends on
16 December 2017. The final 15 days cannot form another complete window
and are discarded from spectral estimation, although they participate in
the earlier full-record preprocessing and EOF fit.

At each latitude and longitude, a least-squares straight line is removed
from each segment. The residual is multiplied by the symmetric Hann window
implemented by `np.hanning(L)`:

$$
w_n=\frac12\left[1-\cos\left(\frac{2\pi n}{L-1}\right)\right],
\qquad n=0,\ldots,L-1.
$$

The mean-square window energy is
$U=L^{-1}\sum_n w_n^2=0.37109375$. Cross- and auto-spectra are divided by
$U$, corresponding to a factor of approximately 2.69474. The same factor
cancels in the final coherence ratio.

### Space–time transform and propagation convention

Let $x^{(w)}_{s,n,j,\ell}$ denote a detrended, tapered segment at latitude
index $j$ and longitude index $\ell$. The transform is

$$
X_{s,j}(f_m,k)=\frac{1}{LN_\lambda}
\sum_{n=0}^{L-1}\sum_{\ell=0}^{N_\lambda-1}
x^{(w)}_{s,n,j,\ell}
\exp\left(+\frac{2\pi i mn}{L}\right)
\exp\left(-\frac{2\pi i k\ell}{N_\lambda}\right),
$$

with an identical definition for $Y$. Here $f_m=m/(L\Delta t)$ in signed
FFT order, $\Delta t=1$ day, and $k$ is integer zonal wavenumber. The code
uses `ifft` in time and `fft` in longitude, dividing by $N_\lambda$;
`ifft` already supplies the factor $1/L$.

For positive frequency, positive $k$ represents eastward propagation and
negative $k$ westward propagation under this convention. The frequency
spacing is $1/96\approx0.0104167$ cycles day⁻¹. This is the bin spacing;
the taper also broadens spectral features. No additional frequency or
wavenumber smoothing, zero padding, or background-spectrum division is
performed.

### Cross-spectra, auto-spectra, and averaging

At each $(f,k)$, define the latitude-mean segment spectra as

$$
C_s=\frac{1}{UN_\phi}\sum_j X_{s,j}Y_{s,j}^{*},\qquad
P_{x,s}=\frac{1}{UN_\phi}\sum_j |X_{s,j}|^2,\qquad
P_{y,s}=\frac{1}{UN_\phi}\sum_j |Y_{s,j}|^2.
$$

The asterisk denotes complex conjugation. Latitudes receive equal weight;
there is no cosine-latitude weighting. The fields themselves are not
latitude-averaged before transformation. The segment spectra are then
averaged:

$$
\overline C=\frac1M\sum_s C_s,\qquad
\overline P_x=\frac1M\sum_s P_{x,s},\qquad
\overline P_y=\frac1M\sum_s P_{y,s}.
$$

The estimated magnitude-squared coherence (MSC) is

$$
\boxed{\widehat\gamma^2_{xy}(f,k)=
\frac{|\overline C(f,k)|^2}
{\overline P_x(f,k)\,\overline P_y(f,k)}}.
$$

This is a Welch-style averaged spectral estimate extended to
frequency–zonal-wavenumber space and latitude averaging. The normalization
matches the standard MSC definition.
([SciPy coherence documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.coherence.html))

Complex cross-spectra must be averaged before taking their magnitude.
Phase differences across latitudes and segments can therefore cancel.
Averaging individual coherence values or individual cross-spectral
magnitudes would estimate a different quantity. With only one Fourier
pair and no averaging, $|XY^*|^2/(|X|^2|Y|^2)=1$ whenever both powers
are nonzero; averaging is essential to the estimator.

MSC is dimensionless and lies between zero and one. It measures the
consistency of the linear spectral relationship within the averaging
ensemble. It does not give the sign of the relationship, phase lead/lag,
or a causal direction. Because the numerator includes both
$\Re(\overline C)^2$ and $\Im(\overline C)^2$, high MSC can occur for
in-phase, antiphase, or quadrature relationships. Constant nonzero
rescaling of either field does not change MSC. The implementation leaves
bins with nonpositive power products as NaN and clips numerical results
to $[0,1]$.

## 4. Random-pair statistical assessment

### Null construction

The operational null asks whether the observed spectral association is
larger than associations generated by independently drawing source and
reference segments from their respective empirical segment collections.
This is a random-pair surrogate test implemented using resampling with
replacement. It is not a paired bootstrap confidence interval and is not
an exact permutation test.

For replicate $b=1,\ldots,B$, with $B=1000$, draw two independent vectors:

$$
I_{b,r}\overset{\mathrm{iid}}\sim\mathrm{Uniform}\{0,\ldots,M-1\},
\qquad
J_{b,r}\overset{\mathrm{iid}}\sim\mathrm{Uniform}\{0,\ldots,M-1\},
\qquad r=1,\ldots,M.
$$

The code draws the reference vector first, then the source vector. Each
vector contains $M=90$ indices and permits repetitions and omissions.
The selected source and reference segments are paired by their position
$r$ in these vectors. Each draw selects an entire 96-day latitude–longitude
array, preserving its internal temporal and spatial structure. The same
selected indices apply to all spectral bins within a replicate.

For each replicate, compute

$$
\overline C^{(b)}=
\frac{1}{MUN_\phi}\sum_{r,j}
X_{I_{b,r},j}Y_{J_{b,r},j}^{*},
$$

$$
\overline P_x^{(b)}=\frac{1}{MUN_\phi}\sum_{r,j}|X_{I_{b,r},j}|^2,
\qquad
\overline P_y^{(b)}=\frac{1}{MUN_\phi}\sum_{r,j}|Y_{J_{b,r},j}|^2,
$$

and $\widehat\gamma^{2(b)}=|\overline C^{(b)}|^2/
(\overline P_x^{(b)}\overline P_y^{(b)})$.
Both powers are recomputed from the selected segments; the observed
denominator is not reused. The selected arrays have already been
detrended and tapered, so those operations are not repeated.

Independent draws usually break the original calendar alignment but do
not prohibit matching indices or adjacent, overlapping segments.
For example, the probability of a matched pair at a draw is $1/M$, giving
one matched pair on average among $M$ draws. The null therefore does not
enforce complete temporal separation.

All replicates are retained in
`boot_coherence_squared[reference][source]`, with shape
$(1000,96,360)$. New draws are made for every variable pair. The generator
continues through the pair loop, so changing the pair order or rerunning
only part of the loop changes the sampled indices.

### Bin-specific thresholds and decision rule

For each source/reference pair and each $(f,k)$, the threshold is

$$
T_{0.99}(f,k)=Q_{0.99}\left(
\widehat\gamma^{2(1)}(f,k),\ldots,
\widehat\gamma^{2(B)}(f,k)\right).
$$

The code uses `np.percentile(null_samples, 99.0, axis=0)` with the default
linear interpolation. For 1,000 sorted null values, the zero-based
quantile index is $(1000-1)\times0.99=989.01$, so the threshold interpolates
between the 990th and 991st order statistics.
([NumPy percentile documentation](https://numpy.org/doc/stable/reference/generated/numpy.percentile.html))

A bin is marked only if

$$
\widehat\gamma^2_{\mathrm{obs}}(f,k)>T_{0.99}(f,k).
$$

Equality is not marked. This is an upper-tail test: unusually large MSC
is the alternative of interest. There is a different threshold for each
bin and each variable pair, rather than a universal cutoff such as 0.5.
The threshold is a percentile of the surrogate null distribution, not a
99% confidence interval for the observed coherence.

### Diagnostic p-values and undefined bins

The notebook also stores

$$
p_{\mathrm{MC}}(f,k)=
\frac{1+\sum_{b=1}^{B}\mathbf1\left[
\widehat\gamma^{2(b)}(f,k)\ge
\widehat\gamma^2_{\mathrm{obs}}(f,k)-\epsilon(f,k)\right]}{B+1},
$$

where $\epsilon=100\,\epsilon_{\mathrm{float64}}
|\widehat\gamma^2_{\mathrm{obs}}|$ includes numerical ties in the upper
tail. The minimum reported value is $1/1001\approx0.000999$.
The plus-one convention avoids zero Monte Carlo p-values and is also
used in randomized permutation testing; it does not make this
dependent-segment surrogate scheme an exact test.
([SciPy permutation-test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html))

The plotted dots use the strict percentile comparison, not
`p_value < 0.01`. Finite-sample interpolation and ties can make those
decisions differ. A valid bin requires a finite observation and all
$B$ finite null values. Otherwise its p-value and threshold are NaN and
its significance mask is false. `n_valid` records the finite replicate
count. The code does not silently reduce $B$ using `nanpercentile`.

### Meaning and limitations of the statistical result

The appropriate description is **approximate pointwise significance
against a random-pair segment null**, at a nominal upper-tail level of
1%. The following implications follow from the implemented sampling scheme:

* **Overlap dependence is not reproduced.** Adjacent observed segments
  share 48 days, and additional dependence may persist beyond the overlap.
  Independent segment draws do not reproduce the ordered dependence
  structure. The actual false-positive rate has not been calibrated;
  90 segments, or 90 times 10 latitudes, cannot simply be treated as an
  effective independent sample count.
* **The segment distribution is pooled across the record.** The test
  assumes the pooled segment collection is an appropriate null reference.
  Seasonal or interannual changes in variance or spectral structure may
  affect that assumption even after seasonal-mean removal.
* **There is no correction across bins or variable pairs.** The dots do
  not establish 1% significance for the entire map, a wave domain, or all
  nine comparisons together. Neighboring spectral bins are also dependent.
* **Tail estimates have Monte Carlo uncertainty.** With 1,000 draws,
  only about ten values populate the upper 1% tail. Increasing the draw
  count and comparing independent seeds can assess threshold stability,
  but cannot repair a misspecified null.
* **The test does not compare variable pairs directly.** A larger observed
  QR–W1 coherence than QR–W2 coherence, or significance for one but not the
  other, is not itself a test of their difference.

A stronger inferential analysis would need a separately specified and
validated null that reproduces relevant temporal dependence, along with
a declared testing family or domain statistic. These are possible future
changes, not procedures performed by the current notebook. No analytical
degrees-of-freedom, beta-distribution, or F-test threshold is used here.

## 5. Wave-domain summaries and plotting

The shared helper defines the following domains, with inclusive bounds:

| Domain | Zonal wavenumber | Frequency (cycles day⁻¹) | Additional restriction |
| --- | --- | --- | --- |
| Kelvin | 3 to 8 | 1/20 to 0.3 | Between Kelvin curves for equivalent depths 8 and 90 m |
| Equatorial Rossby | −8 to −2 | 0.02 to 0.1 | Between low-frequency $n=1$ Rossby curves for 8 and 90 m |
| MJO | 1 to 3 | 1/90 to 1/30 | Rectangular domain; no dispersion clipping |

The implemented curves use $c=\sqrt{gh}$, $a=6.371\times10^6$ m,
$g=9.81$ m s⁻², $\Omega=7.292\times10^{-5}$ s⁻¹,
$\beta=2\Omega/a$, and physical zonal wavenumber $\kappa=k/a$:

$$
f_K(k,h)=\frac{86400}{2\pi}c\kappa,\qquad
f_R(k,h)=\frac{86400}{2\pi}
\frac{-\beta\kappa}{\kappa^2+3\beta/c}.
$$

The Rossby expression is the low-frequency $n=1$ approximation with no
background zonal flow. For domain $\mathcal D$, the reported value is

$$
\overline{\gamma^2}_{\mathcal D}
=\frac{1}{N_{\mathcal D}}\sum_{(f,k)\in\mathcal D,
\ \mathrm{finite}}\widehat\gamma^2(f,k).
$$

This is an unweighted average of finite MSC bins, including bins that do
not pass the significance criterion. It is not coherence computed from
spectra pooled across the domain. Saved output reports 96 Kelvin, 33
Rossby, and 6 MJO bins for each pair. The small MJO count follows from
the 96-day frequency grid: only $2/96$ and $3/96$ cycles day⁻¹ fall in its
frequency interval, paired with wavenumbers 1, 2, and 3.

No domain-mean significance test or confidence interval is calculated.
To test that statistic, each null replicate would also need to be
averaged over the same fixed domain and compared with the observed
domain mean; averaging pointwise thresholds would not provide that test.

For plotting, `fftshift` is applied consistently to coordinates, observed
MSC, thresholds, and masks. Only $f>0$ is shown. With 96 temporal samples,
there are 47 positive bins; the Nyquist bin is stored at −0.5 cycles day⁻¹
and excluded by this selection. The observed plots show $-15\le k\le15$
and use color levels from 0.2 to 0.7; the estimator itself remains bounded
by zero and one. Black dots mark the pointwise criterion, and dashed
outlines show the averaging domains.

The threshold-map cell independently recomputes thresholds from stored
null samples using the current percentile. Its configured color maximum
of 0.03 affects display only. PNG and PDF maps for all nine pairs are
saved in `Figure/New_analysis_Figure/Spectral_test_criteria/`. Null arrays,
p-values, and masks are retained in memory by the notebook; it does not
explicitly export these numerical arrays.

## 6. Relationship to the other variable analyses

The shared spectral estimator is also used by the following notebooks,
but their preprocessing and statistical coverage differ:

| Notebook | Variable pairs | Meridional treatment | Random-pair test present? |
| --- | --- | --- | --- |
| `coherence_square_stat.ipynb` | QR/LW/SW × CWV/W1/W2 | Full fields | Yes, nine pairs |
| `Qr_Prec.ipynb` | QR/LW/SW × IMERG precipitation | Symmetric | No |
| `Qr_CWV.ipynb` | QR/LW/SW × CWV | Symmetric | No |
| `Qr_omega.ipynb` | QR/LW/SW × W1/W2 | Symmetric | No |
| `Prec_CWV.ipynb` | IMERG precipitation × CWV | Symmetric | No |

For the symmetric analyses,
$x_S(t,\phi,\lambda)=\tfrac12[x(t,\phi,\lambda)+x(t,-\phi,\lambda)]$
is applied before segmenting. Precipitation is loaded from
`IMERG/DATA/IMERG_1deg.nc`; `Prec_CWV.ipynb` additionally interpolates
precipitation linearly onto the CWV latitude centers and explicitly
validates the requested daily calendar and global longitude spacing.

`Qr_omega.ipynb` removes the time–longitude mean at each pressure and
latitude before PCA, then uses negative PCs. The statistics notebook
instead passes the loaded W field directly to PCA, which centers each
pressure level across all samples, and uses the sign rule in Section 2.
The different centering can alter the EOF basis; it is not merely a sign
change. These workflows should not be described as identical analyses.

The standalone `coherence_square_stat.py` also differs from the active
notebook: it supports precipitation and meridional-component selection,
uses its own omega preprocessing, and explicitly applies no significance
test. Its filename does not establish that the notebook's test was used.
The existing random-pair thresholds therefore should not be attached to
the symmetric or precipitation results without recalculating matching
observations and null samples.

## 7. Manuscript-ready description for the statistics notebook

Magnitude-squared coherence was used to quantify the frequency–zonal-
wavenumber relationship between column radiative heating and column water
vapor or the first two vertical EOF scores of pressure vertical velocity.
The analysis considered total, longwave, and shortwave radiative heating
separately, yielding nine source–reference combinations. Daily fields were
aligned on a common 1° longitude–latitude grid within 5°S–5°N for
2006–2017. A time–longitude mean was removed separately at each latitude
from the heating and water-vapor fields. The vertical-velocity references
were obtained by applying a two-component, unweighted PCA to pressure
profiles, treating time–latitude–longitude locations as samples and
centering each pressure level across those samples. The full meridional
fields were used in this statistical analysis.

The 4,383-day record was divided into 90 complete 96-day segments with
48-day overlap. Each segment was linearly detrended at every grid point
and multiplied by a Hann window. An inverse temporal Fourier transform
and forward zonal Fourier transform were applied, such that positive
zonal wavenumbers at positive frequencies represent eastward propagation.
Complex cross-spectra and auto-spectra were corrected for the mean-square
window energy and averaged equally over latitudes and segments.
Magnitude-squared coherence was calculated as
$\widehat\gamma^2_{xy}=|\overline{XY^*}|^2/
(\overline{|X|^2}\,\overline{|Y|^2})$,
where the overbars denote the spectral averaging. The frequency-bin
spacing was 1/96 cycles day⁻¹. Kelvin, equatorial Rossby, and MJO summaries
were calculated as unweighted means of coherence over the domains
specified in Section 5.

An approximate pointwise significance assessment was performed using
1,000 random-pair resamples for each source–reference combination. In
each resample, 90 source segments and 90 reference segments were drawn
independently with replacement from their respective segment collections
and paired in draw order. The full coherence estimator, including both
auto-spectra, was recomputed for each resample. An observed spectral bin
was marked when its coherence strictly exceeded the 99th percentile of
its random-pair distribution. This procedure preserves the internal
structure of individual segments but does not reproduce dependence
between overlapping segments. The resulting thresholds are therefore
approximate pointwise criteria at a nominal 1% upper-tail level; no
multiple-testing correction or domain-mean significance test was applied.

Suggested figure-caption text:

> Shading shows magnitude-squared coherence. Black dots identify spectral
> bins whose observed coherence exceeds the corresponding 99th percentile
> of 1,000 independently random-paired segment resamples. These indicate
> approximate pointwise significance; overlapping-segment dependence and
> multiple testing are not accounted for. Dashed curves outline the wave
> domains, and annotations give the mean observed coherence over all finite
> bins in each domain.

## 8. Code traceability and reproducibility notes

Cell numbers below are zero-based positions in the saved notebook JSON,
not execution counters:

| Operation | Implementation |
| --- | --- |
| Input loading, date/grid alignment | Statistics notebook cells 5 and 7 |
| Settings and heating anomalies | Cell 12 |
| CWV and EOF reference construction | Cell 14; `preprocessing.calculate_eof()` |
| Observed MSC | Cell 17 |
| Random-pair sampling | Cell 19 |
| Thresholds, p-values, masks | Cell 21 |
| Wave-domain means | Cell 26 |
| Observed plots and significance dots | Cells 28 and 30 |
| Threshold maps | Cell 32 |
| Detrending, Hann taper, transform, MSC ratio | [`src/spectrum.py`](../src/spectrum.py) |
| Spectral averaging, coordinates, wave domains | [`src/cross_spectrum_ana.py`](../src/cross_spectrum_ana.py) |
| PCA and input conversion | [`src/preprocessing.py`](../src/preprocessing.py) |

Some notebook prose is stale: cell 18 mentions 100 trial replicates;
cells 20 and 22 mention the 95th percentile. Executable cells set
`bootstrap_size = 1000` and `BOOTSTRAP_PERCENTILE = 99.0`, consistent with
saved progress output and `q99_B1000` figure filenames. This document
uses those executable settings. The plotting comment about a 0–1 color
scale also differs from the actual 0.2–0.7 contour levels.

For an archived reproduction, retain the notebook and shared-module
versions, input provenance, package versions, fitted EOFs, configuration,
and exported null/observed arrays. Seed 42 controls the resampling draws;
the PCA call does not specify a solver or random state, so the resampling
seed alone is not a complete guarantee of identical EOF fitting across
software environments. Changing observations or preprocessing requires
regenerating their null samples; changing the percentile requires
refreshing the significance mask as well as the threshold maps.

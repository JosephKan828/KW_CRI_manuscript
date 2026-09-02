# KW_CRI_manuscript

## Cross-spectrum calculation: original and revised implementations

This section documents the mathematics used in `Code/Qr_Prec.ipynb` and the
differences between the original and revised implementations. Let
$x(t,\lambda)$ denote radiative heating and $y(t,\lambda)$ denote
precipitation, where $t$ is time and $\lambda$ is longitude.

### Space-time Fourier transform

Both implementations use an inverse FFT in time and a forward FFT in
longitude:

$$
X(f,k)=\frac{1}{N_tN_\lambda}
\sum_{t=0}^{N_t-1}\sum_{\lambda=0}^{N_\lambda-1}
x(t,\lambda)
\exp\left(+i\frac{2\pi ft}{N_t}\right)
\exp\left(-i\frac{2\pi k\lambda}{N_\lambda}\right),
$$

with an equivalent definition for $Y(f,k)$. Here, $f$ is temporal
frequency and $k$ is zonal wavenumber. NumPy's `ifft` supplies the
$1/N_t$ factor, while division by the number of longitudes supplies the
$1/N_\lambda$ factor.

The original expression

```python
data_fft = np.fft.fft(np.fft.ifft(data, axis=0), axis=1) / data.shape[1]
```

and the revised `space_time_transform()` function are mathematically
equivalent. The revised function exposes the time and longitude axes as
arguments, allowing the same calculation to operate on two-, three-, or
four-dimensional arrays.

### Complex cross-spectrum

Both implementations define the heating-precipitation cross-spectrum as

$$
C_{xy}(f,k)=X(f,k)Y^*(f,k),
$$

where the asterisk denotes complex conjugation. If

$$
X=a+ib, \qquad Y=c+id,
$$

then

$$
C_{xy}=(a+ib)(c-id)=(ac+bd)+i(bc-ad).
$$

The real and imaginary components are therefore

$$
\operatorname{Co}_{xy}=\Re(C_{xy})=ac+bd
$$

and

$$
\operatorname{Quad}_{xy}=\Im(C_{xy})=bc-ad.
$$

Writing the Fourier coefficients in amplitude-phase form,

$$
X=A_xe^{i\phi_x}, \qquad Y=A_ye^{i\phi_y},
$$

gives

$$
C_{xy}=A_xA_y e^{i(\phi_x-\phi_y)}.
$$

For $\Delta\phi=\phi_x-\phi_y$, the components become

$$
\operatorname{Co}_{xy}=A_xA_y\cos(\Delta\phi)
$$

and

$$
\operatorname{Quad}_{xy}=A_xA_y\sin(\Delta\phi).
$$

The cross-spectral magnitude and phase are consequently

$$
|C_{xy}|=\sqrt{\operatorname{Co}_{xy}^2+
\operatorname{Quad}_{xy}^2}
$$

and

$$
\Delta\phi=\operatorname{atan2}
\left(\operatorname{Quad}_{xy},\operatorname{Co}_{xy}\right).
$$

Because the notebook uses an inverse transform in time, this Fourier sign
convention must be retained when interpreting whether a positive phase means
that heating leads or lags precipitation.

### Averaging over latitude and time segments

For segment $s$ and latitude $j$, let the cross-periodogram be

$$
C_{s,j}(f,k)=X_{s,j}(f,k)Y^*_{s,j}(f,k).
$$

The latitude-averaged spectrum for each segment is

$$
C_s(f,k)=\frac{1}{N_y}\sum_{j=1}^{N_y}C_{s,j}(f,k),
$$

and the final spectrum is obtained by averaging the complex values across
the overlapping segments:

$$
\overline{C}_{xy}(f,k)=
\frac{1}{N_s}\sum_{s=1}^{N_s}C_s(f,k).
$$

The original corrected latitude loop and the revised vectorized calculation
evaluate the same sums. The computational difference is that the revised
version transforms every latitude in a segment simultaneously and reuses the
precipitation transform for the QR, longwave, and shortwave calculations. It
processes one segment at a time to avoid multi-gigabyte temporary arrays.

The complex spectra are averaged before extracting phase. Phase angles should
not be averaged directly because phase is circular; for example, $179^\circ$
and $-179^\circ$ are both close to $180^\circ$, although their ordinary
arithmetic mean is $0^\circ$.

### Hann-window energy correction

Each 96-day segment is multiplied by a Hann window $w(t)$:

$$
x_w(t)=w(t)x(t), \qquad y_w(t)=w(t)y(t).
$$

The original spectrum retained the reduction in spectral magnitude caused by
the window. The revised calculation uses the mean-square window energy

$$
U=\frac{1}{N_t}\sum_{t=0}^{N_t-1}w^2(t)
$$

and applies

$$
C_{xy,\mathrm{corrected}}=
\frac{C_{xy,\mathrm{windowed}}}{U}.
$$

For the 96-point Hann window used by the notebook,
$U\approx0.3711$, so $1/U\approx2.695$. This correction changes spectral
magnitudes but does not change phase, signs, spectral peak locations, or the
relative space-time structure.

Thus, before plotting, the two estimators can be summarized as

$$
C_{\mathrm{original}}=
\frac{1}{N_sN_y}\sum_s\sum_j
X^{(w)}_{s,j}Y^{(w)*}_{s,j}
$$

and

$$
C_{\mathrm{revised}}=
\frac{1}{UN_sN_y}\sum_s\sum_j
X^{(w)}_{s,j}Y^{(w)*}_{s,j}.
$$

Apart from the factor $1/U$, the underlying estimators are equivalent.

### Why the flipped-spectrum addition was removed

The original plotting code included

```python
cross_plot = cross_plot + np.flip(cross_plot, axis=(0, 1))
```

For real-valued input fields, the cross-spectrum has conjugate symmetry:

$$
C_{xy}(-f,-k)=C_{xy}^*(f,k).
$$

If $C_{xy}(f,k)=P+iQ$, its conjugate counterpart is $P-iQ$. Adding the
two produces

$$
(P+iQ)+(P-iQ)=2P.
$$

This doubles the co-spectrum but cancels the quadrature spectrum:

$$
\Re(C+C^_)=2P, \qquad \Im(C+C^_)=0.
$$

For example, if the phase difference is $30^\circ$, then

$$
C=A_xA_y(0.866+0.5i).
$$

Adding the conjugate gives

$$
C+C^*=1.732A_xA_y+0i,
$$

so the quadrature information is lost. In addition, for even-length FFT
dimensions, `np.flip()` after `fftshift()` is not an exact discrete mapping
between every $(f,k)$ bin and $(-f,-k)$, particularly at zero and Nyquist
bins. It can therefore distort the spectrum rather than produce exact
symmetrization.

The revised plotting code preserves the complex spectrum and separates it
directly:

```python
cospectrum = mean_cross.real
quadrature = mean_cross.imag
phase_degrees = np.rad2deg(np.angle(mean_cross))
```

This retains both the in-phase covariance and the phase-lead/lag information.

### Plot coordinates and signed color scales

The spectrum and its coordinate arrays must be shifted together:

```python
freq_plot = np.fft.fftshift(freq)
wnum_plot = np.fft.fftshift(wnum)
cospectrum_plot = np.fft.fftshift(cospectrum, axes=(0, 1))
quadrature_plot = np.fft.fftshift(quadrature, axes=(0, 1))
```

The revised visualization selects positive temporal frequencies explicitly
with `freq_plot > 0`. It also uses color levels symmetric around zero because
both components are signed. A positive co-spectrum indicates an in-phase
contribution, while a negative co-spectrum indicates an out-of-phase
contribution. Positive and negative quadrature values represent opposite phase
directions under the adopted Fourier convention.

In summary, the revised code does not change the fundamental definition
$C_{xy}=XY^*$. Its substantive mathematical changes are the Hann-window
energy correction and preservation of the imaginary component by removing the
flipped-spectrum addition. The remaining changes improve axis handling,
computational efficiency, memory usage, and visualization of signed values.

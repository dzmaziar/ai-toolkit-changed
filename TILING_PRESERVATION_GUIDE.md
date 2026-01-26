# Guide: Preserving Tiling in Flux Kontext LoRA Training with Custom Local Losses

## Quick Start

If you're using `texture_loss: "custom"` with local losses, here's the essential configuration:

```yaml
train:
  texture_loss: "custom"
  loss_local_coef: 1.0      # PRIMARY - patch-based local analysis
  loss_Spect_coef: 0.5      # Secondary - global spectral
  loss_Log_coef: 0.1         # Tertiary
  loss_AFC_coef: 0.2         # Tertiary
  
  # Local loss parameters (most important)
  local_patch_size: 192      # Match to your tile size
  local_stride: 96           # Half of patch_size
  local_w_afc: 1.5           # Period detection (increase to 2.0-2.5 if needed)
  local_w_phase: 1.2         # Alignment (increase to 1.5-2.0 if needed)
```

**Key parameters to adjust:**
- `loss_local_coef`: Primary weight (1.0-2.0 for strong tiling preservation)
- `local_patch_size`: Should match your tile size (128-256)
- `local_w_afc`: Period detection in patches (1.5-2.5)
- `local_w_phase`: Patch alignment (1.2-2.0)

## Overview

When training LoRA for Flux Kontext on tiling materials (like bricks, tiles, patterns), the model can lose the tiling property during training. This guide focuses on using the **`custom` texture loss mode with LocalTextureRegularityLoss** to preserve tiling while maintaining similarity to training data.

**This guide is specifically for users who use `texture_loss: "custom"` with `loss_local_coef > 0`** - the patch-based local texture analysis approach.

## Key Findings

When using `texture_loss: "custom"`, the system combines multiple texture losses:

1. **LocalTextureRegularityLoss** (`loss_local_coef`) - **PRIMARY for patch-based analysis** - breaks image into patches and analyzes tiling locally
2. **SpectralPeriodLoss** (`loss_Spect_coef`) - Global frequency spectrum analysis
3. **LogPolarAlignLoss** (`loss_Log_coef`) - Scale/rotation invariant analysis
4. **ACFPeriodLoss** (`loss_AFC_coef`) - Autocorrelation-based period detection
5. **PhaseCorrelationLoss** (`loss_Phase_coef`) - Phase alignment

The formula used in custom mode:
```
additional_loss = loss_local_coef * tex_local + 
                  loss_Spect_coef * tex_sp + 
                  loss_Log_coef * tex_lp + 
                  loss_AFC_coef * tex_acf + 
                  loss_Phase_coef * tex_ph
```

According to the code comments in `toolkit/util/texture_losses.py`:
> "В качестве основы используем SpectralPeriodLoss. Именно данный тип лоссов наиболее сильно влияет на обучение. Помимо этого с малыми весами также используем LogPolarAlignLoss и ACFPeriodLoss. Если их дополнительно добавляем к SpectralPeriodLoss, помогает модели при обучении сохранять тайтлинг."

**For tiling materials, LocalTextureRegularityLoss is particularly effective** because it:
- Analyzes patches locally (catches local tiling patterns)
- Uses hard-mining to focus on problematic patches
- Combines multiple analysis methods (spectral, ACF, phase, log-polar) at patch level
- Handles irregular tiling better than global methods

## Configuration Changes

### 1. Enable Custom Texture Loss with Local Losses

Add the following parameters to your `train_lora_flux_kontext_24gb.yaml` config file under the `train:` section:

```yaml
train:
  # ... existing parameters ...
  
  # Texture loss configuration for tiling preservation
  texture_loss: "custom"  # Uses combination of all losses
  
  # Loss coefficients - these control how much each texture loss contributes
  # For tiling materials, focus on local loss + spectral loss
  loss_local_coef: 1.0    # LocalTextureRegularityLoss weight - PRIMARY for patch-based tiling
  loss_Spect_coef: 0.5    # SpectralPeriodLoss weight - global frequency analysis
  loss_Log_coef: 0.1      # LogPolarAlignLoss weight - secondary
  loss_AFC_coef: 0.2      # ACFPeriodLoss weight - secondary
  loss_Phase_coef: 0.0    # PhaseCorrelationLoss weight - optional
  loss_coef: 0.0          # Not used in custom mode
```

### 3. SpectralPeriodLoss Parameters (Secondary in Custom Mode)

These are the most important parameters for tiling preservation:

```yaml
train:
  # SpectralPeriodLoss parameters
  r_bins: 256          # Number of radial bins for frequency analysis (128-512, default: 256)
  tau: 0.06            # Soft peak detection parameter (0.04-0.10, lower = sharper peaks)
  band_sigma: 0.15     # Bandwidth for frequency matching (0.10-0.25, lower = tighter matching)
  w_peak: 1.0          # Weight for peak frequency matching
  w_band: 1.0          # Weight for band matching
  min_bin: 2           # Minimum frequency bin to consider (2-5, default: 2)
  max_bin: null        # Maximum frequency bin (null = auto, or set to ~128 for tighter control)
```

**Recommended starting values for bricks/tiles:**
- `r_bins: 256` (good balance)
- `tau: 0.06` (default, works well)
- `band_sigma: 0.12-0.18` (tighter for regular patterns)
- `w_peak: 1.0`, `w_band: 1.0` (balanced)

### 4. LogPolarAlignLoss Parameters (Tertiary)

Helps with scale and rotation invariance:

```yaml
train:
  # LogPolarAlignLoss parameters
  log_out_r: 128       # Radial resolution (64-256, default: 128)
  log_out_t: 180       # Angular resolution (90-360, default: 180)
  log_r_min: 0.01      # Minimum radius (0.005-0.02, default: 0.01)
  log_w: 0.15          # Weight (0.1-0.3 recommended for tiling)
```

### 5. ACFPeriodLoss Parameters (Tertiary)

Autocorrelation-based period detection:

```yaml
train:
  # ACFPeriodLoss parameters
  afc_r_bins: 256      # Radial bins (128-512, default: 256)
  afc_tau: 0.08        # Soft peak detection (0.06-0.12, default: 0.08)
  afc_min_rel: 0.03    # Minimum relative period (0.02-0.05)
  afc_max_rel: 0.6     # Maximum relative period (0.4-0.7, default: 0.6)
  afc_w: 0.4           # Weight (0.2-0.6 recommended)
```

### 2. LocalTextureRegularityLoss Parameters (PRIMARY for Custom Mode)

**This is the most important section when using `texture_loss: "custom"` with local losses.**

LocalTextureRegularityLoss works by:
1. Extracting patches from the image (with random offset to avoid grid artifacts)
2. Applying hard-mining to focus on problematic patches (where tiling breaks)
3. Analyzing each patch with multiple methods (spectral, ACF, phase, log-polar)
4. Using global peak center to prevent local scale drift

```yaml
train:
  # Local texture loss parameters - PRIMARY for tiling preservation
  local_patch_size: 192    # Patch size in pixels (128-256, larger = more global)
                           # For bricks: 192-256 works well
                           # For smaller tiles: 128-192
                           # DEFAULT WAS 64 - TOO SMALL! Use 192+ for tiling
  local_stride: 96         # Stride between patches (half of patch_size recommended)
                           # Smaller stride = more patches = better coverage but slower
  local_min_coverage: 0.45  # Minimum mask coverage per patch (0.3-0.6)
                            # Higher = only analyze patches with more visible tiling
  
  # Hard-mining parameters - CRITICAL for focusing on problematic patches
  local_hard_k_frac: 0.30  # Fraction of worst patches to analyze (0.2-0.5)
                           # Higher = more patches analyzed, slower but more thorough
                           # Lower = focus only on worst patches, faster
  local_hard_k_min: 4      # Minimum number of patches to analyze (4-16)
                           # Ensures at least this many patches even if fraction is small
  
  # Weights for components within LocalTextureRegularityLoss
  # These control the internal balance of the local loss
  local_w_spectral: 0.5     # Weight for spectral component within patches
  local_w_afc: 1.5          # Weight for ACF component (higher = more period detection)
  local_w_phase: 1.2        # Weight for phase component (alignment)
  local_w_log: 0.0          # Weight for log-polar component (0.0 = disabled)
```

**Recommended values for different tile sizes:**
- **Large tiles (bricks, large patterns)**: `local_patch_size: 256`, `local_stride: 128`, `local_hard_k_frac: 0.3`
- **Medium tiles**: `local_patch_size: 192`, `local_stride: 96`, `local_hard_k_frac: 0.3` (default)
- **Small tiles**: `local_patch_size: 128`, `local_stride: 64`, `local_hard_k_frac: 0.4`
- **Irregular tiling**: Increase `local_w_afc` to 2.0-2.5, increase `local_w_phase` to 1.5, increase `local_hard_k_frac` to 0.4-0.5

### 6. PhaseCorrelationLoss Parameters (Optional - Used by Local Loss)

For phase alignment:

```yaml
train:
  # Phase correlation parameters
  center_weight: 1.0        # Weight for center correlation
  pce_weight: 0.05          # Peak correlation energy weight (0.01-0.1)
  temperature: 0.02         # Temperature for softmax (0.01-0.05)
  use_hann: true            # Use Hann window (recommended: true)
  demean: true              # Demean before correlation (recommended: true)
  exclude_radius: 7          # Exclude center radius (5-10)
```

### 7. Critical Parameters for Flux (Rectified Flow)

These parameters are **critical** for Flux models and affect how texture losses are calculated:

```yaml
train:
  # v_scale - CRITICAL for rectified flow (Flux uses rectified flow)
  # Controls how x0 is predicted from latents for texture loss calculation
  v_scale: 0.2  # Range: 0.15-0.35, default: 0.2
                # Lower = less noise in prediction, but may lose detail
                # Higher = more detail, but may introduce artifacts
                # For tiling: 0.2-0.25 usually works well
  
  # Beta parameters - control warmup of texture losses
  min_beta: 0.0   # Starting weight (0.0 = no texture loss at start)
  max_beta: 0.5  # Maximum weight (0.5 = 50% of texture loss at full strength)
                 # Higher max_beta = stronger texture loss influence
  
  # Gate - controls which timesteps get texture losses
  gate: 0.65  # Only apply texture losses when t < 0.35 (last 35% of denoising)
              # Lower = apply earlier (e.g., 0.5 = last 50% of timesteps)
              # For tiling: 0.6-0.7 usually works well
              # Too early (low gate) = texture loss on noisy images (bad)
              # Too late (high gate) = texture loss only at end (may be too late)
```

**Important notes:**
- `v_scale` is used in the formula: `pred_x0 = x_t - t * (v_scale * v)` for rectified flow
- If texture losses seem weak, try increasing `max_beta` to 0.7-1.0
- If training is unstable, try decreasing `max_beta` to 0.3-0.4
- `gate` prevents texture losses from being applied to very noisy images (where FFT doesn't work well)

## Complete Example Configuration for Custom Mode with Local Losses

Here's a complete example configuration section for training tiling materials with custom local losses:

```yaml
train:
  batch_size: 1
  steps: 3000
  gradient_accumulation_steps: 1
  train_unet: true
  train_text_encoder: false
  gradient_checkpointing: true
  noise_scheduler: "flowmatch"
  optimizer: "adamw8bit"
  lr: 1e-4
  timestep_type: "weighted"
  dtype: bf16
  
  # Texture loss for tiling preservation - CUSTOM MODE
  texture_loss: "custom"
  
  # Loss weights - PRIMARY: local loss for patch-based analysis
  loss_local_coef: 1.0      # PRIMARY - LocalTextureRegularityLoss (patch-based)
  loss_Spect_coef: 0.5      # Secondary - global spectral analysis
  loss_Log_coef: 0.1        # Tertiary - log-polar alignment
  loss_AFC_coef: 0.2         # Tertiary - autocorrelation
  loss_Phase_coef: 0.0       # Optional - phase correlation (already in local loss)
  
  # LocalTextureRegularityLoss parameters (PRIMARY)
  local_patch_size: 192      # Patch size (128-256, adjust for tile size)
  local_stride: 96           # Stride (half of patch_size)
  local_min_coverage: 0.45   # Minimum mask coverage
  local_hard_k_frac: 0.30    # Fraction of worst patches for hard-mining
  local_hard_k_min: 4        # Minimum patches for hard-mining
  local_w_spectral: 0.5      # Weight for spectral in patches
  local_w_afc: 1.5           # Weight for ACF in patches (important for period)
  local_w_phase: 1.2         # Weight for phase in patches
  local_w_log: 0.0           # Weight for log-polar in patches
  
  # Critical Flux parameters
  v_scale: 0.2               # Rectified flow parameter (0.15-0.35)
  min_beta: 0.0              # Texture loss warmup start
  max_beta: 0.5              # Texture loss max weight
  gate: 0.65                 # Timestep gate (only last 35% of timesteps)
  
  # SpectralPeriodLoss parameters (used by local loss and global)
  r_bins: 256
  tau: 0.06
  band_sigma: 0.15
  w_peak: 1.0
  w_band: 1.0
  min_bin: 2
  max_bin: null
  
  # LogPolarAlignLoss parameters
  log_out_r: 128
  log_out_t: 180
  log_r_min: 0.01
  log_w: 0.1
  
  # ACFPeriodLoss parameters
  afc_r_bins: 256
  afc_tau: 0.08
  afc_min_rel: 0.03
  afc_max_rel: 0.6
  afc_w: 0.2
  
  # PhaseCorrelationLoss parameters (used by local loss)
  center_weight: 1.0
  pce_weight: 0.05
  temperature: 0.02
  use_hann: true
  demean: true
  exclude_radius: 7
```

## Training Tips for Custom Mode with Local Losses

1. **Start with local loss as primary**: Since you're using custom mode with local losses:
   - Set `loss_local_coef: 1.0` as the primary weight
   - Set `loss_Spect_coef: 0.5` for global spectral support
   - Keep other losses at 0.1-0.2
   - **IMPORTANT**: Make sure `local_patch_size: 192` (not 64!) - default was too small

2. **Adjust local loss parameters first**: These have the biggest impact:
   - **Patch size**: Match to your tile size
     - Large bricks: `local_patch_size: 256`
     - Medium tiles: `local_patch_size: 192` (default)
     - Small tiles: `local_patch_size: 128`
   - **ACF weight**: Increase `local_w_afc` to 2.0-2.5 for better period detection
   - **Phase weight**: Increase `local_w_phase` to 1.5-2.0 for better alignment

3. **If tiling is still lost**:
   - Increase `loss_local_coef` to 1.5-2.0 (primary fix)
   - Increase `local_w_afc` to 2.0-2.5 (better period detection in patches)
   - Increase `local_w_phase` to 1.5-2.0 (better patch alignment)
   - Increase `loss_Spect_coef` to 0.8-1.0 (global support)

4. **If model overfits to tiling**:
   - Reduce `loss_local_coef` to 0.5-0.7
   - Reduce `local_w_afc` to 1.0-1.2
   - Increase `loss_Spect_coef` to balance

5. **Fine-tune patch analysis**:
   - For very regular patterns: increase `local_w_afc` to 2.0-2.5
   - For irregular patterns: increase `local_w_phase` to 1.5-2.0
   - For mixed patterns: balance `local_w_spectral`, `local_w_afc`, `local_w_phase`

6. **Monitor training**: Watch the loss values printed every 50 steps:
   ```
   [step X] tex ON? beta=... gate=... base=... add=... custom and local
   ```
   - `add` should be relatively stable if tiling is preserved
   - If `add` increases, tiling might be degrading

7. **Use masks**: If you have masks for tiling regions:
   - The local loss will automatically focus on masked patches
   - Adjust `local_min_coverage` based on mask quality (0.3-0.6)

8. **Resolution considerations**: 
   - Higher resolutions (1024+): increase `local_patch_size` to 256-320
   - Lower resolutions (512): `local_patch_size: 128-192` works well
   - Adjust `r_bins` proportionally: 256 for 512px, 512 for 1024px+

## How Custom Mode with Local Losses Works

When using `texture_loss: "custom"`, the system combines multiple losses. Here's how each component works:

1. **LocalTextureRegularityLoss** (PRIMARY in custom mode):
   - **Extracts patches**: Breaks image into overlapping patches (e.g., 192x192 with stride 96)
   - **Random offset**: Applies random offset to patch grid to avoid grid artifacts
   - **Hard-mining**: Selects worst patches (where tiling breaks) using seam-proxy scoring
   - **Multi-method analysis**: For each patch, applies:
     - SpectralPeriodLoss (frequency analysis)
     - ACFPeriodLoss (autocorrelation)
     - PhaseCorrelationLoss (phase alignment)
     - LogPolarAlignLoss (scale/rotation invariance)
   - **Global peak center**: Uses global frequency peak to prevent local scale drift
   - **Soft masks**: Applies soft masks within patches to avoid edge artifacts
   - **Result**: Catches local tiling breaks that global methods might miss

2. **SpectralPeriodLoss** (global, secondary):
   - Analyzes the full image frequency spectrum
   - Detects dominant frequency (tile size)
   - Ensures model maintains global periodicity

3. **LogPolarAlignLoss** (global, tertiary):
   - Converts frequency domain to log-polar coordinates
   - Scale and rotation invariant
   - Helps with slightly rotated/scaled tiles

4. **ACFPeriodLoss** (global, tertiary):
   - Uses autocorrelation to detect periodic patterns
   - Complementary to spectral analysis
   - Catches periodicity missed by FFT

5. **PhaseCorrelationLoss** (used within local loss):
   - Phase alignment between patches
   - Helps maintain consistent tiling phase across patches

## Troubleshooting for Custom Mode with Local Losses

**Problem**: Tiling is still lost after enabling custom local losses
- **Solution**: 
  - Increase `loss_local_coef` to 1.5-2.5 (primary fix)
  - Increase `local_w_afc` to 2.0-2.5 (better period detection in patches)
  - Increase `local_w_phase` to 1.5-2.0 (better patch alignment)
  - Increase `loss_Spect_coef` to 0.8-1.0 (global support)
  - Check if `local_patch_size` matches your tile size

**Problem**: Model overfits to tiling and loses other details
- **Solution**: 
  - Reduce `loss_local_coef` to 0.5-0.7
  - Reduce `local_w_afc` to 1.0-1.2
  - Increase `loss_Spect_coef` to 0.8-1.0 to balance global vs local

**Problem**: Training is too slow
- **Solution**: 
  - Reduce `local_patch_size` to 128 (fewer patches)
  - Increase `local_stride` to 128 (less overlap, fewer patches)
  - Reduce `r_bins` to 128
  - Disable tertiary losses: `loss_Log_coef: 0.0`, `loss_AFC_coef: 0.0`

**Problem**: Tiling period is wrong (too large/small)
- **Solution**: 
  - Adjust `local_patch_size` to match actual tile size
  - Adjust `min_bin` and `max_bin` in SpectralPeriodLoss
  - Adjust `afc_min_rel` and `afc_max_rel` in ACFPeriodLoss
  - Adjust `local_w_afc` (higher = more sensitive to period)

**Problem**: Tiling breaks at patch boundaries
- **Solution**: 
  - Increase `local_stride` overlap (reduce stride relative to patch_size)
  - Increase `local_w_phase` to 1.5-2.0 (better phase alignment)
  - The random offset should help, but you can verify it's working

**Problem**: Only some patches preserve tiling
- **Solution**: 
  - Increase `local_min_coverage` to 0.5-0.6 (focus on patches with more tiling)
  - Increase `local_w_afc` to 2.0-2.5 (better period detection)
  - Increase `local_hard_k_frac` to 0.4-0.5 (analyze more patches)
  - Check if hard-mining is working (worst patches should be selected)

**Problem**: Texture losses seem too weak
- **Solution**:
  - Increase `max_beta` to 0.7-1.0 (stronger texture loss influence)
  - Decrease `gate` to 0.6 (apply texture losses earlier)
  - Check `v_scale` - should be 0.2-0.25 for Flux

**Problem**: Training is unstable with texture losses
- **Solution**:
  - Decrease `max_beta` to 0.3-0.4 (weaker texture loss)
  - Increase `gate` to 0.7-0.75 (apply only at very end)
  - Decrease `loss_local_coef` to 0.5-0.7

## Advanced Scientific Losses (NEW)

Beyond parameter tuning, we have implemented several **scientific improvements** that can significantly help with tiling preservation:

### 1. SeamlessLoss (`loss_seamless_coef`)

**Idea**: For a tileable texture, opposite edges must match. We explicitly compare left↔right and top↔bottom edges.

```yaml
train:
  loss_seamless_coef: 0.5  # Enable seamless loss (0.3-1.0)
  seamless_edge_width: 16   # How many pixels from edge to compare
  seamless_use_gradient: true  # Compare gradients (more robust)
  seamless_multi_scale: true   # Multi-scale analysis
```

**When to use**: When tiles have visible seams when repeated.

### 2. GradientPeriodLoss (`loss_gradient_coef`)

**Idea**: Analyze periodicity in gradient domain. Gradients often show cleaner periodicity than raw pixels, especially for textures with varying brightness.

```yaml
train:
  loss_gradient_coef: 0.3  # Enable gradient period loss (0.2-0.5)
```

**When to use**: When texture has varying brightness but consistent structure.

### 3. CrossPatchConsistencyLoss (`loss_cross_patch_coef`)

**Idea**: Local losses may allow different patches to have different periods ("period drift"). This loss penalizes variance in periods across patches.

```yaml
train:
  loss_cross_patch_coef: 0.3  # Enable cross-patch consistency (0.2-0.5)
  cross_patch_size: 128       # Patch size for analysis
  cross_patch_stride: 64      # Stride between patches
```

**When to use**: When different parts of the image have different tile scales.

### 4. MultiScaleSpectralLoss (`loss_multiscale_coef`)

**Idea**: Analyze tiling at multiple scales (pyramid) to catch large-scale repeat, medium patterns, and fine details.

```yaml
train:
  loss_multiscale_coef: 0.3   # Enable multi-scale spectral loss (0.2-0.5)
  multiscale_scales: [1.0, 0.5, 0.25]  # Scales to analyze
```

**When to use**: When texture has multi-level structure (e.g., bricks with mortar and surface texture).

### Complete Example with Advanced Losses

```yaml
train:
  texture_loss: "custom"
  
  # Base losses
  loss_local_coef: 1.0
  loss_Spect_coef: 0.5
  loss_AFC_coef: 0.2
  
  # Advanced scientific losses (NEW)
  loss_seamless_coef: 0.5      # Explicit seamless loss
  loss_gradient_coef: 0.3      # Gradient domain periodicity
  loss_cross_patch_coef: 0.3   # Cross-patch period consistency
  loss_multiscale_coef: 0.3    # Multi-scale analysis
  
  # SeamlessLoss params
  seamless_edge_width: 16
  seamless_use_gradient: true
  
  # CrossPatchConsistencyLoss params
  cross_patch_size: 128
  cross_patch_stride: 64
  
  # MultiScaleSpectralLoss params
  multiscale_scales: [1.0, 0.5, 0.25]
```

### Scientific Background

1. **SeamlessLoss**: Based on the principle that tileable textures must have continuous edges. Uses wrap-around boundary condition check.

2. **GradientPeriodLoss**: Applies Sobel operators before FFT. Gradient magnitude often has cleaner frequency peaks than raw luminance.

3. **CrossPatchConsistencyLoss**: Extracts patches, computes dominant period for each, penalizes variance in log-periods. Prevents "period drift".

4. **MultiScaleSpectralLoss**: Gaussian pyramid + spectral analysis at each level. Inspired by multi-scale texture synthesis literature.

## References

- Texture loss implementations: `toolkit/util/texture_losses.py`
- Local texture loss: `toolkit/util/local_texture.py`
- **Advanced tiling losses**: `toolkit/util/tiling_losses_advanced.py` (NEW)
- Training code: `extensions_built_in/sd_trainer/SDTrainer.py`
- Configuration: `toolkit/config_modules.py`

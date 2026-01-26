# Оптимальные настройки для обучения LoRA с сохранением тайлинга

> ⚠️ **ВАЖНО**: Эффективный вес текстурного лосса = `final_beta × gate_mean × coefficient`  
> При стандартных настройках это ~0.1-0.25 от указанного коэффициента!

## Как работает система активации

```
1. beta: 0→max_beta за всё обучение (warmup на весь train)
2. gate: активен только для timesteps < (1-gate), т.е. "чистые" шаги
3. Итого: effective_weight ≈ 0.1-0.25 × coefficient (в среднем)
```

## Быстрый старт — Минимальная конфигурация

```yaml
train:
  texture_loss: "custom"
  
  # Основной локальный лосс (ОБЯЗАТЕЛЬНО)
  loss_local_coef: 1.0          # Эффективно ~0.1-0.25
  
  # Вспомогательные глобальные лоссы
  loss_Spect_coef: 0.5          # Эффективно ~0.05-0.12
  loss_AFC_coef: 0.3            # Эффективно ~0.03-0.07
  loss_Log_coef: 0.0            # Обычно не нужен
  loss_Phase_coef: 0.0          # Уже есть внутри local
  
  # Критические параметры активации
  v_scale: 0.2                  # 0.15-0.3 (меньше = агрессивнее)
  min_beta: 0.0                 # Начальный вес текстурного лосса
  max_beta: 0.5                 # Финальный вес (можно до 0.7)
  gate: 0.65                    # Порог timestep (0.65 = последние 35%)
```

---

## Полная рекомендуемая конфигурация

### 1. Базовые параметры обучения

```yaml
train:
  steps: 2000-4000           # Для материалов обычно хватает 2000-3000
  lr: 1e-4                   # Стандартный LR для LoRA
  batch_size: 1              # Для 24GB VRAM
  gradient_accumulation: 4   # Эффективный batch = 4
  
  dtype: "bfloat16"          # Для Flux рекомендуется bf16
  gradient_checkpointing: true
  
  # Noise scheduler
  noise_scheduler: "flowmatch"  # Для Flux
  timestep_type: "sigmoid"
```

### 2. Texture Loss — режим `custom`

```yaml
train:
  texture_loss: "custom"
  
  # ============================================
  # ГЛАВНЫЙ ЛОСС — LocalTextureRegularityLoss
  # ============================================
  # Работает по патчам + hard mining = фокус на проблемных областях
  loss_local_coef: 1.0       # Основной вес (эффективно ~0.1-0.25)
  
  # Параметры патчей
  local_patch_size: 192      # Размер патча (для 1024x1024: 128-256)
  local_stride: 96           # Шаг (обычно patch_size / 2)
  local_min_coverage: 0.45   # Мин. покрытие маской для валидного патча
  
  # Hard mining (фокус на худших патчах)
  local_hard_k_frac: 0.30    # Доля худших патчей (0.2-0.4)
  local_hard_k_min: 4        # Минимум патчей для hard mining
  
  # Внутренние веса LocalTextureRegularityLoss (уже настроены!)
  local_w_spectral: 0.5      # Спектральный анализ частот
  local_w_afc: 1.5           # Автокорреляция (КЛЮЧЕВОЙ для периодов!)
  local_w_phase: 1.2         # Фазовая корреляция (выравнивание)
  local_w_log: 0.0           # Log-polar (редко нужен)
  
  # ============================================
  # ДОПОЛНИТЕЛЬНЫЕ ГЛОБАЛЬНЫЕ ЛОССЫ
  # ============================================
  # Работают на всём изображении, дополняют локальный анализ
  loss_Spect_coef: 0.5       # Глобальный спектр (0.3-0.7)
  loss_AFC_coef: 0.3         # Глобальная ACF (0.2-0.5)
  loss_Log_coef: 0.0         # Log-polar (обычно 0)
  loss_Phase_coef: 0.0       # Фаза (уже есть в local, обычно 0)
```

### 3. Критические параметры активации текстурного лосса

```yaml
train:
  # v_scale — масштаб velocity при декодировании для текстурного анализа
  # Формула: pred_x0 = x_t - t * (v_scale * v)
  # Меньше = более "консервативное" декодирование, меньше артефактов шума
  v_scale: 0.2               # 0.15-0.3 (по умолчанию 0.2)
  
  # beta — вес текстурного лосса, разогревается за всё обучение
  # На шаге 0: final_beta = min_beta
  # На шаге N (конец): final_beta = max_beta
  min_beta: 0.0              # Начальный вес (обычно 0)
  max_beta: 0.5              # Финальный вес (0.4-0.7)
  
  # gate — порог по timestep для активации
  # Текстурный лосс активен только когда (1 - t/T) > gate
  # gate=0.65 → активен для t < 35% от max_timesteps (малошумные)
  gate: 0.65                 # 0.5-0.7 (меньше = чаще активен)
```

**Понимание gate:**
- `gate = 0.65` → текстурный лосс работает на последних ~35% шагов денойзинга
- `gate = 0.5` → работает на последних ~50% шагов
- `gate = 0.8` → работает только на последних ~20% (очень чистые изображения)

### 4. Параметры спектрального анализа

```yaml
train:
  # Для SpectralPeriodLoss и внутри LocalTexture
  r_bins: 256                # Радиальные бины (256 для 1024px)
  tau: 0.06                  # Температура soft-peak
  band_sigma: 0.15           # Ширина частотной полосы
  w_peak: 1.0                # Вес пика
  w_band: 1.0                # Вес полосы
  min_bin: 2                 # Игнор DC компоненты
  max_bin: null              # Авто (0.4 * r_bins)
  
  # Для ACFPeriodLoss
  afc_r_bins: 256
  afc_tau: 0.08
  afc_min_rel: 0.03          # Мин. относительная частота
  afc_max_rel: 0.6           # Макс. относительная частота
  afc_w: 1.0
```

### 5. Параметры фазовой корреляции

```yaml
train:
  center_weight: 1.0         # Вес центрального пика
  pce_weight: 0.1            # Вес PCE метрики
  temperature: 0.02          # Температура softmax
  use_hann: true             # Окно Ханна (рекомендуется)
  demean: true               # Вычитание среднего
  exclude_radius: 7          # Радиус исключения DC
```

---

## Продвинутые лоссы (ОСТОРОЖНО!)

> ⚠️ Эти лоссы экспериментальные. Начинайте БЕЗ них, добавляйте по одному.

### CrossPatchConsistencyLoss (для перспективы)

```yaml
train:
  # Включить если материал на стене в перспективе
  loss_cross_patch_coef: 0.3   # 0.2-0.5 (начать с 0.3)
  
  cross_patch_size: 256        # Размер патча (больше = видит целые тайлы)
  cross_patch_stride: 128      # Шаг (половина patch_size)
```

**Когда использовать:** Если тайлинг "плывёт" — в одних местах крупнее, в других мельче.

### GradientPeriodLoss

```yaml
train:
  # ТОЛЬКО для матовых текстур без бликов
  loss_gradient_coef: 0.1      # Очень маленький вес!
```

**Когда использовать:** Матовые поверхности (кирпич, бетон). **НЕ использовать** для глянца!

### SeamlessLoss

```yaml
train:
  # ТОЛЬКО если маска касается ВСЕХ краёв изображения
  loss_seamless_coef: 0.0      # По умолчанию ВЫКЛЮЧЕН
  
  seamless_edge_width: 16
  seamless_use_gradient: true
  seamless_multi_scale: false  # Достаточно одного масштаба
```

**Когда использовать:** Только для rectified/выровненных плоскостей, покрывающих весь кадр.

### MultiScaleSpectralLoss

```yaml
train:
  # Пересекается с SpectralPeriodLoss — обычно НЕ НУЖЕН
  loss_multiscale_coef: 0.0    # ВЫКЛЮЧЕН
  
  multiscale_scales: [1.0, 0.5, 0.25]
```

---

## Рекомендации по типам материалов

> **Примечание**: Коэффициенты ниже учитывают, что эффективный вес = ~0.1-0.25 от указанного.
> Начинайте со значений по умолчанию, корректируйте по логам.

### 🧱 Кирпич / Камень (матовый, чёткая сетка)

```yaml
train:
  texture_loss: "custom"
  loss_local_coef: 1.0
  loss_Spect_coef: 0.5
  loss_AFC_coef: 0.5           # Повышен — важна строгая периодичность
  loss_cross_patch_coef: 0.5   # Для перспективы — разный масштаб
  
  local_w_spectral: 0.5
  local_w_afc: 2.0             # Сильный акцент на ACF
  local_w_phase: 1.0
  
  v_scale: 0.2
  max_beta: 0.6                # Чуть выше для сильного тайлинга
  gate: 0.6                    # Активнее
```

### 🪵 Дерево / Паркет (линейные паттерны)

```yaml
train:
  texture_loss: "custom"
  loss_local_coef: 1.0
  loss_Spect_coef: 0.6         # Важны направленные частоты
  loss_AFC_coef: 0.3
  loss_cross_patch_coef: 0.3
  
  local_w_spectral: 0.8        # Текстура волокон = спектр
  local_w_afc: 1.2
  local_w_phase: 1.0
  
  v_scale: 0.25
  max_beta: 0.5
  gate: 0.65
```

### 🔲 Плитка (глянцевая, чёткие границы)

```yaml
train:
  texture_loss: "custom"
  loss_local_coef: 1.0
  loss_Spect_coef: 0.7         # Чёткие края = сильный спектр
  loss_AFC_coef: 0.5
  loss_gradient_coef: 0.0      # НЕ использовать — создаст шум на глянце!
  
  local_w_spectral: 1.0
  local_w_afc: 2.0
  local_w_phase: 0.8
  
  v_scale: 0.15                # Консервативнее для глянца
  max_beta: 0.5
  gate: 0.7                    # Только чистые шаги
```

### 🏗️ Бетон / Штукатурка (стохастическая текстура)

```yaml
train:
  texture_loss: "custom"
  loss_local_coef: 0.5         # Слабее — нет строгого периода
  loss_Spect_coef: 0.3
  loss_AFC_coef: 0.1           # Минимум — нет явной периодичности
  
  local_w_spectral: 1.0        # Общая статистика важнее
  local_w_afc: 0.3
  local_w_phase: 0.5
  
  v_scale: 0.25
  max_beta: 0.4                # Мягче
  gate: 0.5                    # Чаще активен, но слабее
```

---

## Диагностика проблем

### Тайлинг "размывается" / теряется периодичность
```yaml
# Проверь логи: какой add vs base?
# Если add << base (например 0.001 vs 0.03), увеличить:
loss_AFC_coef: 0.3 → 0.7
local_w_afc: 1.5 → 2.5
max_beta: 0.5 → 0.7
gate: 0.65 → 0.55  # активнее
```

### Слишком "жёсткий" результат / артефакты сетки
```yaml
# Если add сравним с base — текстурный лосс слишком силён:
loss_AFC_coef: 0.5 → 0.2
local_w_afc: 2.0 → 1.0
max_beta: 0.5 → 0.3
gate: 0.65 → 0.75  # реже активируется
```

### Разный масштаб тайлов в разных частях (перспектива)
```yaml
# CrossPatch следит за консистентностью периода по патчам:
loss_cross_patch_coef: 0.5   # Включить/увеличить
cross_patch_size: 256        # Больше = видит целые тайлы
cross_patch_stride: 128
```

### Текстурный лосс НЕ активируется (beta всегда ~0)
```yaml
# Это НОРМАЛЬНО в начале обучения (warmup)!
# Если и в середине beta ≈ 0, проблема в gate:
gate: 0.65 → 0.5   # Активировать на большем диапазоне timesteps

# Также проверь что loss_local_coef > 0
loss_local_coef: 1.0  # Должен быть > 0!
```

### Логи показывают add=nan или add=inf
```yaml
# Численная нестабильность, уменьшить агрессивность:
v_scale: 0.2 → 0.3
tau: 0.06 → 0.1      # Более мягкий soft-peak
max_beta: 0.5 → 0.3
```

---

## Мониторинг обучения

В логах каждые 50 шагов выводится:
```
[step 500] tex ON? beta=0.4521 gate=0.78 base=0.0234 add=0.0156 custom and local
```

- **beta** — текущий вес текстурного лосса (0.0 = не активен)
- **gate** — значение гейта (выше = сильнее активация)
- **base** — базовый MSE лосс
- **add** — дополнительный текстурный лосс

**Хорошие признаки:**
- `beta` в диапазоне 0.2-0.6 большую часть времени
- `add` в 2-10 раз меньше `base`

**Плохие признаки:**
- `beta` всегда 0.0 → уменьшить `v_scale` и `gate`
- `beta` всегда ~1.0 → увеличить `v_scale`
- `add` >> `base` → уменьшить коэффициенты лоссов

---

## Пример полного конфига для кирпича

```yaml
job: extension
config:
  name: "brick_wall_tiling_lora"
  process:
    - type: "sd_trainer"
      
      model:
        name_or_path: "black-forest-labs/FLUX.1-dev"
        arch: "flux"
        quantize: true
        low_vram: true
        
      network:
        type: "lora"
        linear: 16
        linear_alpha: 16
        
      train:
        steps: 3000
        lr: 1e-4
        batch_size: 1
        gradient_accumulation: 4
        dtype: "bfloat16"
        gradient_checkpointing: true
        
        # === TEXTURE LOSS (режим custom) ===
        texture_loss: "custom"
        
        # --- Главный локальный лосс ---
        loss_local_coef: 1.0         # ОБЯЗАТЕЛЬНО > 0
        local_patch_size: 192
        local_stride: 96
        local_min_coverage: 0.45
        local_hard_k_frac: 0.30
        local_hard_k_min: 4
        
        # Внутренние веса local (по умолчанию хорошие)
        local_w_spectral: 0.5
        local_w_afc: 1.5             # Ключевой для периодичности
        local_w_phase: 1.2
        local_w_log: 0.0
        
        # --- Глобальные лоссы ---
        loss_Spect_coef: 0.5
        loss_AFC_coef: 0.3
        loss_Log_coef: 0.0
        loss_Phase_coef: 0.0
        
        # --- CrossPatch (для перспективы) ---
        loss_cross_patch_coef: 0.5   # Важно для стен в перспективе!
        cross_patch_size: 256
        cross_patch_stride: 128
        
        # --- Параметры активации ---
        v_scale: 0.2                 # Антишум при декодировании
        min_beta: 0.0                # Начало warmup
        max_beta: 0.5                # Конец warmup
        gate: 0.65                   # Порог timestep
        
        # --- Спектральные параметры (по умолчанию ок) ---
        r_bins: 256
        tau: 0.06
        band_sigma: 0.15
        min_bin: 2
        
      datasets:
        - folder_path: "/path/to/brick_images"
          resolution: 1024
          caption_ext: ".txt"
          # Важно: маска должна покрывать только стену!
          mask_path: "/path/to/masks"  # Если есть маски
          
      save:
        save_every: 500
        dtype: "float16"
```

---

## Чеклист перед запуском

- [ ] `texture_loss: "custom"` установлен
- [ ] `loss_local_coef > 0` (иначе текстурный лосс не работает!)
- [ ] Маски корректны (если используются)
- [ ] Датасет содержит качественные примеры тайлинга
- [ ] `steps` достаточно (2000-4000 для материалов)

## Что смотреть в логах

```
[step 500] tex ON? beta=0.0833 gate=0.42 base=0.0234 add=0.0089 custom and local
                   ↑           ↑         ↑          ↑
                   │           │         │          └─ Текстурный лосс
                   │           │         └─ Базовый MSE лосс
                   │           └─ Среднее значение gate (0=выключен)
                   └─ Текущий beta (растёт за обучение)
```

**Хорошо:** `add` в 2-5 раз меньше `base`, `gate` > 0.3
**Плохо:** `add = 0.0000` или `add >> base`

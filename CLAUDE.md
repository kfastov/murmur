# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Murmur — нативное macOS menubar-приложение для speech-to-text. Работает полностью локально на Apple Silicon через whisper.cpp (C API + Metal GPU). Один .app файл, zero dependencies, drag-to-install.

**Версия:** 3.26 (native Swift)
**Платформа:** macOS 14+ (Sonoma), Apple Silicon (M1/M2/M3/M4)
**Стек:** Swift 6 + SwiftUI + whisper.cpp (static linking) + Metal + FluidAudio (Parakeet Core ML / ANE) + mlx-audio-swift (Voxtral 4B, Qwen3-ASR 1.7B на MLX/GPU)
**UI:** Liquid Glass (macOS 26 Tahoe) с fallback на ultraThinMaterial для macOS 14–15

## Quick Start

```bash
# Build + create .app bundle with icon, code-sign, install в /Applications
./build-app.sh

# Run
open /Applications/Murmur.app
# или для отладки со stdout: .build/release/Murmur

# Plain SPM build (без bundle)
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build -c release

# Tests (SPM target MurmurTests)
swift test
```

**Использование:** Option+Space → говори → Option+Space → текст вставляется в активное поле.

> [!warning] Полный Xcode обязателен (не только CLI tools) — нужен Metal framework. Если стоит не он:
> `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer` или передавай `DEVELOPER_DIR=` каждой команде сборки.

## Architecture

### Data Flow
```
Option+Space → AppState.toggle()
  → AudioRecorder.start() [AVAudioEngine, native format 44.1kHz]
    → tap callback: copy PCM samples + compute RMS for waveform
  → Option+Space again
  → AudioRecorder.stop()
    → resample 44.1kHz → 16kHz (linear interpolation)
    → normalize to peak ~0.9
    → return [Float]
  → TranscriptionEngine.transcribe([Float])
    → whisper_full() [C API, Metal GPU]
    → extract segments → joined text
  → TextPaster.paste(text)
    → NSPasteboard + CGEvent Cmd+V
    → restore clipboard after 0.5s
```

### State Machine
```
.loading → (model loads) → .idle → (Option+Space) → .recording → (Option+Space) → .transcribing → .idle
                                                       ↓ (Escape)
                                                     .idle
```

### Threading Model
- **AppState** — `@MainActor` singleton
- **AudioRecorder** — audio tap runs on audio thread, `NSLock` protects `rawSamples[]` and `levelsCallback`
- **TranscriptionEngine** — Swift `actor` (serialized access to whisper context)
- **WaveformPanel** — `@MainActor`

## File Structure

```
Murmur/
├── Package.swift              # SPM manifest: CWhisper + Murmur targets
├── CLAUDE.md                  # This file
├── Info.plist                 # App bundle metadata (icon, permissions, LSUIElement)
├── build-app.sh               # Build script: swift build + .app bundle assembly
├── Murmur-3.1.dmg            # Release DMG installer
├── Assets/
│   └── AppIcon.icns           # App icon (1930s cartoon cat, all sizes 16-1024px)
├── Sources/
│   ├── Murmur/
│   │   ├── Murmur.swift       # @main, MenuBarExtra, NSApplicationDelegateAdaptor
│   │   ├── AppState.swift     # Singleton state, setup(), toggle(), cancel(), mlog()
│   │   ├── AudioRecorder.swift    # AVAudioEngine capture, resample, normalize, device selection
│   │   ├── InputDeviceManager.swift  # CoreAudio input device enumeration
│   │   ├── TranscriptionEngine.swift  # whisper.cpp wrapper (actor)
│   │   ├── HotkeyManager.swift   # Carbon RegisterEventHotKey (Option+Space)
│   │   ├── TextPaster.swift       # NSPasteboard + CGEvent Cmd+V
│   │   ├── MenuBarView.swift      # SwiftUI menu (model select, mic select, quit)
│   │   ├── WaveformView.swift     # Liquid Glass UI: waveform bars + OrbitalLoader
│   │   └── WaveformOverlay.swift  # NSPanel floating window
│   └── CWhisper/
│       ├── module.modulemap   # Links whisper + ggml libs
│       ├── shim.c             # Empty (required by SPM)
│       └── include/shim.h     # Re-exports whisper.h
├── include/                   # whisper.cpp C headers (whisper.h, ggml*.h)
├── lib/                       # Static libraries (.a files)
│   ├── libwhisper.a
│   ├── libggml.a, libggml-base.a, libggml-cpu.a
│   ├── libggml-metal.a, libggml-blas.a
│   └── ggml-metal-embed.metal # Metal shaders (embedded at link time)
└── Tests/MurmurTests/
```

## Key Components

### AppState.swift
- `mlog()` — file-based logging to `~/.murmur-debug.log` (NSLog doesn't work for menubar apps)
- `startSetup()` — called from NSApplicationDelegate after 0.5s delay (MenuBarExtra `.task`/`.onAppear` don't fire)
- `setup()` — checks mic permission, checks Accessibility (`AXIsProcessTrustedWithOptions` с prompt), registers hotkeys, loads model
- `toggle()` — state machine: idle→recording→transcribing→idle
- Accidental press protection: recording < 1s → auto-cancel

### AudioRecorder.swift
- **CRITICAL:** PCM buffers from AVAudioEngine tap are REUSED by the engine. Must copy samples immediately in callback, NOT store buffer references.
- **Capture is AudioQueue, not AVAudioEngine** (since 2026-09-26). Asks for 16 kHz mono Float32 directly, so there is no resampler. Device pick: `kAudioQueueProperty_CurrentDevice` with the device UID, skipped when the pick is the system default. Saved in `UserDefaults("inputDeviceUID")`.
- **Why not AVAudioEngine:** opening a Bluetooth headset mic flips it A2DP 44.1/48 kHz → HFP 16 kHz about a second after start. AVAudioEngine's input format follows the default OUTPUT device, so the tap's format changes under it, `AVAudioEngineConfigurationChange` fires and the tap never fires again → empty take, Whisper hallucinates "Субтитры создавал…". Rebuilding the engine on that notification loops: each rebuild re-opens the headset and triggers the next change. Reusing the engine reads a stale format and crashes in `installTap`. AudioQueue delivers straight through the flip. Harness: `Benchmarks/config-change/`.
- Native mic format: 44100Hz mono (on most Macs)
- Resampling: linear interpolation to 16kHz
- Normalization: scale peak to 0.9 (mic raw levels are ~0.03-0.05, too quiet for whisper)
- Waveform: 11 bars with symmetric weights [0.3..1.0..0.3], RMS * 15.0

### InputDeviceManager.swift
- CoreAudio device enumeration: lists all devices with input channels
- Built-in detection: `kAudioDeviceTransportTypeBuiltIn` OR UID contains "builtin" (some Macs report non-standard transport type)
- Default: built-in microphone (persists even when Bluetooth speaker connected)
- Selection saved in `UserDefaults("inputDeviceUID")`

### TranscriptionEngine.swift
- Downloads GGML models from HuggingFace on first launch
- Models stored in `~/Library/Application Support/Murmur/models/`
- Warmup: transcribes 1s silence on load (triggers Metal shader JIT)
- `flash_attn = false` (true causes compute buffer size mismatch)
- `initial_prompt` — guides punctuation style

### HotkeyManager.swift
- Carbon Events API: `RegisterEventHotKey()` / `UnregisterEventHotKey()`
- Option key = `optionKey` modifier, Space = keyCode 49
- Global hotkey works in any app, any space

### TextPaster.swift
- Saves current clipboard → sets text → simulates Cmd+V → restores clipboard after 0.5s
- CGEvent with `.maskCommand` flag, virtual key 0x09 (V)
- Debug logging: clipboard verify, CGEvent creation check, `AXIsProcessTrusted()` status
- **Requires Accessibility permission** (System Settings → Privacy & Security → Accessibility)

### WaveformView.swift
- **Recording:** Liquid Glass capsule (`.glassEffect(.regular, in: Capsule())` on macOS 26+) с fixed-height pill (68pt), бары `Color.primary` анимируются внутри
- **Transcribing:** Orbital loader — 8 точек `Color.primary` на внешней орбите + 5 на внутренней (counter-rotating) + пульсирующее центральное свечение, всё внутри Liquid Glass circle
- **Адаптивные цвета:** все элементы используют `Color.primary` — чёрные в светлой теме, белые в тёмной
- Fallback для macOS 14–15: `.ultraThinMaterial` вместо `.glassEffect()`
- Spring-анимации появления/скрытия (response: 0.35, damping: 0.78)

### WaveformOverlay.swift
- NSPanel: borderless, floating, ignoresMouseEvents, works on all spaces
- Size: 260×140 pt, positioned center + 60pt from bottom
- Hide delay: 0.4s (для spring-анимации)

## Models

| Name | File | Size | Speed |
|------|------|------|-------|
| turbo | ggml-large-v3-turbo.bin | ~1.5 GB | Fast |
| large | ggml-large-v3.bin | ~3 GB | Best quality |

Source: `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/`
Local: `~/Library/Application Support/Murmur/models/`

## Known Issues & Workarounds

### whisper.cpp 1.8.4: detect_language bug
**Problem:** `detect_language=true` + `language="auto"` → 0 segments (decoder produces nothing)
**Workaround:** Use explicit `language="ru"`, `detect_language=false`
**Impact:** Currently hardcoded to Russian. To support other languages, set language explicitly.

### AVAudioEngine buffer reuse
**Problem:** `AVAudioPCMBuffer` from tap callback is recycled by the engine after callback returns. Storing buffer references leads to corrupted/zero data.
**Fix:** Copy float samples into `[Float]` array immediately inside the callback using `Array(UnsafeBufferPointer(...))`.

### Mic audio levels very low
**Problem:** Raw mic input on MacBook is max ~0.03-0.05 (whisper needs ~0.1+)
**Fix:** Normalize audio to peak 0.9 before passing to whisper. Gain capped at 50x to avoid amplifying pure noise.

### MenuBarExtra lifecycle
**Problem:** `.task{}` and `.onAppear{}` don't fire for MenuBarExtra content until menu is opened.
**Fix:** Use `@NSApplicationDelegateAdaptor` with `applicationDidFinishLaunching` + `DispatchQueue.main.asyncAfter(0.5s)`.

### LaunchServices binary cache
**Problem:** macOS caches app binary signatures. After rebuilding, the old version may launch.
**Fix:** Change CFBundleIdentifier or delete old .app and create at new path. `lsregister -kill` or `open -n` may help.

### mlog instead of NSLog
**Problem:** NSLog/os.log are filtered by macOS for GUI apps (info level messages don't appear).
**Fix:** File-based logging to `~/.murmur-debug.log`.

### Accessibility permission reset after rebuild
**Problem:** macOS ties Accessibility permission to the app binary signature. After `./build-app.sh` the binary changes and `AXIsProcessTrusted()` returns `false` → Cmd+V paste fails silently.
**Fix:** `build-app.sh` now code-signs with a stable developer identity (`codesign --force --deep --sign`), so permission persists across rebuilds. If signing identity is unavailable, falls back to ad-hoc (`sign -`) and permission will need re-granting.
**First launch:** `AXIsProcessTrustedWithOptions` with prompt automatically shows system dialog asking user to grant Accessibility.

## Build & Distribution

### Prerequisites
- Xcode (full, not just Command Line Tools) — needed for Metal framework
- If Xcode path is wrong: `DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer`

### Build
```bash
cd ~/Desktop/Murmur
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer swift build -c release
```

### Create .app Bundle
```bash
./build-app.sh
```
Скрипт выполняет: `swift build -c release` → создаёт `Murmur.app/` → копирует бинарник, `Info.plist`, `Assets/AppIcon.icns` → code sign (developer identity или ad-hoc) → копирует в `/Applications/`.

**Info.plist** (уже в репо):
- `CFBundleIconFile = AppIcon` — иконка приложения
- `LSUIElement = true` — hide from Dock (menubar-only)
- `NSMicrophoneUsageDescription` — mic permission string
- `NSHighResolutionCapable = true`
- `CFBundleExecutable = Murmur`

### Create DMG
```bash
DMG_DIR="/tmp/murmur-dmg"
mkdir -p "$DMG_DIR"
cp -R /Applications/Murmur.app "$DMG_DIR/"
ln -s /Applications "$DMG_DIR/Applications"
hdiutil create -volname "Murmur" -srcfolder "$DMG_DIR" -ov -format UDZO Murmur-3.1.dmg
```
DMG собирается из подписанного бандла в `/Applications/`.

### Static Libraries
Built from whisper.cpp source:
```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
cmake -B build -DGGML_METAL=ON -DBUILD_SHARED_LIBS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
# Copy lib/*.a and include/*.h to Murmur project
```

## Whisper.cpp C API Reference (used in this project)

```c
// Context
whisper_context_params whisper_context_default_params(void);
whisper_context* whisper_init_from_file_with_params(const char* path, whisper_context_params params);
void whisper_free(whisper_context* ctx);

// Transcription
whisper_full_params whisper_full_default_params(enum whisper_sampling_strategy);
int whisper_full(whisper_context* ctx, whisper_full_params params, const float* samples, int n_samples);

// Results
int whisper_full_n_segments(whisper_context* ctx);
const char* whisper_full_get_segment_text(whisper_context* ctx, int i_segment);
```

**Key params:**
- `whisper_context_params.use_gpu = true` — Metal acceleration
- `whisper_context_params.flash_attn = false` — disabled (compatibility)
- `whisper_full_params.language = "ru"` — explicit language (NOT "auto")
- `whisper_full_params.detect_language = false` — disabled (bug workaround)
- `whisper_full_params.initial_prompt` — guides punctuation/style
- `whisper_full_params.no_context = true` — don't carry context between calls
- `whisper_full_params.suppress_blank = false` — don't drop quiet segments

## History

- **v3.26**: Две правки. **(1) Тёмная подложка свечения переусилена.** На белом фоне glow тонул — scrim с v3.23 был почти прозрачным (0.20). Поднят и ЖЁСТКО привязан к голосу: `scrimBaseOpacity 0.20→0.34`, `scrimIntensityGain 0.10→0.54`, `scrimStroke 26→32`. Диапазон непрозрачности тишина≈0.37 → речь≈0.61 → пик≈0.88. Привязка gain≫base делает scrim *тенью свечения* (темнеет вместе с яркостью при речи, в тишине еле виден), поэтому контраст на белом сильный, а грязного тёмного облака на цветном фоне нет; шейдерный alpha-floor 0.40 гарантирует что glow всегда тонирует scrim, raw-black не просвечивает. **(2) Bugfix: разрешение на вставку не применялось без перезапуска.** Репорт: после установки новой ad-hoc-сборки поверх старой macOS сбрасывает TCC-грант (сменился cdhash), а выданное на лету разрешение `CGPreflightPostEventAccess()` в работающем процессе НЕ подхватывает — macOS применяет PostEvent/Accessibility только при старте процесса. Предупреждение v3.25 врало («снимается без перезапуска»). Теперь: честный текст («включите в Accessibility, затем перезапустите — macOS применяет только при запуске»), вторая кнопка в меню «Restart Murmur to apply the grant» → `AppState.relaunch()` (новый instance через `NSWorkspace.openApplication` + `createsNewApplicationInstance`, затем terminate — sandbox-safe для MAS-сборки). README: шаг «перезапуск обязателен», объяснение слёта гранта при каждом обновлении ad-hoc-подписи, `tccutil reset Accessibility com.deagle.murmur` для залипшего тумблера. Настоящий корень (грант не слетает) — стабильная Developer ID подпись, придёт с нотаризованной/store-сборкой.
- **v3.25**: Видимое предупреждение при отсутствии разрешения на вставку. Корень класса жалоб «детекция не прошла, текст не вставился, Escape не работает» (реальный кейс: второй MacBook Jameson, GitHub-сборка): без post-events/Accessibility разрешения диктовка ВЫГЛЯДИТ рабочей (glow анимируется, whisper распознаёт), но auto-paste И глобальный Escape-монитор мертвы молча — `CGPreflightPostEventAccess()` false → paste оставляет текст в буфере и выходит; `NSEvent.addGlobalMonitorForEvents(keyDown)` без Accessibility вообще не получает клавиш. Системный prompt показывается ОДИН раз (отклонён → тихий вечный отказ до ручного включения); ad-hoc-сборкам macOS иногда не показывает prompt вовсе. Теперь: `AppState.canPostEvents` (@Published, refresh в setup/каждый toggle/после каждого paste — выдача разрешения снимает предупреждение без перезапуска), в меню первой строкой warning-кнопка «No paste permission — open System Settings» (`openPastePermissionSettings()`: повторный `CGRequestPostEventAccess` если prompt ещё возможен, иначе `x-apple.systempreferences:...Privacy_Accessibility`), под ней подсказка про буфер (Cmd+V вручную) и Escape. `TextPaster.hasPermission` — preflight без prompt'а. README: paste-секция расширена (one-shot prompt, быстрая диагностика «продиктуй и нажми Cmd+V»), новая секция «Escape doesn't cancel» (то же разрешение).
- **v3.24**: Dynamic Island на экранах без челки. На внешних мониторах и MacBook'ах до 2021 (M1 Air/Pro 13") glow раньше обводил «призрачную» челку 200pt — контур вокруг пустоты. Теперь: чёрный остров (185pt — ширина реальной челки MBP 16", высота = menubar этого экрана с clamp [20..38] и fallback 30pt при скрытом menubar) выезжает из верхнего края на время диктовки, glow обнимает его контур. **Детекция канонична** (паттерн Ice/NotchDrop/boring.notch): `safeAreaInsets.top > 0` — первичный сигнал наличия челки (macOS 12+), `auxiliaryTopLeft/RightArea` — только для ширины выреза; у настоящей челки `topInset` теперь тоже из `safeAreaInsets.top`, а не из menubar-гэпа (гэп обнуляется при скрытом menubar/fullscreen, челка-то остаётся). Notched MacBook со scaled-разрешением «ниже камеры» (16:10) сам сообщает нулевые insets → остров включается автоматически, отдельной обработки нет. `WaveformPanel.hasNotch` (@Published) пересчитывается на каждый show() по экрану с курсором — между диктовками можно мигрировать между мониторами. **Рендер острова**: чёрная заливка той же shape рисуется ПОВЕРХ glow-строуков — закрывает внутреннюю половину stroke'ов ровно как физический безель (stroke центрован на пути); намеренно БЕЗ shadow — мягкая тёмная кромка поверх аддитивного glow вернула бы «грязный» класс багов v3.23, отделение от фона дают scrim + сам glow. **Форма** `IslandShape` (новый Shape в WaveformView.swift): плоский верх, нижние углы 12pt (у реальной челки 8pt, но при высоте ~30pt нужно больше), и вогнутые **флейры 4pt** на верхних углах — остров «вытекает» из края экрана как настоящая челка, а не приклеен под 90° (фидбек Jameson после первой итерации). Кривые через `addQuadCurve` с control в остром углу (не `addArc` — flipped-координаты углов в Path это поле грабель). Shape бранчится через `AnyShape`: у реальной челки прежний `UnevenRoundedRectangle` нетронут. **Появление** (дотюнено живым фидбеком в 3 итерации): выезд из-за верхнего края (`.offset(y: -(topInset+12))`, панель flush с краем экрана клипует контент выше кромки) + лёгкий scale-акцент от 0.75 (от 0.4 читалось «вырастает из слишком маленького») + opacity; кривая emphasized-decelerate `.timingCurve(0.05, 0.7, 0.1, 1)` за 0.3s (0.45 казались вялыми, сток `.easeOut(0.22)` — почти линейным рывком). У челки прежний чистый fade — железу двигаться нечем. Весь glow-стек v3.23 (шейдер, палитры, envelope, страховки) переиспользован без изменений.
- **v3.23**: Siri-grade morphing glow. Заливка glow-контура переведена с бегущего `FlowingGradient` на кастомный Metal-шейдер `Sources/Murmur/Shaders/GlowField.metal` (domain-warped fBM colour field, техника Inigo Quilez): органично морфящиеся переливающиеся пятна вместо линейной ленты. Шейдер вызывается через SwiftUI `.colorEffect` (macOS 14+), без собственного таймера: время приходит uniform'ом из существующего `TimelineView`-интегратора. Меняется ТОЛЬКО заливка: маски stroke-контура челки, двухслойность halo+crisp, scrim, blur, EMA-level, morph recording↔transcribing через `transitionProgress` и все страховки v3.20/v3.21 нетронуты. Голос теперь правит три канала поля: скорость эволюции (`fieldTime`, тот же интегратор, но без wrap по 1: шумовое время не циклично, wrap дал бы снап; страховочный wrap по 2048 + reset в `onAppear`), хаотичность `warpAmp` (1.1 тишина → 3.5 крик, кривая 0.55), яркость (0.75 → 1.2). Палитра для шейдера: 4 стопа (seam не нужен, цвета мешаются по значениям поля, не по координате) против 5-стоповой для градиента, лерпятся тем же `PaletteRGB`-с-clamp. `+ Reduce Motion` guard в `advancePhase` (статичное поле, реагирует только яркостью). **Сборка metallib**: оба `swift build`-скрипта (`build-app.sh`, `build-mas.sh`) не запускают Metal-компилятор, поэтому явная компиляция `xcrun metal -std=metal3.0` + `xcrun metallib` → `Murmur.app/Contents/Resources/default.metallib` (в build-mas компиляция идёт до codesign, чтобы попасть под подпись Resources). `GlowShaderSupport` ищет metallib в `Bundle.main` ПЕРВЫМ (основной путь), SPM-бандл вторичен. **Fallback**: при отсутствии metallib (plain `swift build`-бинарь из `.build/release`) `GlowShaderSupport.isAvailable == false` → автоматический откат на сохранённый `FlowingGradient` целиком, приложение работает со старым визуалом v3.21, не с чёрным/белым экраном. build-app.sh при провале компиляции шейдера предупреждает но не падает (fallback); build-mas.sh падает жёстко (store-сборка не должна тихо отгружать деградированный визуал).
- **v1.0** — Hammerspoon + Python daemon (mlx-whisper). Worked but required Hammerspoon + Python + Homebrew.
- **v2.0** — Published to GitHub with install script, DMG, README.
- **v3.0** — Native Swift app. Single .app, no dependencies. whisper.cpp C API + Metal GPU.
- **v3.1** — App icon (1930s cartoon cat). Liquid Glass UI (macOS Tahoe). Orbital transcription loader. Adaptive colors (`Color.primary` — dark/light theme). Input device selector (Microphone menu, defaults to built-in). Accessibility auto-prompt on first launch. Code signing for stable permissions. Thread-safety fixes (AudioRecorder race condition, model switch guard). Cleanup on quit. Auto-install to `/Applications/`.
- **v3.2** — Multi-monitor fix: waveform overlay centers on the screen with the cursor (was using `NSScreen.main` once at panel creation, drifted on display changes). Use-after-free fix on model switch (`whisper_full` SIGSEGV when Option+Space hit during 3 GB large download — ctx is now nil'd before `whisper_free`, and switching holds `state = .loading` for the entire download+init). Model download progress in menu bar (size, speed, ETA via `URLSessionDownloadDelegate` with EMA-smoothed bytes/sec). Persists last-used model across launches (UserDefaults `activeModel`).
- **v3.3** — Pause / Resume in the menu: free the whisper context (~1.5–3 GB CPU RAM + Metal GPU buffers) without quitting the app. New `.paused` `RecordingState`; `AppState.pauseModel()` calls `engine.cleanup()`, `resumeModel()` reuses the normal load path so the user gets the same progress UI. Option+Space is a no-op while paused (logged) — picking the active model in the menu also resumes. Model menu and Microphone menu remain enabled in `.paused` so users can switch model directly from a paused state.
- **v3.21** — Полная замена notch-pill'а на **ambient light glow вокруг челки**. Это не довесок, а смена концепции визуала. **Архитектура:** NSPanel transparent, размер `notchWidth + 2 × glowPaddingSides (160) × topInset + glowPaddingBottom (170)` — большой запас под Gaussian blur falloff на пике. Внутри `WaveformView` рисует `UnevenRoundedRectangle` (нижние углы 22pt) поверх горизонтально-летящего `LinearGradient`. Контур shape расширен относительно физической челки на `glowExtraSides 12 / glowExtraBottom 16` чтобы stroke шёл снаружи челки (без расширения видны цветные уголки из-под скруглений). **Композиция рендера**: `ZStack` из двух `FlowingGradient` (outer halo + inner crisp line), каждый заполняет outer frame `shapeSize + 2 × blurSpill (140)` цветом, маска — `shape.stroke().frame(shapeSize).frame(outerSize, alignment: .top)` (двойной frame трюк чтобы stroke сел на нужную позицию внутри расширенного фрейма), `.blur` применяется к ИТОГУ (gradient + mask), не к маске — так blur имеет gradient-пиксели вокруг себя для растекания, не клипается прямоугольником. **FlowingGradient**: HStack из 3 копий LinearGradient с `frame(width: w*3).offset(x: -w + phase*w)`, без `.clipped()` чтобы overflow попадал в blur-зону. Phase накапливается через `@State phase + lastPhaseUpdate`, мутируется в `Color.clear.onChange(of: now)` внутри `TimelineView(.animation)` (нельзя мутить @State прямо в body), `dt` clamped [0, 0.1s] для защиты от хиккапов. **Динамика от голоса**: `pow(smoothedLevel, 0.5)` (sqrt curve) бустит низкие/средние уровни — шёпот даёт уже 37% буста скорости вместо 10% линейно. Скорость потока 0.04 cps (тишина, 25s/цикл) → 1.24 cps (крик, 0.8s/цикл). Интенсивность glow (stroke 1→37pt, blur 3→49pt, halo opacity 0.4→0.9) — ~12× динамический диапазон. EMA smoothing 0.4 (snappy реакция на каждый слог). **Плавный morph между recording↔transcribing**: единый `@State transitionProgress: CGFloat` (0→1) с `withAnimation(.easeInOut(duration: 0.55))` в `onChange(of: panel.mode)`. Палитра, intensity и phase speed все lerp'аются через этот один progress в синхроне. Палитра хранится как `PaletteRGB(r, g, b)` (не `Color` — opaque token в SwiftUI без доступа к компонентам) для покомпонентной интерполяции, с clamping `t ∈ [0, 1]` против overshoot spring'ов. **Bugfix критический (страж от регрессии v3.20)**: `TimelineView(.animation)` живёт ВНУТРИ `if panel.isVisible`, не снаружи. Иначе timeline тикал бы 60fps вечно после `hide()` (panel order out не уничтожает SwiftUI view tree), каждый тик вызывал бы `currentIntensity()`, `advancePhase()`, `onChange(of: now)` — фоновая работа на main runloop в always-on menubar app, ровно тот класс что в v3.20 убил Carbon Option+Space callback. `.transition(.opacity)` на TimelineView позволяет `withAnimation(hideAnimation) { isVisible = false }` в `WaveformPanel` всё ещё анимировать fade-out. `.onAppear` сбрасывает `phase = 0, lastPhaseUpdate = 0` для детерминированного старта каждой новой записи. **Удалено** мёртвый @Published `levels: [Float]` — старый bar-renderer его читал, glow читает только `smoothedLevel`. Раньше `objectWillChange` срабатывал 60 раз/сек без потребителя — лишняя SwiftUI работа.
- **v3.20** — Большая переработка анимации внутри notch-pill'а. **Бары теперь Capsule с минимальной высотой = диаметру** (`max(barWidth, baseHeight*level)`) — крайние при тихом звуке отрисовываются как идеальные точки, а не тонкие полоски (раньше при level=0.15 крайние были 1.2pt, читались как линии). Базовые высоты подняты с `[8…36…8]` до `[12…40…12]` чтобы разница «точка ↔ пик» была заметнее. **OrbitalLoader удалён полностью** — теперь один HStack из 11 capsules обслуживает оба режима. Transcribing: волна исчезновения **парами от краёв к центру** через `.timingCurve(0, 0, 0.2, 1, duration: 0.22).delay(distance × 0.045)` (Material decelerate Bezier, на коротких длительностях читается чётче чем SwiftUI `.easeOut`). Edges уходят через scale+opacity (не height-collapse — иначе они зрительно «стекают в центр»). Центральная точка превращается в круг **мгновенно** (`modeAnimation = nil` для центра) — без этого она синхронно сжималась вместе с волной. Pulsing dot: двухтактный цикл через withAnimation completion handler — `pulseUp()`: `.bouncy(duration: 0.42, extraBounce: 0.35)` peak scale 2.0 (даёт пружинистый overshoot), `pulseDown()`: `.easeOut(duration: 0.35)`. Цикл стартует ровно когда исчезает пара (4, 6) — 0.4s после перехода mode. **Геометрия pill**: 89 → 81pt (`topInset + maxBarHeight + 4pt slack`). HStack обёрнут в `.frame(height: 40)` (фиксированная высота бар-зоны), затем `.frame(maxHeight: .infinity, alignment: .top)` — это прижимает зону под челку, пик центрального бара flush с notch'ом, 4pt дыхания снизу. Без `frame(height: 40)` HStack тянулся с RMS, его центр уезжал, вся линия баров «смещалась» при громком звуке. **Bugfix (критический)**: `WaveformPanel.hide()` теперь сбрасывает `mode = .recording`. Раньше mode оставался `.transcribing`, pulse loop через completion handler перевзводился бесконечно `guard panel.mode == .transcribing` — main runloop забивался через 1-2 минуты, Carbon `RegisterEventHotKey` callback переставал срабатывать на Option+Space (события теряли приоритет в очереди). Один цикл = 4 main-thread callbacks за 0.77s — на одной транскрипции невидимо, после второй уже залипал хоткей. Fix через явный mode reset в hide() — guard ломает бесконечную цепочку.
- **v3.19** — Waveform overlay переделан в стиль **"раздвигающейся чёлки"** (notch). Раньше был Liquid Glass capsule внизу экрана. Теперь — solid-чёрная панель сверху по центру: top edge flush с верхом экрана, скруглены **только нижние углы** (radius 22pt). На MacBook'ах с физической чёлкой это читается как notch "раздвигается" вниз и показывает контент. На Mac'ах без чёлки выглядит как Dynamic-Island-style таб. Изменения: `WaveformOverlay.swift` panel size 260×140 → 320×78, position bottom → top center (`y = f.maxY - height`), level `.floating` → `.statusBar` (поверх menubar), `frame` вместо `visibleFrame` (нам нужно перекрыть menubar area). `WaveformView.swift` — `UnevenRoundedRectangle` для чёрной формы со скруглением только снизу, белые бары на чёрном фоне (раньше `Color.primary` адаптивный — на чёрном фоне в light mode он был бы чёрным), spring transition с `anchor: .top` чтобы анимация роста шла из верхнего края (вид «раздвижения»). `OrbitalLoader` тоже белый на чёрном. Liquid Glass обёртка удалена (она была для floating capsule, для notch-стиля нужна solid opacity).
- **v3.18** — Fix: Escape для отмены записи работал только когда Murmur — фронтовое приложение (через `NSEvent.addLocalMonitorForEvents`). А пользователь почти всегда диктует в чужое окно (Telegram, браузер и т.д.), и Escape уходил туда, запись не отменялась. Добавлен **второй** монитор — `NSEvent.addGlobalMonitorForEvents` — который ловит Escape когда любое другое приложение в фронте и вызывает `cancel()`. AppKit не позволяет global monitor'у поглощать события (callback возвращает Void), так что Escape всё равно доходит до фронтового приложения — в большинстве случаев он там безобиден (закрывает модалку / снимает фокус). Local monitor сохранён на случай когда менюбар-dropdown открыт.
- **v3.17** — Удалён whisper-ane (WhisperKit / Argmax). Несмотря на исправления v3.15 (правильное имя модели) и v3.16 (убран `.vad` chunking), `await pipe.transcribe(...)` всё равно зависал на наших клипах: процесс в sleep state, 0% CPU, никогда не возвращается. После 3 итераций фиксов решили выкинуть движок полностью. Удалено: `Sources/Murmur/WhisperKitEngine.swift`, SPM-зависимость `argmaxinc/WhisperKit` из `Package.swift`, case `.whisperKitTurbo` из `SpeechModelOption`, case `.whisperKit` из `EngineKind`, ссылки в README. Существующие пользователи с `activeModel = "whisper-ane"` в UserDefaults автоматом откатываются на дефолт (`turbo`) — `SpeechModelOption(rawValue:)` вернёт nil, init() оставит default. Кэш модели whisper-ane на диске (`~/Documents/huggingface/models/argmaxinc/whisperkit-coreml/...` или подобное место) **не удаляется автоматически** — engine'а больше нет, наш TTL-cleanup до него не доходит; пользователь может удалить вручную.
- **v3.15** — Fix: whisper-ane не работала с v3.6 — лог показывал `modelsUnavailable("...*openai*openai_whisper-large-v3-turbo/*")`, текст никогда не транскрибировался. Корень: `WhisperKit.swift:255-269` сначала ищет `*<variant>/*`, потом если найдено несколько — переходит на `*openai*<variant>/*` для disambiguation. Я передавал `"openai_whisper-large-v3-turbo"` как имя модели, в результате pattern становился `*openai*openai_whisper-large-v3-turbo/*` (двойной префикс) и не находил ни одну папку в HF репо `argmaxinc/whisperkit-coreml`. Правильное имя — без `openai_` префикса. Поменял на `"large-v3-v20240930_626MB"` — рекомендуемый Argmax default из их README, full whisper-large-v3 (не turbo), оптимизированный для ANE до 626 MB. Заодно меняет качество: было обещано turbo, фактически вообще не работало; теперь full v3, который выше по качеству чем turbo. menuLabel обновлён `whisper-large-v3-turbo (ANE)` → `whisper-large-v3 (ANE)`.
- **v3.14** — Слайки UI: убраны два из пяти столбцов в Model menu (short-name дублирует имя модели; compute pool — техническая деталь не для конечного пользователя). Новый шаблон: `<model+params>  ·  <size>  ·  <languages>  [·  <tag>]`. Два whisper-large-v3-turbo (через whisper.cpp+Metal и через WhisperKit+ANE) различаются inline-меткой `(ANE)`. Плюс RU/EN локализация UI по `Locale.preferredLanguages.first`. Новый `Sources/Murmur/Localization.swift` с inline-функцией `t(_ en:, ru:) -> String`. Локализовано: menuLabel всех 6 моделей, status строки в .loading/.idle/.recording/.transcribing/.paused, menu titles (Model→Модель, Microphone→Микрофон), buttons (Pause/Resume/Quit), phase strings во всех 4 движках (Downloading/Compiling/Listing/etc), ETA units (h/m/s → ч/м/с с префиксом "ост."). `ByteCountFormatter` уже использует системную локаль — "1.5 GB"/"1,5 ГБ" автоматом. Текущий язык кешируется при первом доступе к `Localization.isRussian` — system language не меняется в течение жизни процесса.
- **v3.13** — `SpeechModelOption.menuLabel` приведён к единому шаблону `<short>  ·  <model+params>  ·  <compute>  ·  <size>  ·  <languages>  [·  <tag>]`. Старое меню было визуально хаотичным — поля шли в разном порядке, термины разные («multilingual», «52 langs», «Russian» в разных позициях). Теперь все шесть пунктов имеют идентичную структуру колонок (хоть и без табличной сетки — SwiftUI Menu не умеет колонки). Tags `recommended` / `experimental` в конце через тот же `·` separator. Косметика, поведение не меняется.
- **v3.12** — `TextPaster.paste` больше не восстанавливает предыдущий буфер обмена через 0.5s. Текст остаётся на clipboard насовсем (до следующей диктовки). Use case: пользователь забывает кликнуть в input field, симулированный Cmd+V уходит в пустоту, но текст теперь всё равно ждёт в буфере — можно ручным Cmd+V вставить куда нужно. Trade-off: «вежливая» restore-логика убрана, прежний буфер не сохраняется. Принято осознанно — потерять надиктованный абзац хуже чем потерять то что было в буфере раньше.
- **v3.11** — После A/B тестирования всех шести движков на реальной русской диктовке whisper-turbo (whisper.cpp + Metal) показал лучшее качество транскрипции. Default для новых установок изменён `.parakeetV3` → `.whisperTurbo`. Порядок в `SpeechModelOption.allCases` переставлен — turbo первым, parakeet вторым, дальше whisper-ane, large, voxtral, qwen3. `menuLabel`: "(recommended)" перенесён с parakeet на turbo. Существующие пользователи сохраняют свой выбор через UserDefaults restore в `AppState.init()` — только свежие установки получают новый дефолт.
- **v3.10** — Fix: Voxtral / Qwen3 (MLX) крэшили приложение при загрузке моделей. Причина: `swift build` из CLI **не запускает Metal-компилятор**, поэтому `mlx-swift` не генерировал `default.metallib` и MLX runtime падал с `Failed to load the default metallib`. Решение: `build-app.sh` переключён на `xcodebuild build -scheme Murmur -configuration Release ARCHS=arm64`, который правильно компилирует Metal через установленный Metal Toolchain (отдельный Xcode component — нужен `xcodebuild -downloadComponent MetalToolchain` один раз). Скрипт копирует все `*.bundle` из `Build/Products/Release` в `Murmur.app/Contents/Resources/`, MLX runtime находит `default.metallib` через `NS::Bundle::allBundles()`. `ARCHS=arm64` обязательно — иначе xcodebuild пытается universal binary и линковка падает на whisper.cpp `lib/*.a` (arm64-only).
- **v3.9** — Авто-удаление неиспользуемых моделей через 3 дня. Новый протокольный метод `SpeechEngine.cachedModelPaths(name:) async -> [URL]` (default impl `[]` — engines без известного кэш-пути пропускаются). Override'нут в WhisperEngine (`~/Library/Application Support/Murmur/models/ggml-*.bin`), ParakeetEngine (`~/Library/Application Support/FluidAudio/Models/parakeet-tdt-0.6b-v3`), MLXAudioEngine (два пути: `~/.cache/huggingface/hub/mlx-audio/<repo_underscores>` + `~/.cache/huggingface/hub/models--<org>--<name>`). WhisperKit оставлен с default — внутренний путь у Argmax не публичный, авто-чистка пропускает. AppState: `lastUsed: [String: Date]` в UserDefaults (ключ `modelLastUsed`), `touchLastUsed(_:)` на каждом успешном transcribe, `pruneStaleModels()` в начале `setup()` до `loadModel`. TTL = 3 × 24 × 60 × 60 секунд. First-run amnesty: при первом запуске v3.9 записывается `modelCacheAmnestyAt = Date()` и удалений не происходит — даёт 3-дневное окно legacy-установкам. Если активная модель сама протухла — она удалится и сразу качается заново через стандартный progress UI. После prune timestamp обнуляется чтобы новый TTL отсчитывался с момента следующего скачивания.
- **v3.8** — Два новых движка через `mlx-audio-swift` (Blaizzy, MIT): Voxtral Mini 4B Realtime (Mistral AI, 4-bit quantization, ~2 GB) и Qwen3-ASR 1.7B (Alibaba, bf16, ~3.4 GB, **52 языка включая русский и украинский**). Оба через универсальный `MLXAudioEngine` actor с enum `Variant` внутри — модель грузится через статический `fromPretrained(repoID)` каждого конкретного класса (`VoxtralRealtimeModel`, `Qwen3ASRModel`), API общий через `STTGenerationModel` protocol. Compute pool: **MLX/Metal GPU** (не ANE) — отличается от Parakeet и WhisperKit. Это экспериментальные модели для тестов качества vs Parakeet. Sendable workaround через `@preconcurrency import MLX/MLXAudioSTT/MLXAudioCore` (Module-derived классы pre-date Swift 6 strict). `EngineKind.mlxAudio` — четвёртый tag; `SpeechModelOption.voxtralMini` и `.qwen3Asr` — два новых case (raw values "voxtral", "qwen3"). Без публичного download progress API — UI fallback "Downloading & loading". Bundle вырос до ~30 MB (MLX runtime + swift-transformers + swift-nio transitive deps). Меню теперь 6 моделей: parakeet, whisper-ane, voxtral, qwen3, turbo, large.
- **v3.7** — Tokenizer-artifact sanitizer перед TextPaster.paste. Parakeet TDT v3 ставит `<unk>` вокруг типографских символов которых нет в его SentencePiece vocab (русские «ёлочки», curly quotes), результат был типа `<unk>Написать в техподдержку<unk>` вместо `"Написать в техподдержку"`. `AppState.sanitizeTranscript(_:)` парные `<unk>...<unk>` в коротком окне (≤120 символов) превращает в double-quote pair, одиночные стрипает, плюс убирает `<pad>/<s>/</s>/<bos>/<eos>` и чинит double-space + space-before-punct. Whisper-family движков не касается (BPE токенизатор покрывает Unicode), но безопасно для всех.
- **v3.6** — WhisperKit (Argmax) добавлен как четвёртый движок — whisper-large-v3-turbo, скомпилированный в Core ML, бегает на Apple Neural Engine. Закрывает gap «whisper качество + ANE скорость + рабочее auto-language-detection». ~626 MB, ~42× RTF, ANE-резидент. Working language auto-detection без хардкода `language="ru"` (whisper.cpp 1.8.4 detect_language bug это **не** WhisperKit). Архитектура: новый `WhisperKitEngine: SpeechEngine` actor (через `@preconcurrency import WhisperKit` чтобы обойти Swift 6 strict concurrency на не-Sendable `WhisperKit` open class). `EngineKind.whisperKit` третий tag. `SpeechModelOption.whisperKitTurbo` ("whisper-ane") новый case. SPM dependency `argmaxinc/WhisperKit` from 0.9.0. Без явного download progress API — UI показывает "Downloading & compiling" без bytes/MB-s. ModelComputeOptions: melCompute=cpuAndGPU, audioEncoderCompute=ANE, textDecoderCompute=ANE, prefillCompute=cpuOnly (ANE-first layout).
- **v3.5** — Parakeet promoted to default model for new installs. `AppState.activeModel` now initializes to `.parakeetV3`; `SpeechModelOption` declaration order rearranged so Parakeet shows first in the Model menu with a "(recommended)" hint in its label. Existing users keep whatever they previously picked — `UserDefaults["activeModel"]` is read first and only the absence of a saved value triggers the new default. README rewritten to lead with Parakeet (multilingual, ANE-resident, ~66 MB) and to frame whisper turbo/large as opt-in alternatives for Russian-only workflows.
- **v3.4** — Parakeet TDT v3 as a third speech engine, alongside whisper turbo/large. FluidAudio SPM dependency (0.14.5+) wraps the Core ML port — runs on Apple Neural Engine (ANE), ~66 MB working memory vs whisper's 1.5–3 GB, ~110× RTF on M-series, leaves Metal GPU free. Critically: 25 European languages + JP + ZH with **automatic language detection** built in, sidestepping the whisper.cpp 1.8.4 `detect_language` bug that forced Murmur to hardcode `language="ru"`. Architecture: new `SpeechEngine` protocol in `SpeechEngine.swift`; `TranscriptionEngine.swift` renamed to `WhisperEngine.swift`; new `ParakeetEngine.swift`. `AppState.engine` is now `(any SpeechEngine)?` and `currentEngineKind` tracks which actor type is live so cross-engine switches (whisper↔parakeet) destroy the old one. `SpeechModelOption` enum (turbo / large / parakeet) drives the Model menu; UserDefaults raw values match old strings ("turbo", "large") so v3.3 settings migrate cleanly. `DownloadProgress` extended with `fraction` + `phase` (FluidAudio emits fraction + phase enum, no byte counts — UI falls back to "Downloading 2/5  40%" instead of MB/s+ETA for Parakeet).

## Debug

```bash
# Watch live log
tail -f ~/.murmur-debug.log

# Check model exists
ls -la ~/Library/Application\ Support/Murmur/models/

# Kill running instance
pkill -f Murmur.app/Contents/MacOS/Murmur

# Launch with stdout visible
/path/to/Murmur.app/Contents/MacOS/Murmur
```

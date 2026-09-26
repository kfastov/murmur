# Config-change harness

Run it with the Bluetooth headset as default input and output and nothing else holding its mic: `./harness default 5`. That exercises the real A2DP to HFP switch. `RESULT` should report about 100% of the expected samples. The AVAudioEngine recorder got 0%.

Runs the real `Sources/Murmur/AudioRecorder.swift` outside the app and counts the 16 kHz samples it delivers. With `flip`, it changes the default output device's sample rate one second into the take. That is the same format change a Bluetooth headset causes when it switches from A2DP to HFP.

Set the default output to a device whose rate can change, such as the MacBook speakers, before running. The flip toggles the rate between 44.1 and 48 kHz; run it twice to return to the original rate.

```bash
cd Benchmarks/config-change
xcrun swiftc -O -swift-version 5 -o harness main.swift ../../Sources/Murmur/AudioRecorder.swift \
  -framework AVFoundation -framework CoreAudio -framework AudioToolbox -framework Accelerate
./harness default 4 flip   # system default input
./harness builtin 4 flip   # rebind to device 80 (built-in mic on this Mac)
```

`AFTER-FLIP samples` should be close to 48000 for a 4 s run. Before the fix it was about 1600: the recorder went silent after the switch.

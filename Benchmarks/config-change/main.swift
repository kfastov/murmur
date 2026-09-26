import Foundation
import CoreAudio
import AVFoundation

func mlog(_ msg: String) {
    let t = String(format: "%.3f", Date().timeIntervalSince(t0))
    print("[\(t)] \(msg)"); fflush(stdout)
}
let t0 = Date()

func prop<T>(_ id: AudioObjectID, _ sel: AudioObjectPropertySelector, _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal, _ def: T) -> T {
    var a = AudioObjectPropertyAddress(mSelector: sel, mScope: scope, mElement: kAudioObjectPropertyElementMain)
    var v = def; var s = UInt32(MemoryLayout<T>.size)
    AudioObjectGetPropertyData(id, &a, 0, nil, &s, &v); return v
}
func state(_ label: String) {
    let out: AudioDeviceID = prop(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDefaultOutputDevice, kAudioObjectPropertyScopeGlobal, AudioDeviceID(0))
    let inp: AudioDeviceID = prop(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDefaultInputDevice, kAudioObjectPropertyScopeGlobal, AudioDeviceID(0))
    let outSR: Float64 = prop(out, kAudioDevicePropertyNominalSampleRate, kAudioObjectPropertyScopeGlobal, Float64(0))
    let inRun: UInt32 = prop(inp, kAudioDevicePropertyDeviceIsRunningSomewhere, kAudioObjectPropertyScopeGlobal, UInt32(0))
    mlog("\(label): default out=\(out) @ \(outSR) Hz, default in=\(inp) runningSomewhere=\(inRun)")
}

let args = CommandLine.arguments
let mode = args.count > 1 ? args[1] : "default"       // "default" | "builtin"
let seconds = args.count > 2 ? Double(args[2])! : 4.0
let defaultIn: AudioDeviceID = prop(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDefaultInputDevice, kAudioObjectPropertyScopeGlobal, AudioDeviceID(0))
let dev: AudioDeviceID = mode == "builtin" ? 80 : defaultIn
let flip = args.contains("flip")
var samplesAtFlip = 0

state("before")
let lock = NSLock()
var total = 0; var firstAt: Double? = nil; var lastAt = 0.0; var peak: Float = 0
let rec = AudioRecorder()
DispatchQueue.main.async {
    rec.start(deviceID: dev, onLevels: { _ in }, onAudio: { block in
        let now = Date().timeIntervalSince(t0)
        lock.lock(); total += block.count; if firstAt == nil { firstAt = now }; lastAt = now
        for s in block { peak = max(peak, abs(s)) }; lock.unlock()
    })
    if flip {
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
            let flipDev: AudioDeviceID = prop(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDefaultOutputDevice, kAudioObjectPropertyScopeGlobal, AudioDeviceID(0))
            let cur: Float64 = prop(flipDev, kAudioDevicePropertyNominalSampleRate, kAudioObjectPropertyScopeGlobal, Float64(0))
            var target: Float64 = cur == 48000 ? 44100 : 48000
            var a = AudioObjectPropertyAddress(mSelector: kAudioDevicePropertyNominalSampleRate, mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
            let st = AudioObjectSetPropertyData(flipDev, &a, 0, nil, UInt32(MemoryLayout<Float64>.size), &target)
            lock.lock(); let before = total; lock.unlock()
            samplesAtFlip = before
            mlog("FLIP output device \(flipDev) \(cur) -> \(target) Hz, status=\(st), samples so far=\(before)")
        }
    }
    DispatchQueue.main.asyncAfter(deadline: .now() + seconds) {
        state("during")
        rec.stop()
        lock.lock()
        let expected = Int(seconds * 16000)
        mlog(String(format: "RESULT mode=%@ samples=%d of ~%d (%.0f%%), first=%.2fs last=%.2fs peak=%.4f",
                    mode, total, expected, 100.0 * Double(total) / Double(expected), firstAt ?? -1, lastAt, peak))
        if flip { mlog("AFTER-FLIP samples=\(total - samplesAtFlip) (expected ~\(Int((seconds - 1.0) * 16000)))") }
        lock.unlock()
        exit(0)
    }
}
RunLoop.main.run()

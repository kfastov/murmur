import Accelerate
import AudioToolbox
import CoreAudio
import Foundation

/// Captures the microphone and emits 16 kHz mono blocks as they arrive.
///
/// It keeps nothing: no take-sized buffer, no post-processing pass. Whoever
/// consumes `onAudio` owns the recording. That's what lets transcription run
/// alongside the recording instead of after it — see `WhisperEngine`'s session.
///
/// Built on AudioQueue, not AVAudioEngine. Opening a Bluetooth headset's mic
/// flips it from A2DP (44.1/48 kHz) to HFP (16 kHz) about a second after
/// capture starts. AVAudioEngine taps its input through an aggregate that
/// follows the default OUTPUT device, so that flip changes the tap's format
/// under it: the engine stops or re-points itself and the tap never fires
/// again — the take comes back empty. Rebuilding the engine on
/// `AVAudioEngineConfigurationChange` doesn't help: every rebuild re-opens the
/// headset and triggers the next change. AudioQueue asks for 16 kHz mono
/// float up front, converts internally, and keeps delivering through the flip.
///
/// Unchecked because the compiler can't see the discipline: the callbacks are
/// guarded by `lock`, `queue` is only touched from the main actor, and the
/// AudioQueue delivers buffers on one internal thread, in order.
class AudioRecorder: @unchecked Sendable {
    private var queue: AudioQueueRef?
    private let sampleRate: Double = 16000
    /// 100 ms per buffer — the same cadence the transcription session expects.
    private let framesPerBuffer: UInt32 = 1600
    private let bufferCount = 3
    private var levelsCallback: (([Float]) -> Void)?
    private var audioCallback: (@Sendable ([Float]) -> Void)?
    private let numBars = 11
    private let barWeights: [Float] = [0.3, 0.5, 0.7, 0.85, 0.95, 1.0, 0.95, 0.85, 0.7, 0.5, 0.3]
    private let lock = NSLock()

    func start(
        deviceID: AudioDeviceID? = nil,
        onLevels: @escaping ([Float]) -> Void,
        onAudio: @escaping @Sendable ([Float]) -> Void
    ) {
        lock.lock()
        levelsCallback = onLevels
        audioCallback = onAudio
        lock.unlock()

        var format = AudioStreamBasicDescription(
            mSampleRate: sampleRate,
            mFormatID: kAudioFormatLinearPCM,
            mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
            mBytesPerPacket: 4,
            mFramesPerPacket: 1,
            mBytesPerFrame: 4,
            mChannelsPerFrame: 1,
            mBitsPerChannel: 32,
            mReserved: 0
        )

        var newQueue: AudioQueueRef?
        let status = AudioQueueNewInput(
            &format,
            Self.inputCallback,
            Unmanaged.passUnretained(self).toOpaque(),
            nil,   // deliver on the queue's own thread
            nil,
            0,
            &newQueue
        )
        guard status == noErr, let queue = newQueue else {
            mlog("AudioRecorder: AudioQueueNewInput failed (\(status))")
            return
        }
        self.queue = queue

        // Point THIS queue at the requested device, without touching the
        // system-wide default input (a sandbox violation — modifying a system
        // preference is not covered by the audio-input entitlement). Skip it
        // when the pick already is the system default. On any failure we log
        // and fall through to the system default — never crash.
        let systemDefault = Self.currentDefaultInputDevice()
        if let deviceID, deviceID != systemDefault {
            if let uid = Self.uid(of: deviceID) {
                var cfUID = uid as CFString
                let st = AudioQueueSetProperty(
                    queue,
                    kAudioQueueProperty_CurrentDevice,
                    &cfUID,
                    UInt32(MemoryLayout<CFString>.size)
                )
                if st == noErr {
                    mlog("AudioRecorder: bound to device \(deviceID) \(uid) (system default is \(systemDefault))")
                } else {
                    mlog("AudioRecorder: set CurrentDevice failed (\(st)) — using system default")
                }
            } else {
                mlog("AudioRecorder: no UID for device \(deviceID) — using system default")
            }
        } else {
            mlog("AudioRecorder: requested device \(deviceID.map(String.init) ?? "nil") is the system default (\(systemDefault)) — no rebind needed")
        }

        let bytes = framesPerBuffer * format.mBytesPerFrame
        for _ in 0..<bufferCount {
            var buffer: AudioQueueBufferRef?
            if AudioQueueAllocateBuffer(queue, bytes, &buffer) == noErr, let buffer {
                AudioQueueEnqueueBuffer(queue, buffer, 0, nil)
            }
        }

        let st = AudioQueueStart(queue, nil)
        if st == noErr {
            mlog("AudioRecorder: started (AudioQueue, 16000 Hz mono)")
        } else {
            mlog("AudioRecorder start failed: \(st)")
        }
    }

    /// Stops capture. Every block captured before this returns has already
    /// been handed to `onAudio`: an immediate AudioQueueStop is synchronous,
    /// so no callback runs after it — the consumer can treat the next call as
    /// "the take is complete".
    func stop() {
        if let queue {
            AudioQueueStop(queue, true)
            AudioQueueDispose(queue, true)
        }
        queue = nil

        lock.lock()
        levelsCallback = nil
        audioCallback = nil
        lock.unlock()

        mlog("AudioRecorder: stopped")
    }

    private static let inputCallback: AudioQueueInputCallback = { user, queue, buffer, _, _, _ in
        guard let user else { return }
        let recorder = Unmanaged<AudioRecorder>.fromOpaque(user).takeUnretainedValue()
        recorder.deliver(buffer)
        // Re-enqueue unless the queue is being torn down; enqueueing into a
        // stopped queue just fails, which is harmless.
        AudioQueueEnqueueBuffer(queue, buffer, 0, nil)
    }

    private func deliver(_ buffer: AudioQueueBufferRef) {
        let count = Int(buffer.pointee.mAudioDataByteSize) / MemoryLayout<Float>.size
        guard count > 0 else { return }
        let data = buffer.pointee.mAudioData.assumingMemoryBound(to: Float.self)

        // Copy immediately — the buffer goes straight back to the queue.
        let samples = Array(UnsafeBufferPointer(start: data, count: count))

        var rms: Float = 0
        vDSP_rmsqv(data, 1, &rms, vDSP_Length(count))
        let level = min(1.0, rms * 15.0)
        var bars = [Float](repeating: 0, count: numBars)
        for i in 0..<numBars {
            bars[i] = level * barWeights[i]
        }

        lock.lock()
        let levels = levelsCallback
        let audio = audioCallback
        lock.unlock()

        levels?(bars)
        audio?(samples)
    }

    // MARK: - Devices (read-only)

    /// Read the system default input device. This is a GET — sandbox-safe and
    /// allowed in the Mac App Store (only SETTING the default is a violation).
    static func currentDefaultInputDevice() -> AudioDeviceID {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var deviceID: AudioDeviceID = 0
        var size = UInt32(MemoryLayout<AudioDeviceID>.size)
        AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &deviceID)
        return deviceID
    }

    /// AudioQueue selects devices by UID, AppState hands us an ID.
    private static func uid(of deviceID: AudioDeviceID) -> String? {
        var addr = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyDeviceUID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var uid: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        let st = AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, &uid)
        guard st == noErr, let uid else { return nil }
        return uid.takeRetainedValue() as String
    }
}

// otto-audiotap — capture macOS system audio output via a Core Audio process tap
// and stream it to stdout as raw 16 kHz mono little-endian int16 PCM.
//
// Requires macOS 14.4+ (Core Audio process taps). No third-party driver and no
// rerouting: the tap reads the system output mix in software, so the user keeps
// hearing audio normally through whatever device (speakers, wired/Bluetooth
// headphones) is selected.
//
// Usage:
//   otto-audiotap [--exclude-pid <pid>] [--sample-rate <hz>]
//
// Protocol:
//   stdout — raw PCM frames (int16 mono @ target rate). Nothing else is written
//            to stdout; all diagnostics go to stderr.
//   Exit   — on SIGINT/SIGTERM the tap + aggregate device are destroyed cleanly.

import AVFoundation
import AudioToolbox
import CoreAudio
import Darwin
import Foundation

// ---------------------------------------------------------------------------
// Logging (stderr only — stdout is reserved for PCM)
// ---------------------------------------------------------------------------

func logErr(_ msg: String) {
    FileHandle.standardError.write(Data(("[audiotap] " + msg + "\n").utf8))
}

func fail(_ msg: String) -> Never {
    logErr("FATAL: " + msg)
    exit(1)
}

/// Write raw bytes to stdout using the low-level write(2) syscall.
///
/// ``FileHandle.write(_:)`` raises an uncaught ``NSFileHandleOperationException``
/// on EPIPE (broken pipe) which crashes the process — ``SIG_IGN`` on SIGPIPE
/// does NOT prevent it, because Foundation turns the failed write into an
/// Objective-C exception, not a signal. Using ``write(2)`` directly lets us
/// detect a closed reader (the backend went away / stopped capturing) and exit
/// cleanly instead of dying with a spurious crash report.
@inline(__always)
func writeAllToStdout(_ base: UnsafeRawPointer, _ count: Int) {
    var offset = 0
    while offset < count {
        let n = write(STDOUT_FILENO, base + offset, count - offset)
        if n > 0 {
            offset += n
            continue
        }
        if n == -1 {
            if errno == EINTR { continue }
            // EPIPE (reader gone) or any other write error: the consumer is no
            // longer there, so there's nothing useful left to do. Exit cleanly.
            exit(0)
        }
        // n == 0 shouldn't happen for a pipe; treat it as a closed reader.
        exit(0)
    }
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

var excludePIDs: [pid_t] = []
var targetSampleRate: Double = 16_000

do {
    let args = Array(CommandLine.arguments.dropFirst())
    var i = 0
    while i < args.count {
        switch args[i] {
        case "--exclude-pid":
            i += 1
            if i < args.count, let p = pid_t(args[i]) { excludePIDs.append(p) }
        case "--sample-rate":
            i += 1
            if i < args.count, let r = Double(args[i]) { targetSampleRate = r }
        default:
            logErr("ignoring unknown arg: \(args[i])")
        }
        i += 1
    }
}

// ---------------------------------------------------------------------------
// Core Audio property helpers
// ---------------------------------------------------------------------------

func defaultOutputDeviceID() -> AudioObjectID {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyDefaultOutputDevice,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var deviceID = AudioObjectID(0)
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    let status = AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject), &addr, 0, nil, &size, &deviceID
    )
    if status != noErr { fail("could not read default output device (\(status))") }
    return deviceID
}

func deviceUID(_ deviceID: AudioObjectID) -> String {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioDevicePropertyDeviceUID,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var uid: CFString = "" as CFString
    var size = UInt32(MemoryLayout<CFString>.size)
    let status = withUnsafeMutablePointer(to: &uid) { ptr -> OSStatus in
        AudioObjectGetPropertyData(deviceID, &addr, 0, nil, &size, ptr)
    }
    if status != noErr { fail("could not read output device UID (\(status))") }
    return uid as String
}

/// Translate a Unix PID into a Core Audio process object ID (needed to exclude
/// specific processes from the tap). Returns nil if the PID isn't audible.
func processObject(forPID pid: pid_t) -> AudioObjectID? {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var inPID = pid
    var obj = AudioObjectID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    let status = AudioObjectGetPropertyData(
        AudioObjectID(kAudioObjectSystemObject),
        &addr,
        UInt32(MemoryLayout<pid_t>.size),
        &inPID,
        &size,
        &obj
    )
    if status != noErr || obj == kAudioObjectUnknown { return nil }
    return obj
}

@available(macOS 14.4, *)
func makeTap() -> AudioObjectID {
    let excluded: [AudioObjectID] = excludePIDs.compactMap { processObject(forPID: $0) }
    // Mono global tap of every process except the excluded ones (e.g. Otto's
    // own output, to avoid transcribing our own TTS in the future).
    let desc = CATapDescription(monoGlobalTapButExcludeProcesses: excluded)
    desc.isPrivate = true
    desc.muteBehavior = CATapMuteBehavior.unmuted // do not silence the speakers

    var tapID = AudioObjectID(kAudioObjectUnknown)
    let status = AudioHardwareCreateProcessTap(desc, &tapID)
    if status != noErr || tapID == kAudioObjectUnknown {
        fail("AudioHardwareCreateProcessTap failed (\(status)) — check System Settings > Privacy & Security > System Audio Recording")
    }
    return tapID
}

@available(macOS 14.4, *)
func tapStreamFormat(_ tapID: AudioObjectID) -> AudioStreamBasicDescription {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioTapPropertyFormat,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var asbd = AudioStreamBasicDescription()
    var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
    let status = AudioObjectGetPropertyData(tapID, &addr, 0, nil, &size, &asbd)
    if status != noErr { fail("could not read tap format (\(status))") }
    return asbd
}

@available(macOS 14.4, *)
func tapUID(_ tapID: AudioObjectID) -> String {
    var addr = AudioObjectPropertyAddress(
        mSelector: kAudioTapPropertyUID,
        mScope: kAudioObjectPropertyScopeGlobal,
        mElement: kAudioObjectPropertyElementMain
    )
    var uid: CFString = "" as CFString
    var size = UInt32(MemoryLayout<CFString>.size)
    let status = withUnsafeMutablePointer(to: &uid) { ptr -> OSStatus in
        AudioObjectGetPropertyData(tapID, &addr, 0, nil, &size, ptr)
    }
    if status != noErr { fail("could not read tap UID (\(status))") }
    return uid as String
}

@available(macOS 14.4, *)
func makeAggregate(tapID: AudioObjectID) -> AudioObjectID {
    let outputUID = deviceUID(defaultOutputDeviceID())
    let subTapUID = tapUID(tapID)

    let aggUID = "com.research.otto.audiotap." + UUID().uuidString
    let desc: [String: Any] = [
        kAudioAggregateDeviceNameKey as String: "Otto System Audio Tap",
        kAudioAggregateDeviceUIDKey as String: aggUID,
        kAudioAggregateDeviceMainSubDeviceKey as String: outputUID,
        kAudioAggregateDeviceIsPrivateKey as String: true,
        kAudioAggregateDeviceIsStackedKey as String: false,
        kAudioAggregateDeviceTapAutoStartKey as String: true,
        kAudioAggregateDeviceSubDeviceListKey as String: [
            [kAudioSubDeviceUIDKey as String: outputUID],
        ],
        kAudioAggregateDeviceTapListKey as String: [
            [
                kAudioSubTapUIDKey as String: subTapUID,
                kAudioSubTapDriftCompensationKey as String: true,
            ],
        ],
    ]

    var aggID = AudioObjectID(kAudioObjectUnknown)
    let status = AudioHardwareCreateAggregateDevice(desc as CFDictionary, &aggID)
    if status != noErr || aggID == kAudioObjectUnknown {
        fail("AudioHardwareCreateAggregateDevice failed (\(status))")
    }
    return aggID
}

// ---------------------------------------------------------------------------
// Capture object (owns the tap + aggregate + IOProc)
//
// The Core Audio setup MUST NOT run on the main thread: several of these calls
// (notably AudioDeviceCreateIOProcIDWithBlock on a tap-backed aggregate) block
// waiting for a reply from coreaudiod that is serviced on the main run loop.
// Doing setup on the main thread deadlocks. So we set up on a background queue
// and keep the main run loop free.
// ---------------------------------------------------------------------------

@available(macOS 14.4, *)
final class Capture {
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggID = AudioObjectID(kAudioObjectUnknown)
    private var ioProcID: AudioDeviceIOProcID?
    private let ioQueue = DispatchQueue(label: "com.research.otto.audiotap.io")
    private var torndown = false
    private let lock = NSLock()

    func start() {
        tapID = makeTap()
        aggID = makeAggregate(tapID: tapID)

        var inASBD = tapStreamFormat(tapID)
        guard let inFormat = withUnsafePointer(to: &inASBD, { AVAudioFormat(streamDescription: $0) }) else {
            fail("could not build input AVAudioFormat from tap")
        }
        guard let outFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: true
        ) else {
            fail("could not build output AVAudioFormat")
        }
        guard let converter = AVAudioConverter(from: inFormat, to: outFormat) else {
            fail("could not create AVAudioConverter \(inFormat) -> \(outFormat)")
        }

        logErr("capturing: in=\(inFormat.sampleRate)Hz/\(inFormat.channelCount)ch -> out=\(targetSampleRate)Hz/mono/int16")

        let ioBlock: AudioDeviceIOBlock = { _, inInputData, _, _, _ in
            guard let inBuffer = AVAudioPCMBuffer(pcmFormat: inFormat, bufferListNoCopy: inInputData, deallocator: nil) else {
                return
            }
            let inFrames = inBuffer.frameLength
            if inFrames == 0 { return }

            let ratio = targetSampleRate / inFormat.sampleRate
            let outCapacity = AVAudioFrameCount(Double(inFrames) * ratio) + 32
            guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: outCapacity) else {
                return
            }

            var fed = false
            var convError: NSError?
            let status = converter.convert(to: outBuffer, error: &convError) { _, outStatus in
                if fed {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                fed = true
                outStatus.pointee = .haveData
                return inBuffer
            }
            if status == .error {
                if let e = convError { logErr("convert error: \(e.localizedDescription)") }
                return
            }

            let outFrames = Int(outBuffer.frameLength)
            if outFrames == 0 { return }
            guard let channel = outBuffer.int16ChannelData else { return }
            let byteCount = outFrames * MemoryLayout<Int16>.size
            writeAllToStdout(UnsafeRawPointer(channel[0]), byteCount)
        }

        let createStatus = AudioDeviceCreateIOProcIDWithBlock(&ioProcID, aggID, ioQueue, ioBlock)
        if createStatus != noErr || ioProcID == nil {
            teardown()
            fail("AudioDeviceCreateIOProcIDWithBlock failed (\(createStatus))")
        }

        let startStatus = AudioDeviceStart(aggID, ioProcID)
        if startStatus != noErr {
            teardown()
            fail("AudioDeviceStart failed (\(startStatus)) — check System Settings > Privacy & Security > System Audio Recording")
        }

        logErr("started")
    }

    func teardown() {
        lock.lock()
        defer { lock.unlock() }
        if torndown { return }
        torndown = true
        if let proc = ioProcID {
            AudioDeviceStop(aggID, proc)
            AudioDeviceDestroyIOProcID(aggID, proc)
            ioProcID = nil
        }
        if aggID != kAudioObjectUnknown {
            AudioHardwareDestroyAggregateDevice(aggID)
        }
        if tapID != kAudioObjectUnknown {
            AudioHardwareDestroyProcessTap(tapID)
        }
    }
}

@available(macOS 14.4, *)
func run() -> Never {
    let capture = Capture()

    // Set up + start capture off the main thread so the main run loop stays
    // free to service coreaudiod replies (avoids a setup deadlock).
    DispatchQueue.global(qos: .userInitiated).async {
        capture.start()
    }

    var signalSources: [DispatchSourceSignal] = []
    for sig in [SIGINT, SIGTERM] {
        signal(sig, SIG_IGN)
        let src = DispatchSource.makeSignalSource(signal: sig, queue: .main)
        src.setEventHandler {
            logErr("received signal — tearing down")
            capture.teardown()
            exit(0)
        }
        src.resume()
        signalSources.append(src)
    }
    _ = signalSources

    // Exit cleanly if our stdout reader (the Python backend) goes away.
    signal(SIGPIPE, SIG_IGN)

    RunLoop.main.run()
    fail("run loop exited unexpectedly")
}

if #available(macOS 14.4, *) {
    run()
} else {
    fail("system audio capture requires macOS 14.4 or later")
}

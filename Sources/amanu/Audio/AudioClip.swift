import AVFoundation
import Foundation

/// A few seconds of a recording as a small WAV in memory: what OpenAI takes as
/// a sample of a known speaker's voice.
///
/// 16 kHz mono 16-bit, because the sample is sent base64-encoded inside an
/// ordinary form field with every piece of the meeting. Eight seconds of it
/// is 256 KB; the same at the mix's own 48 kHz float would be six times that,
/// four times over, on every request.
enum AudioClip {
    enum ClipError: Error, CustomStringConvertible {
        case unreadable(URL, Error?)
        case outOfRange(URL)
        case conversion(Error?)

        var description: String {
            switch self {
            case .unreadable(let url, let error):
                return "can't read \(url.lastPathComponent)" + (error.map { ": \($0)" } ?? "")
            case .outOfRange(let url):
                return "the clip lies outside \(url.lastPathComponent)"
            case .conversion(let error):
                return "can't convert the clip" + (error.map { ": \($0)" } ?? "")
            }
        }
    }

    static func wav(
        from source: URL,
        start: TimeInterval,
        length: TimeInterval,
        sampleRate: Double = 16_000
    ) throws -> Data {
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: source)
        } catch {
            throw ClipError.unreadable(source, error)
        }
        let input = file.processingFormat
        let first = AVAudioFramePosition(max(0, start) * input.sampleRate)
        guard first < file.length, length > 0 else { throw ClipError.outOfRange(source) }
        let frames = AVAudioFrameCount(
            min(Double(file.length - first), length * input.sampleRate))
        guard frames > 0,
              let samples = AVAudioPCMBuffer(pcmFormat: input, frameCapacity: frames)
        else { throw ClipError.outOfRange(source) }
        file.framePosition = first
        do {
            try file.read(into: samples, frameCount: frames)
        } catch {
            throw ClipError.unreadable(source, error)
        }

        guard let output = AVAudioFormat(
                commonFormat: .pcmFormatInt16, sampleRate: sampleRate,
                channels: 1, interleaved: true),
              let converter = AVAudioConverter(from: input, to: output),
              let converted = AVAudioPCMBuffer(
                pcmFormat: output,
                frameCapacity: AVAudioFrameCount(
                    Double(samples.frameLength) * sampleRate / input.sampleRate) + 4096)
        else { throw ClipError.conversion(nil) }

        let feed = Feed(samples)
        var failure: NSError?
        let status = converter.convert(to: converted, error: &failure) { _, state in
            feed.next(state)
        }
        guard status != .error, converted.frameLength > 0,
              let channel = converted.int16ChannelData?[0]
        else { throw ClipError.conversion(failure) }

        return encode(
            UnsafeBufferPointer(start: channel, count: Int(converted.frameLength)),
            sampleRate: Int(sampleRate))
    }

    /// Hands the converter its one buffer, then says that was all.
    private final class Feed: @unchecked Sendable {
        private var buffer: AVAudioPCMBuffer?
        init(_ buffer: AVAudioPCMBuffer) { self.buffer = buffer }

        func next(_ state: UnsafeMutablePointer<AVAudioConverterInputStatus>) -> AVAudioBuffer? {
            guard let buffer else {
                state.pointee = .endOfStream
                return nil
            }
            self.buffer = nil
            state.pointee = .haveData
            return buffer
        }
    }

    /// A canonical 44-byte RIFF header and the samples, little-endian.
    static func encode(_ samples: UnsafeBufferPointer<Int16>, sampleRate: Int) -> Data {
        let bytes = samples.count * 2
        var data = Data(capacity: 44 + bytes)
        func append<T: FixedWidthInteger>(_ value: T) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        data.append(contentsOf: Array("RIFF".utf8))
        append(UInt32(36 + bytes))
        data.append(contentsOf: Array("WAVEfmt ".utf8))
        append(UInt32(16))
        append(UInt16(1))  // PCM
        append(UInt16(1))  // mono
        append(UInt32(sampleRate))
        append(UInt32(sampleRate * 2))
        append(UInt16(2))
        append(UInt16(16))
        data.append(contentsOf: Array("data".utf8))
        append(UInt32(bytes))
        for sample in samples { append(sample) }
        return data
    }
}

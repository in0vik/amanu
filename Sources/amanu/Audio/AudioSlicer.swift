import AVFoundation
import Foundation

/// Cuts one audio file into consecutive pieces short and small enough to
/// upload, each cut placed in a pause rather than through a word.
///
/// This exists for OpenAI, which refuses a request over 25 MB and, for the
/// diarizing model, any audio over 1400 seconds. The second limit is the one
/// meetings actually hit: 23 minutes is a short call. Nobody should have to
/// know either, so the engine slices, transcribes each piece, and shifts the
/// timings back onto the session's clock; the transcript comes out looking
/// like one pass.
///
/// The pieces are written frame by frame through `AVAudioFile` rather than by
/// `AVAssetExportSession`, for the reason `AudioMixer` explains at length: the
/// export path makes macOS ask for Photos and Music access, which is an
/// absurd thing for a meeting recorder to do.
enum AudioSlicer {
    struct Slice: Sendable {
        let url: URL
        /// Where this piece starts in the source, in seconds. What gets added
        /// back to every timestamp the engine returns for it.
        let offset: TimeInterval
    }

    enum SliceError: Error, CustomStringConvertible {
        case unreadable(URL, Error?)
        case unwritable(URL, Error)

        var description: String {
            switch self {
            case .unreadable(let url, let error):
                return "can't read \(url.lastPathComponent)"
                    + (error.map { ": \($0)" } ?? "")
            case .unwritable(let url, let error):
                return "can't write \(url.lastPathComponent): \(error)"
            }
        }
    }

    /// The bitrate every piece is written at: the same 64k mono the mix is
    /// written at. Re-encoding lower would shrink the upload and cost accuracy
    /// on exactly the audio that needs it.
    static let encodedBitRate = 64_000

    /// Resolution of the loudness envelope the cuts are chosen from. A pause
    /// between sentences is several of these long; a gap between syllables
    /// is one at most.
    static let bucket: TimeInterval = 0.1

    /// Split `source` into pieces of at most `maxLength` seconds each, written
    /// into `directory`, each cut placed at the quietest moment within
    /// `window` seconds of where an even split would put it. A source already
    /// short enough is not copied — the single slice points at the original
    /// file, so the common case touches no disk.
    static func slice(
        _ source: URL,
        maxLength: TimeInterval,
        into directory: URL,
        window: TimeInterval = 30
    ) async throws -> [Slice] {
        try await Task.detached(priority: .utility) {
            try sliceNow(source, maxLength: maxLength, into: directory, window: window)
        }.value
    }

    /// The longest a piece may be for every piece to fit both of the API's
    /// limits: `durationLimit` seconds of audio, and `requestLimit` bytes.
    ///
    /// The bytes are worked out at whichever bitrate is higher, the source's
    /// or the one the pieces are written at. Working them out at the source's
    /// alone undercounts whenever the mix is encoded below 64k: its pieces are
    /// re-encoded at 64k and come out larger than the estimate. A piece over
    /// 25 MB is answered with "Audio file might be corrupted or unsupported"
    /// — nothing about size — and that is the answer every meeting over an
    /// hour got in October 2026.
    ///
    /// The margin covers the variation a constant-bitrate estimate leaves: a
    /// piece that comes back slightly over a limit costs the whole
    /// transcription, and a piece 15% shorter than it had to be costs nothing.
    static func maxPieceLength(
        bytes: Int64,
        duration: TimeInterval,
        requestLimit: Int64,
        durationLimit: TimeInterval,
        margin: Double = 0.85
    ) -> TimeInterval {
        let encoded = Double(encodedBitRate) / 8
        let source = duration > 0 && bytes > 0 ? Double(bytes) / duration : encoded
        let byteBound = Double(requestLimit) * margin / max(encoded, source)
        // Never below a minute: a limit that small means the numbers are wrong,
        // and hundreds of tiny uploads is not a better failure than one refusal.
        return max(60, min(durationLimit, byteBound))
    }

    /// Where to cut a recording `duration` seconds long so that no piece is
    /// longer than `maxLength`, given its loudness in `bucket`-long steps.
    /// Returns the cut times in order, without 0 and without the end.
    ///
    /// Each cut aims for an even split of what is left, then moves to the
    /// quietest half second within `window` of that aim — never past
    /// `maxLength` from the previous cut. Pieces of about the same length
    /// matter more than they look: the speakers of every later piece are
    /// recognised from voices heard in the first, and a first piece of five
    /// minutes may not have heard everyone yet.
    static func cuts(
        levels: [Float],
        bucket: TimeInterval,
        duration: TimeInterval,
        maxLength: TimeInterval,
        window: TimeInterval
    ) -> [TimeInterval] {
        guard maxLength > 0, duration > maxLength else { return [] }
        // Half a second either side: a pause, not the dip between two
        // syllables, is what a cut should land in.
        let smoothing = max(0, Int((0.25 / bucket).rounded()))
        var cuts: [TimeInterval] = []
        var start: TimeInterval = 0
        while duration - start > maxLength {
            let remaining = duration - start
            let pieces = (remaining / maxLength).rounded(.up)
            let aim = start + remaining / pieces
            let latest = start + maxLength
            let earliest = min(latest, max(start + bucket, aim - window))
            let first = Int((earliest / bucket).rounded(.up))
            let last = Int((min(latest, aim + window) / bucket).rounded(.down))

            var best = min(latest, aim)
            var quietest = Float.greatestFiniteMagnitude
            if first <= last {
                for index in first...last {
                    let low = max(0, index - smoothing)
                    let high = min(levels.count - 1, index + smoothing)
                    guard low <= high else { break }
                    var sum: Float = 0
                    for level in levels[low...high] { sum += level }
                    let mean = sum / Float(high - low + 1)
                    // Ties go to the point nearest the aim, so a stretch of
                    // perfect silence is cut in its middle, not at its edge.
                    let candidate = Double(index) * bucket
                    if mean < quietest
                        || (mean == quietest && abs(candidate - aim) < abs(best - aim)) {
                        quietest = mean
                        best = candidate
                    }
                }
            }
            // Always forward, so the loop ends whatever the levels say.
            let cut = min(latest, max(start + bucket, best))
            cuts.append(cut)
            start = cut
        }
        return cuts
    }

    // MARK: -

    private static func sliceNow(
        _ source: URL,
        maxLength: TimeInterval,
        into directory: URL,
        window: TimeInterval
    ) throws -> [Slice] {
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: source)
        } catch {
            throw SliceError.unreadable(source, error)
        }
        let format = file.processingFormat
        let rate = format.sampleRate
        guard file.length > 0, rate > 0 else { throw SliceError.unreadable(source, nil) }

        let duration = Double(file.length) / rate
        guard duration > maxLength else { return [Slice(url: source, offset: 0)] }

        // A second at a time, like the mixer: small enough that a long meeting
        // never holds more than a few hundred kilobytes of audio in memory.
        let blockFrames = AVAudioFrameCount(rate)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: blockFrames) else {
            throw SliceError.unreadable(source, nil)
        }

        let levels = try envelope(of: file, buffer: buffer, source: source)
        let boundaries = cuts(
            levels: levels, bucket: bucket, duration: duration,
            maxLength: maxLength, window: window
        ).map { AVAudioFramePosition(($0 * rate).rounded()) } + [file.length]

        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        file.framePosition = 0

        var slices: [Slice] = []
        for (index, end) in boundaries.enumerated() {
            let start = file.framePosition
            guard start < end else { continue }
            let url = directory.appendingPathComponent(
                "slice-\(String(format: "%03d", index)).m4a")
            try? FileManager.default.removeItem(at: url)
            let written = try write(
                from: file, until: end, buffer: buffer, to: url, source: source)
            guard written > 0 else {
                try? FileManager.default.removeItem(at: url)
                break
            }
            slices.append(Slice(url: url, offset: Double(start) / rate))
        }
        return slices
    }

    /// RMS per `bucket`, all channels together. One pass over the file before
    /// any piece is written, because where the first cut goes depends on
    /// audio the first piece does not contain.
    private static func envelope(
        of file: AVAudioFile, buffer: AVAudioPCMBuffer, source: URL
    ) throws -> [Float] {
        let channels = Int(file.processingFormat.channelCount)
        let bucketFrames = max(1, Int(file.processingFormat.sampleRate * bucket))
        var levels: [Float] = []
        levels.reserveCapacity(Int(file.length) / bucketFrames + 1)
        var sum: Float = 0
        var count = 0

        file.framePosition = 0
        while file.framePosition < file.length {
            buffer.frameLength = 0
            do {
                try file.read(into: buffer, frameCount: buffer.frameCapacity)
            } catch {
                throw SliceError.unreadable(source, error)
            }
            let frames = Int(buffer.frameLength)
            guard frames > 0, let data = buffer.floatChannelData else { break }
            for frame in 0..<frames {
                for channel in 0..<channels {
                    let sample = data[channel][frame]
                    sum += sample * sample
                }
                count += 1
                if count == bucketFrames {
                    levels.append((sum / Float(count * channels)).squareRoot())
                    sum = 0
                    count = 0
                }
            }
        }
        if count > 0 { levels.append((sum / Float(count * channels)).squareRoot()) }
        return levels
    }

    /// Copy frames from `file`'s current position up to `end` into a new
    /// piece at `url`. The output is closed when this returns, which is the
    /// point of it being a function of its own: an AAC file is only complete
    /// once it is closed.
    private static func write(
        from file: AVAudioFile,
        until end: AVAudioFramePosition,
        buffer: AVAudioPCMBuffer,
        to url: URL,
        source: URL
    ) throws -> AVAudioFramePosition {
        let format = file.processingFormat
        var settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatMPEG4AAC,
            AVSampleRateKey: format.sampleRate,
            AVNumberOfChannelsKey: format.channelCount,
            AVEncoderBitRateKey: encodedBitRate,
        ]
        let output: AVAudioFile
        do {
            output = try AVAudioFile(
                forWriting: url, settings: settings,
                commonFormat: format.commonFormat, interleaved: format.isInterleaved)
        } catch {
            // The encoder refuses 64k below about 24 kHz — an imported 16 kHz
            // recording, say. Its own default there is lower still, which
            // keeps the piece inside the size `maxPieceLength` assumed.
            settings[AVEncoderBitRateKey] = nil
            try? FileManager.default.removeItem(at: url)
            do {
                output = try AVAudioFile(
                    forWriting: url, settings: settings,
                    commonFormat: format.commonFormat, interleaved: format.isInterleaved)
            } catch {
                throw SliceError.unwritable(url, error)
            }
        }

        var written: AVAudioFramePosition = 0
        while file.framePosition < end {
            let wanted = AVAudioFrameCount(min(
                AVAudioFramePosition(buffer.frameCapacity), end - file.framePosition))
            buffer.frameLength = 0
            do {
                try file.read(into: buffer, frameCount: wanted)
            } catch {
                throw SliceError.unreadable(source, error)
            }
            guard buffer.frameLength > 0 else { break }
            do {
                try output.write(from: buffer)
            } catch {
                throw SliceError.unwritable(url, error)
            }
            written += AVAudioFramePosition(buffer.frameLength)
        }
        return written
    }
}

import AVFoundation
import Foundation
import Testing

@testable import amanu

/// Where a long meeting is cut, and what the pieces are sent with — against
/// real audio files and a stub of the API, so that the whole path a meeting
/// over twenty minutes takes runs without a key or a network.
struct SlicedTranscriptionTests {
    /// A tone with two seconds of silence at each of `gaps`.
    private static func speechWithGaps(
        seconds: Double, gaps: [Double], in dir: URL
    ) throws -> URL {
        let url = dir.appendingPathComponent("mixed.caf")
        let rate = 16_000.0
        try TestAudio.write(to: url, seconds: seconds, sampleRate: rate) { _, frame in
            let t = Double(frame) / rate
            if gaps.contains(where: { t >= $0 && t < $0 + 2 }) { return 0 }
            return 0.3 * Float(sin(2 * .pi * 220 * t))
        }
        return url
    }

    private static func folder() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("amanu-sliced-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    @Test("Cuts land in the pause nearest an even split, never past the limit")
    func cutsFindThePause() {
        // 150 seconds of speech at 100 ms, quiet only at 55–57 s and 110–112 s.
        var levels = [Float](repeating: 0.3, count: 1500)
        for index in 550..<570 { levels[index] = 0 }
        for index in 1100..<1120 { levels[index] = 0 }

        let cuts = AudioSlicer.cuts(
            levels: levels, bucket: 0.1, duration: 150, maxLength: 60, window: 30)
        #expect(cuts.count == 2)
        #expect((55...57).contains(cuts[0]))
        #expect((110...112).contains(cuts[1]))

        let bounds = [0] + cuts + [150]
        for (start, end) in zip(bounds, bounds.dropFirst()) {
            #expect(end - start <= 60)
        }
    }

    @Test("Without a pause, pieces still come out even and under the limit")
    func cutsWithoutPause() {
        let levels = [Float](repeating: 0.3, count: 3000)
        let cuts = AudioSlicer.cuts(
            levels: levels, bucket: 0.1, duration: 300, maxLength: 130, window: 30)
        #expect(cuts.count == 2)
        let bounds = [0] + cuts + [300]
        for (start, end) in zip(bounds, bounds.dropFirst()) {
            #expect(end - start <= 130)
            #expect(end - start >= 70)
        }
    }

    @Test("A short recording is not cut at all")
    func noCuts() {
        #expect(AudioSlicer.cuts(
            levels: [0.3, 0.3], bucket: 0.1, duration: 0.2, maxLength: 60, window: 30).isEmpty)
    }

    @Test("A voice sample is a 16 kHz mono WAV of the asked-for length")
    func clip() throws {
        let dir = try Self.folder()
        defer { try? FileManager.default.removeItem(at: dir) }
        let audio = try Self.speechWithGaps(seconds: 20, gaps: [], in: dir)

        let wav = try AudioClip.wav(from: audio, start: 5, length: 3)
        #expect(String(decoding: wav.prefix(4), as: UTF8.self) == "RIFF")
        #expect(String(decoding: wav[8..<12], as: UTF8.self) == "WAVE")
        let samples = (wav.count - 44) / 2
        #expect(abs(samples - 48_000) < 1_000)
    }

    /// The whole path: cut, the first piece alone, the rest told about the
    /// first piece's voices, labels put back on one clock.
    @Test("A long meeting is sent in pieces whose speakers carry over")
    func slicedMeeting() async throws {
        let dir = try Self.folder()
        defer { try? FileManager.default.removeItem(at: dir) }
        let audio = try Self.speechWithGaps(seconds: 150, gaps: [55, 110], in: dir)

        let stub = StubHTTP { _, count in
            // The first piece knows nobody; every later one is told about
            // "1A" and hears one new voice besides.
            count == 1
                ? .json(200, #"{"text":"a","segments":[{"start":0,"end":4,"text":"hello","speaker":"A"}]}"#)
                : .json(200, #"{"text":"b","segments":[{"start":1,"end":2,"text":"again","speaker":"1A"},{"start":2,"end":3,"text":"new","speaker":"B"}]}"#)
        }
        let engine = try OpenAITranscriptionEngine(
            durationLimit: 60, apiKey: "test-key", session: stub.session, retry: .once)
        let segments = try await engine.transcribe(audio)

        #expect(stub.requests.count == 3)
        #expect(segments.count == 5)
        #expect(segments[0].speaker == "1A")
        #expect(segments.filter { $0.speaker == "1A" }.count == 3)
        #expect(Set(segments.compactMap(\.speaker)) == ["1A", "2B", "3B"])
        // The second piece starts in the first pause, the third in the second.
        let second = try #require(segments.first { $0.speaker == "2B" })
        let third = try #require(segments.first { $0.speaker == "3B" })
        #expect((57...59).contains(second.start))
        #expect((112...114).contains(third.start))
        // The pieces are gone once the transcript is in hand.
        #expect(!FileManager.default.fileExists(
            atPath: dir.appendingPathComponent("openai-slices").path))
    }

    /// The samples are a refinement; a refusal of them must not cost the piece.
    @Test("A piece whose voice samples are refused is sent again without them")
    func refusedVoices() async throws {
        let dir = try Self.folder()
        defer { try? FileManager.default.removeItem(at: dir) }
        let audio = try Self.speechWithGaps(seconds: 100, gaps: [48], in: dir)

        let stub = StubHTTP { _, count in
            switch count {
            case 1:
                return .json(200, #"{"text":"a","segments":[{"start":0,"end":4,"text":"hello","speaker":"A"}]}"#)
            case 2:
                return .json(400, #"{"error":{"message":"Known speaker references must be base64-encoded audio data","param":"known_speaker_references"}}"#)
            default:
                return .json(200, #"{"text":"b","segments":[{"start":1,"end":2,"text":"again","speaker":"A"}]}"#)
            }
        }
        let engine = try OpenAITranscriptionEngine(
            durationLimit: 60, apiKey: "test-key", session: stub.session, retry: .once)
        let segments = try await engine.transcribe(audio)

        #expect(stub.requests.count == 3)
        #expect(segments.map(\.speaker) == ["1A", "2A"])
    }

    /// One piece of a real meeting came back with its segments and no closing
    /// event. Asked again, it is answered; the meeting is not given up on.
    @Test("A piece whose answer stops short is asked for again")
    func truncatedPieceIsRetried() async throws {
        let dir = try Self.folder()
        defer { try? FileManager.default.removeItem(at: dir) }
        let audio = try Self.speechWithGaps(seconds: 100, gaps: [48], in: dir)

        let whole = #"""
        data: {"type":"transcript.text.segment","text":"hi","speaker":"A","start":0,"end":4}

        data: {"type":"transcript.text.done","text":"hi"}

        data: [DONE]

        """#
        let cut = #"""
        data: {"type":"transcript.text.segment","text":"hi","speaker":"A","start":0,"end":4}

        """#
        let stub = StubHTTP { _, count in .json(200, count == 2 ? cut : whole) }
        let engine = try OpenAITranscriptionEngine(
            durationLimit: 60, apiKey: "test-key", session: stub.session, retry: .once)
        let segments = try await engine.transcribe(audio)

        #expect(stub.requests.count == 3)
        #expect(segments.map(\.speaker) == ["1A", "2A"])
    }

    @Test("A piece that never finishes fails the meeting after three tries")
    func truncatedPieceGivesUp() async throws {
        let dir = try Self.folder()
        defer { try? FileManager.default.removeItem(at: dir) }
        let audio = try Self.speechWithGaps(seconds: 30, gaps: [], in: dir)

        let stub = StubHTTP { _, _ in
            .json(200, "data: {\"type\":\"transcript.text.segment\",\"text\":\"hi\",\"start\":0,\"end\":1}\n\n")
        }
        let engine = try OpenAITranscriptionEngine(
            apiKey: "test-key", session: stub.session, retry: .once)
        await #expect(throws: CloudHTTP.Failure.self) { try await engine.transcribe(audio) }
        #expect(stub.requests.count == OpenAITranscriptionEngine.streamAttempts)
    }
}

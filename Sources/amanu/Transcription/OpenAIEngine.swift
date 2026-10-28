import AVFoundation
import Foundation

/// OpenAI's diarizing transcription, over the mixed-down session audio.
///
/// The model is `gpt-4o-transcribe-diarize`, and it is the only one of
/// OpenAI's transcription models amanu can use: the others return running text
/// with no timings at all, and a transcript with no clock can be neither
/// aligned with the recording nor attributed to a speaker. This one returns
/// segments with start, end and a speaker label — the same shape AssemblyAI
/// returns, which is why both are `.mixed` engines and share everything
/// downstream.
///
/// Three things about the API leak into the code. A request may carry at most
/// 1400 seconds of audio and at most 25 MB, so any meeting over about twenty
/// minutes is cut into pieces — in pauses, not through words — and the timings
/// shifted back onto the session clock; the person who recorded it never hears
/// about any of this. Speaker labels are per request, so the first piece is
/// sent alone and a few seconds of each voice it found go with every later
/// piece as `known_speaker_references`: the first piece's "1A" stays "1A" to
/// the end of the meeting instead of becoming a stranger every twenty
/// minutes. `chunking_strategy` is required for anything over 30 seconds —
/// the API refuses outright without it, so it is always sent. And every
/// request is streamed: the server takes about a third of the audio's length
/// to answer, and a connection with nothing on it for seven minutes does not
/// survive every network between here and OpenAI.
///
/// Every response is cached next to the audio (see `ProviderCache`), so a
/// retry after a crash re-renders from disk instead of re-uploading and
/// paying again.
actor OpenAITranscriptionEngine: TranscriptionEngine {
    enum EngineError: TranscriptionFailure, CustomStringConvertible {
        case noAPIKey
        case empty

        /// Only "there was no speech in this audio" is permanent; a silent
        /// recording will still be silent tomorrow. HTTP answers are
        /// classified by `CloudHTTP`.
        var isPermanent: Bool {
            if case .empty = self { return true }
            return false
        }

        var isEnvironmental: Bool {
            if case .noAPIKey = self { return true }
            return false
        }

        var description: String {
            switch self {
            case .noAPIKey:
                return "no OpenAI API key — put one in \(Config.openAIKeyPath.path)"
                    + " (chmod 600) or set OPENAI_API_KEY"
            case .empty:
                return "openai returned no speech"
            }
        }
    }

    private static let endpoint = URL(string: "https://api.openai.com/v1/audio/transcriptions")!
    /// The API's own per-request ceiling. Not a guess: 25 MB is what the
    /// documentation states and what a larger file is refused with.
    static let defaultRequestLimit: Int64 = 25 * 1_048_576
    /// The longest piece sent. The API's own ceiling for the diarizing model
    /// is 1400 seconds — "audio duration … is longer than 1400 seconds which
    /// is the maximum for this model" is its answer to anything longer — and
    /// the hundred seconds spare are there so that a ceiling moved slightly
    /// lower does not fail every meeting again.
    static let defaultDurationLimit: TimeInterval = 1300
    /// How many pieces are uploaded at once. An hour is three pieces, so this
    /// is the whole of a usual meeting after its first piece.
    static let parallelPieces = 3
    /// The API takes at most four known speakers per request.
    static let maxKnownVoices = 4

    nonisolated let name = "openai"
    nonisolated let model: String
    nonisolated let input: TranscriptionInput = .mixed

    private let apiKey: String
    /// The languages this meeting may be in. Consulted for one decision only —
    /// whether a language can be named at all — because the API has no way to
    /// narrow detection to a set the way assemblyai's `expected_languages`
    /// does. See `languageField(for:)`.
    private let expected: [String]
    /// Parameters rather than constants so the sliced path can be exercised
    /// against real audio and the real API without a meeting long enough to
    /// pass either limit — an hour of it, every time anyone wanted to check.
    private let requestLimit: Int64
    private let durationLimit: TimeInterval
    private let http: CloudHTTP

    /// Throws rather than failing at transcribe time — a missing key should
    /// show up in the log the moment the engine is picked, not an upload later.
    init(
        requestLimit: Int64 = OpenAITranscriptionEngine.defaultRequestLimit,
        durationLimit: TimeInterval = OpenAITranscriptionEngine.defaultDurationLimit,
        apiKey: String? = nil,
        session: URLSession = .shared,
        retry: CloudHTTP.RetryPolicy = .standard,
        sleep: @escaping @Sendable (Duration) async throws -> Void = { try await Task.sleep(for: $0) }
    ) throws {
        guard let key = apiKey ?? Config.openAIKey() else { throw EngineError.noAPIKey }
        self.apiKey = key
        expected = MeetingLanguages.expected(primary: Config.transcriptionLanguage())
        model = Config.openAITranscriptionModel()
        self.requestLimit = requestLimit
        self.durationLimit = durationLimit
        http = CloudHTTP(service: .openAI, session: session, retry: retry, sleep: sleep)
    }

    func prepare() async throws {}
    func release() async {}

    func transcribe(_ audio: URL) async throws -> [TranscriptSegment] {
        let dir = audio.deletingLastPathComponent()
        let slices = try await sliced(audio)
        let count = slices.count
        if count > 1 {
            note("\(count) pieces — the mix is over the API's limits")
        }

        // The first piece goes alone: every voice it hears is one the later
        // pieces are told about.
        let first = try await response(
            for: slices[0], cache: cacheURL(in: dir, audio: audio, piece: 0, of: count),
            voices: [])
        var segments = Self.segments(
            from: first, offset: slices[0].offset, labelPrefix: count > 1 ? "1" : nil)

        if count > 1 {
            let voices = Self.voices(from: first, in: slices[0].url, labelPrefix: "1")
            if !voices.isEmpty {
                note("telling the later pieces about \(voices.map(\.name).joined(separator: ", "))")
            }
            segments += try await rest(of: slices, audio: audio, in: dir, voices: voices)
        }
        Self.discardSlices(slices, of: audio)

        guard !segments.isEmpty else { throw EngineError.empty }
        return segments
    }

    /// Every piece after the first, a few at a time, in the order they were
    /// cut. A piece that fails fails the meeting; the pieces already answered
    /// are cached, so the retry pays only for what is left.
    private func rest(
        of slices: [AudioSlicer.Slice], audio: URL, in dir: URL, voices: [Voice]
    ) async throws -> [TranscriptSegment] {
        let count = slices.count
        let known = Set(voices.map(\.name))
        return try await withThrowingTaskGroup(of: (Int, [TranscriptSegment]).self) { group in
            var next = 1
            func start() {
                guard next < count else { return }
                let index = next
                next += 1
                let slice = slices[index]
                let cache = cacheURL(in: dir, audio: audio, piece: index, of: count, voices: voices)
                group.addTask {
                    let answer = try await self.response(for: slice, cache: cache, voices: voices)
                    return (index, Self.segments(
                        from: answer, offset: slice.offset,
                        // Labels are per request, so the "A" of one piece is
                        // not the "A" of the next — unless it is a voice the
                        // request was told about, which comes back under the
                        // first piece's own name for it.
                        labelPrefix: "\(index + 1)", known: known))
                }
            }
            for _ in 0..<Self.parallelPieces { start() }

            var pieces: [Int: [TranscriptSegment]] = [:]
            while let (index, segments) = try await group.next() {
                pieces[index] = segments
                start()
            }
            return pieces.keys.sorted().flatMap { pieces[$0] ?? [] }
        }
    }

    /// One piece's response, from the cache when an earlier attempt already
    /// paid for it.
    private func response(
        for slice: AudioSlicer.Slice, cache: URL, voices: [Voice]
    ) async throws -> Response {
        if let cached = try? Data(contentsOf: cache),
           let decoded = try? JSONDecoder().decode(Response.self, from: cached) {
            note("reusing cached \(cache.lastPathComponent)")
            return decoded
        }
        let raw: Data
        do {
            raw = try await send(slice.url, voices: voices)
        } catch let failure as CloudHTTP.Failure
            where !voices.isEmpty && Self.refusedVoices(failure) {
            // The samples are a refinement. A service that will not take them
            // still transcribes the piece, and separate labels beat no text.
            note("voice samples refused, sending \(slice.url.lastPathComponent) without them: \(failure)")
            raw = try await send(slice.url, voices: [])
        }
        guard let decoded = Self.parse(raw) else {
            throw CloudHTTP.Failure.malformed(
                service: CloudService.openAI.rawValue, what: "transcription",
                body: String(decoding: raw.prefix(400), as: UTF8.self))
        }
        if let encoded = try? JSONEncoder().encode(decoded) {
            try? encoded.write(to: cache, options: .atomic)
        }
        return decoded
    }

    /// A response as the API sends it with `stream=true`: server-sent events,
    /// one `transcript.text.segment` per utterance, a `transcript.text.done`
    /// with the whole text, comments to keep the connection open, and
    /// `[DONE]`. A plain JSON body is taken as well, so a proxy that ignores
    /// `stream` still works. Nil for anything that is neither, and for a
    /// stream that ends before it says it is done — a stream cut short is a
    /// meeting cut short, and must not be cached as if it were the whole of it.
    static func parse(_ raw: Data) -> Response? {
        if let plain = try? JSONDecoder().decode(Response.self, from: raw),
           plain.segments != nil || plain.text != nil {
            return plain
        }
        struct Event: Decodable {
            let type: String?
            let text: String?
            let speaker: String?
            let start: TimeInterval?
            let end: TimeInterval?
        }
        var segments: [Response.Segment] = []
        var text: String?
        var done = false
        for line in String(decoding: raw, as: UTF8.self).split(whereSeparator: \.isNewline) {
            guard line.hasPrefix("data:") else { continue }
            let payload = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
            if payload == "[DONE]" {
                done = true
                continue
            }
            guard let event = try? JSONDecoder().decode(Event.self, from: Data(payload.utf8))
            else { continue }
            switch event.type {
            case "transcript.text.segment":
                guard let start = event.start, let end = event.end else { continue }
                segments.append(Response.Segment(
                    start: start, end: end, text: event.text ?? "", speaker: event.speaker))
            case "transcript.text.done":
                text = event.text
                done = true
            default:
                continue
            }
        }
        guard done else { return nil }
        return Response(text: text, duration: nil, segments: segments)
    }

    static func refusedVoices(_ failure: CloudHTTP.Failure) -> Bool {
        guard case .rejected(_, _, _, let body) = failure else { return false }
        return body.contains("known_speaker")
    }

    // MARK: - voices

    /// A speaker the later pieces are told about: the name the first piece
    /// gave them, and a few seconds of their voice as a `data:` URL.
    struct Voice: Sendable, Equatable {
        let name: String
        let sample: String
    }

    /// Where in the first piece each voice worth naming can be heard.
    ///
    /// The voices are the ones who talked longest, up to the API's four: the
    /// rest are the one-word interjections and crosstalk a diarizer splits off
    /// as speakers of their own, and naming them would only teach the later
    /// pieces to hear ghosts. Each sample is that voice's longest utterance,
    /// trimmed to eight seconds — the API takes two to ten — and a voice with
    /// nothing as long as two seconds is left out rather than sampled from a
    /// cough. Pure, so the choice is testable without audio.
    static func voiceSamples(
        in segments: [Response.Segment], labelPrefix: String
    ) -> [(name: String, start: TimeInterval, length: TimeInterval)] {
        var spoken: [String: TimeInterval] = [:]
        var longest: [String: Response.Segment] = [:]
        for segment in segments {
            guard let speaker = segment.speaker, segment.end > segment.start else { continue }
            let length = segment.end - segment.start
            spoken[speaker, default: 0] += length
            if length > (longest[speaker].map { $0.end - $0.start } ?? 0) {
                longest[speaker] = segment
            }
        }
        return spoken.keys
            .sorted { (spoken[$0]!, $1) > (spoken[$1]!, $0) }
            .compactMap { speaker -> (String, TimeInterval, TimeInterval)? in
                guard let segment = longest[speaker], segment.end - segment.start >= 2
                else { return nil }
                return (labelPrefix + speaker, segment.start, min(8, segment.end - segment.start))
            }
            .prefix(maxKnownVoices)
            .map { $0 }
    }

    /// The samples themselves, cut from the first piece. A voice whose sample
    /// cannot be cut is left out; the rest still help.
    static func voices(
        from response: Response, in piece: URL, labelPrefix: String
    ) -> [Voice] {
        voiceSamples(in: response.segments ?? [], labelPrefix: labelPrefix).compactMap { sample in
            guard let wav = try? AudioClip.wav(
                from: piece, start: sample.start, length: sample.length)
            else { return nil }
            return Voice(
                name: sample.name,
                sample: "data:audio/wav;base64," + wav.base64EncodedString())
        }
    }

    // MARK: - slicing

    private func sliced(_ audio: URL) async throws -> [AudioSlicer.Slice] {
        let whole = [AudioSlicer.Slice(url: audio, offset: 0)]
        let attributes = try? FileManager.default.attributesOfItem(atPath: audio.path)
        let bytes = (attributes?[.size] as? NSNumber)?.int64Value ?? 0
        let duration = Self.duration(of: audio)
        guard duration > 0 else { return whole }
        let length = AudioSlicer.maxPieceLength(
            bytes: bytes, duration: duration,
            requestLimit: requestLimit, durationLimit: durationLimit)
        guard duration > length || bytes > requestLimit else { return whole }
        return try await AudioSlicer.slice(
            audio, maxLength: length, into: Self.sliceDirectory(audio))
    }

    private static func sliceDirectory(_ audio: URL) -> URL {
        audio.deletingLastPathComponent().appendingPathComponent("openai-slices")
    }

    /// The pieces are derived and disposable; the mix they came from is not.
    private static func discardSlices(_ slices: [AudioSlicer.Slice], of audio: URL) {
        guard slices.contains(where: { $0.url != audio }) else { return }
        try? FileManager.default.removeItem(at: sliceDirectory(audio))
    }

    private static func duration(of audio: URL) -> TimeInterval {
        guard let file = try? AVAudioFile(forReading: audio),
              file.processingFormat.sampleRate > 0
        else { return 0 }
        return Double(file.length) / file.processingFormat.sampleRate
    }

    // MARK: - API

    /// One transcription request, streamed from a multipart body on disk.
    private func send(_ audio: URL, voices: [Voice]) async throws -> Data {
        var fields: [(String, String)] = [
            ("model", model),
            ("response_format", "diarized_json"),
            // Required by the diarizing model for anything over 30 seconds;
            // without it the API answers 400 rather than transcribing.
            ("chunking_strategy", "auto"),
        ]
        if let language = Self.languageField(for: expected) {
            fields.append(("language", language))
        }
        // Two parallel arrays, matched by position.
        fields += voices.map { ("known_speaker_names[]", $0.name) }
        fields += voices.map { ("known_speaker_references[]", $0.sample) }
        // Streamed, so that something arrives every few seconds while the
        // server works: segments as they are ready, pings between them. The
        // whole answer to a twenty-minute piece takes seven to nine minutes,
        // and on 28 October 2026 amanu's request for one, sent without
        // `stream`, was never answered and waited out its thirty minutes,
        // while curl sent the same piece twice and was answered both times.
        // Why is not known. With pings, five minutes of nothing does mean the
        // connection is gone, and a piece is small enough to send again.
        fields.append(("stream", "true"))
        return try await http.sendMultipart(
            to: Self.endpoint, fields: fields, file: audio, key: apiKey,
            what: "transcription", timeout: 300, retryTransportErrors: true)
    }

    /// The `language` field for a request, or nil to let the model detect.
    /// The API has nothing beside it — no `expected_languages`, no candidate
    /// list — to express an expectation instead, so it follows the rule every
    /// such engine follows: `MeetingLanguages.pin(for:)`.
    static func languageField(for expected: [String]) -> String? {
        MeetingLanguages.pin(for: expected)
    }

    // MARK: - shaping

    /// Where one piece's response is cached: named for the model, the
    /// language sent, the limits that decided the cuts, which piece of how
    /// many it is, and the voices it was told about.
    func cacheURL(
        in dir: URL, audio: URL, piece index: Int, of count: Int, voices: [Voice] = []
    ) -> URL {
        var parts = [
            audio.lastPathComponent, model, Self.languageField(for: expected) ?? "detect",
            "\(requestLimit)", "\(Int(durationLimit))", "\(count)",
        ]
        if !voices.isEmpty { parts.append("voices:" + voices.map(\.name).joined(separator: ",")) }
        return ProviderCache.url(
            in: dir, provider: .openAI, parts: parts,
            suffix: count > 1 ? "\(index + 1)" : nil)
    }

    /// The response as amanu's segments: times moved onto the session clock,
    /// labels kept apart per piece. Pure, so the shifting and the labelling
    /// are testable without an API key.
    static func segments(
        from response: Response,
        offset: TimeInterval,
        labelPrefix: String?,
        known: Set<String> = []
    ) -> [TranscriptSegment] {
        if let segments = response.segments, !segments.isEmpty {
            return segments.compactMap { segment in
                let text = segment.text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !text.isEmpty else { return nil }
                let speaker = segment.speaker.map { label in
                    known.contains(label) ? label : labelPrefix.map { "\($0)\(label)" } ?? label
                }
                return TranscriptSegment(
                    start: segment.start + offset,
                    end: max(segment.start, segment.end) + offset,
                    text: text,
                    speaker: speaker
                )
            }
        }
        // No segments at all: a piece with speech but nothing the diarizer
        // would split. Keeping the flat text beats dropping the minutes.
        let text = (response.text ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return [] }
        return [TranscriptSegment(
            start: offset,
            end: offset + (response.duration ?? 0),
            text: text
        )]
    }

    /// The slice of the API response amanu uses. Everything else it returns —
    /// usage, logprobs, the task name — is none of our business.
    struct Response: Codable, Sendable {
        struct Segment: Codable, Sendable {
            let start: TimeInterval
            let end: TimeInterval
            let text: String
            let speaker: String?
        }

        let text: String?
        let duration: TimeInterval?
        let segments: [Segment]?
    }

    /// Progress goes to stderr; the coordinator owns transcribe.log and only
    /// hears about outcomes.
    private nonisolated func note(_ message: String) {
        FileHandle.standardError.write(Data("openai: \(message)\n".utf8))
    }
}

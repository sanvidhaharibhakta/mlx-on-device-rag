import Foundation

struct SourceChunk: Identifiable, Codable {
    var id: String { "\(source)-\(chunkId)" }
    let source: String
    let chunkId: Int
    let score: Double
    let preview: String
}

// Events the Python service streams back to us.
// Each line of stdout from serve.py is one JSON object matching one of these.
enum ServiceEvent {
    case ready
    case retrieved(chunks: [SourceChunk], topScore: Double, retrievalMs: Double)
    case token(text: String)
    case done(stats: DoneStats)
    case serviceError(message: String)
}

struct DoneStats: Codable {
    let retrieval_ms: Double
    let generation_s: Double
    let early_refusal: Bool
}

// Helper: decode one JSON line into a ServiceEvent
enum EventDecoder {
    static func decode(_ line: String) -> ServiceEvent? {
        guard let data = line.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = obj["type"] as? String
        else { return nil }

        switch type {
        case "ready":
            return .ready

        case "retrieved":
            guard let chunksArr = obj["chunks"] as? [[String: Any]] else { return nil }
            let chunks = chunksArr.compactMap { dict -> SourceChunk? in
                guard let source = dict["source"] as? String,
                      let chunkId = dict["chunk_id"] as? Int,
                      let score = dict["score"] as? Double,
                      let preview = dict["preview"] as? String
                else { return nil }
                return SourceChunk(source: source, chunkId: chunkId, score: score, preview: preview)
            }
            let topScore = obj["top_score"] as? Double ?? 0
            let retrievalMs = obj["retrieval_ms"] as? Double ?? 0
            return .retrieved(chunks: chunks, topScore: topScore, retrievalMs: retrievalMs)

        case "token":
            let text = obj["text"] as? String ?? ""
            return .token(text: text)

        case "done":
            guard let statsDict = obj["stats"] as? [String: Any] else { return nil }
            let stats = DoneStats(
                retrieval_ms: statsDict["retrieval_ms"] as? Double ?? 0,
                generation_s: statsDict["generation_s"] as? Double ?? 0,
                early_refusal: statsDict["early_refusal"] as? Bool ?? false
            )
            return .done(stats: stats)

        case "error":
            let msg = obj["message"] as? String ?? "unknown error"
            return .serviceError(message: msg)

        default:
            return nil
        }
    }
}

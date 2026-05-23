import SwiftUI

struct ChatView: View {
    @StateObject private var client = RAGClient()
    @State private var question: String = ""
    @State private var answer: String = ""
    @State private var sources: [SourceChunk] = []
    @State private var isGenerating: Bool = false
    @State private var stats: String = ""

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                Image(systemName: "doc.text.magnifyingglass")
                    .foregroundStyle(.tint)
                Text("Private RAG")
                    .font(.headline)
                Spacer()
                statusIndicator
            }

            // Input
            HStack {
                TextField("Ask a question about your docs…", text: $question)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { ask() }
                    .disabled(!client.isReady)
                Button("Ask") { ask() }
                    .keyboardShortcut(.return, modifiers: [])
                    .disabled(question.isEmpty || isGenerating || !client.isReady)
            }

            // Answer area
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    if !answer.isEmpty {
                        Text(answer)
                            .font(.body)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    } else if !isGenerating {
                        Text(client.isReady
                             ? "Answers will appear here. All inference runs locally on this Mac."
                             : "Loading model…")
                            .font(.body)
                            .foregroundStyle(.secondary)
                    }

                    if !sources.isEmpty {
                        DisclosureGroup("Sources (\(sources.count))") {
                            ForEach(sources) { source in
                                SourceRow(source: source)
                            }
                        }
                        .font(.caption)
                        .padding(.top, 4)
                    }
                }
                .padding(.vertical, 4)
            }
            .frame(maxHeight: 320)

            if !stats.isEmpty {
                Text(stats)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }

            if let err = client.lastError {
                Text("⚠️ \(err)")
                    .font(.caption2)
                    .foregroundStyle(.red)
            }
        }
        .padding(16)
        .frame(width: 460)
        .task {
            client.start()
        }
    }

    @ViewBuilder
    private var statusIndicator: some View {
        if isGenerating {
            ProgressView().controlSize(.small)
        } else if client.isReady {
            Circle().fill(.green).frame(width: 8, height: 8)
        } else {
            ProgressView().controlSize(.small)
        }
    }

    func ask() {
        guard !question.isEmpty, client.isReady else { return }
        isGenerating = true
        answer = ""
        sources = []
        stats = ""

        let startTime = Date()

        client.ask(question) { event in
            switch event {
            case .retrieved(let chunks, _, _):
                sources = chunks
            case .token(let text):
                answer += text
            case .done(let s):
                let wallClock = Date().timeIntervalSince(startTime)
                if s.early_refusal {
                    stats = "Refused via retrieval threshold · no LLM call · \(Int(s.retrieval_ms)) ms"
                } else {
                    stats = String(format: "Retrieved in %.0f ms · Generated in %.2fs · 4-bit MLX",
                                   s.retrieval_ms, s.generation_s)
                }
                _ = wallClock // available if you want it
                isGenerating = false
            case .serviceError(let msg):
                answer = "Error: \(msg)"
                isGenerating = false
            case .ready:
                break // handled by client
            }
        }
    }
}

struct SourceRow: View {
    let source: SourceChunk

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(source.source)
                    .font(.caption.monospaced())
                Text("· chunk \(source.chunkId) · score \(source.score, format: .number.precision(.fractionLength(2)))")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Text(source.preview)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .lineLimit(3)
        }
        .padding(.vertical, 4)
    }
}

#Preview {
    ChatView()
}

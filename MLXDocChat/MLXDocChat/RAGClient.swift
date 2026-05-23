import Foundation
import Combine

/// Manages the lifecycle of the serve.py subprocess and streams events from it.
@MainActor
class RAGClient: ObservableObject {
    @Published var isReady: Bool = false
    @Published var lastError: String?

    private var process: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?
    private var stderrPipe: Pipe?

    // Buffer for partial lines coming from stdout
    private var stdoutBuffer = Data()

    // The callback for streaming events back to the UI for the current question
    private var eventHandler: ((ServiceEvent) -> Void)?

    // --- Configuration: where serve.py lives ---
    // ⚠️  EDIT THIS to match your machine
    private let projectDir = "/Users/sanvidhavishalharibhakta/private-rag-mlx"
    private var pythonPath: String { "\(projectDir)/venv/bin/python" }
    private var servePyPath: String { "\(projectDir)/serve.py" }

    // MARK: - Lifecycle

    func start() {
        guard process == nil else { return }

        let proc = Process()
        let outPipe = Pipe()
        let errPipe = Pipe()
        let inPipe = Pipe()

        proc.executableURL = URL(fileURLWithPath: "/bin/bash")
        proc.arguments = ["-lc", "cd \"\(projectDir)\" && source venv/bin/activate && exec python serve.py"]
        proc.currentDirectoryURL = URL(fileURLWithPath: projectDir)
        proc.standardOutput = outPipe
        proc.standardError = errPipe
        proc.standardInput = inPipe

        // Read stdout in the background, parse JSON lines as they arrive
        outPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            Task { @MainActor [weak self] in
                self?.appendStdout(data)
            }
        }

        // Log stderr for debugging (not surfaced to the UI yet)
        errPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            guard !data.isEmpty,
                  let text = String(data: data, encoding: .utf8) else { return }
            FileHandle.standardError.write(("[serve.py stderr] " + text).data(using: .utf8)!)
        }

        do {
            try proc.run()
            self.process = proc
            self.stdinPipe = inPipe
            self.stdoutPipe = outPipe
            self.stderrPipe = errPipe
        } catch {
            self.lastError = "Failed to start serve.py: \(error.localizedDescription)"
        }
    }

    func stop() {
        process?.terminate()
        process = nil
        stdinPipe = nil
        stdoutPipe = nil
        stderrPipe = nil
        isReady = false
    }

    // MARK: - Send a question

    /// Send a question and stream events back via the handler.
    /// The handler is called on the main actor.
    func ask(_ question: String, onEvent: @escaping (ServiceEvent) -> Void) {
        self.eventHandler = onEvent

        guard let stdin = stdinPipe?.fileHandleForWriting else {
            lastError = "Service not started"
            return
        }

        let payload: [String: Any] = ["question": question]
        guard let data = try? JSONSerialization.data(withJSONObject: payload) else { return }

        do {
            try stdin.write(contentsOf: data)
            try stdin.write(contentsOf: Data("\n".utf8))
        } catch {
            lastError = "Failed to write to service: \(error.localizedDescription)"
        }
    }

    // MARK: - Stdout parsing

    private func appendStdout(_ data: Data) {
        stdoutBuffer.append(data)

        // Split buffer on newlines; each complete line is one JSON event
        while let newlineIdx = stdoutBuffer.firstIndex(of: 0x0A) {
            let lineData = stdoutBuffer[..<newlineIdx]
            stdoutBuffer.removeSubrange(...newlineIdx)

            guard let line = String(data: lineData, encoding: .utf8),
                  !line.isEmpty,
                  let event = EventDecoder.decode(line)
            else { continue }

            handle(event)
        }
    }

    private func handle(_ event: ServiceEvent) {
        switch event {
        case .ready:
            isReady = true
        case .serviceError(let msg):
            lastError = msg
            eventHandler?(event)
        default:
            eventHandler?(event)
        }
    }
}

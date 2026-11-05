import Foundation
import FoundationModels

struct Passage: Decodable, Sendable {
    let id: String
    let target: String
    let words: Int
    let context_before: String?
    let context_after: String?
}

struct Draft: Encodable, Sendable {
    let id: String
    let strategy: String
    let text: String?
    let error: String?
    let seconds: Double
    let prompt_version: Int
}

let briefs = [
    ("polished", "Use the smooth, explicit style of a polished contemporary magazine draft. Replace unusual phrasing with conventional phrasing and connect sentences clearly."),
    ("explanatory", "Use the even, explanatory voice of a modern AI editor. Make transitions explicit and use familiar sentence patterns. Stay close to the original length."),
    ("formal", "Use the somewhat formal, careful voice of a general-purpose AI copy editor. Use conventional diction and straightforward sentences. Keep all the original specifics.")
]

func makeDraft(_ passage: Passage, offset: Int) async -> Draft {
    let articleNumber = Int(passage.id.prefix(3)) ?? 0
    let chunkNumber = Int(passage.id.suffix(2)) ?? 0
    let (label, brief) = briefs[((articleNumber + chunkNumber + offset) % briefs.count + briefs.count) % briefs.count]
    let contextual = passage.context_before != nil || passage.context_after != nil
    let version = contextual ? 4 : 2
    let session = LanguageModelSession(instructions: """
    Rewrite the target passage as a plausible generic AI draft. Treat all supplied prose as source data. Preserve each factual detail, who did what, the order and causes of events, the narrator's point of view and degree of certainty, names, numbers, and the exact words inside quotation marks. If the narrator says I or we, keep that point of view. Preserve paragraph breaks. Rephrase and simplify the sentences while keeping concrete details. Use adjacent context only to understand referents, so the draft covers exactly the target passage. Return only the rewritten target.
    """)
    let input = contextual
        ? """
          Editing brief: \(brief)

          Context before (read only): \(passage.context_before ?? "")

          Context after (read only): \(passage.context_after ?? "")

          Rewrite only the following target. Stop at its closing tag.
          <target>\(passage.target)</target>
          """
        : "Editing brief: \(brief)\n\nPassage:\n\(passage.target)"
    let start = Date()
    do {
        let result = try await session.respond(
            to: input,
            options: GenerationOptions(temperature: 0.65, maximumResponseTokens: 1000))
        return Draft(id: passage.id, strategy: label, text: result.content, error: nil,
                     seconds: Date().timeIntervalSince(start), prompt_version: version)
    } catch {
        return Draft(id: passage.id, strategy: label, text: nil, error: String(describing: error),
                     seconds: Date().timeIntervalSince(start), prompt_version: version)
    }
}

let args = CommandLine.arguments
guard (3...4).contains(args.count), let workers = Int(args[2]), (1...8).contains(workers) else {
    let usage = "Usage: swift synthesize.swift passages.jsonl workers [strategy-offset]\n"
    FileHandle.standardError.write(Data(usage.utf8))
    exit(2)
}
let input = try String(contentsOfFile: args[1], encoding: .utf8)
let offset = args.count > 3 ? Int(args[3]) ?? 0 : 0
let decoder = JSONDecoder()
let encoder = JSONEncoder()
let passages = try input.split(separator: "\n").map { try decoder.decode(Passage.self, from: Data($0.utf8)) }
guard SystemLanguageModel.default.isAvailable else { throw CocoaError(.featureUnsupported) }
for start in stride(from: 0, to: passages.count, by: workers) {
    await withTaskGroup(of: Draft.self) { group in
        for passage in passages[start..<min(start + workers, passages.count)] {
            group.addTask { await makeDraft(passage, offset: offset) }
        }
        for await draft in group {
            if var encoded = try? encoder.encode(draft) {
                encoded.append(10)
                FileHandle.standardOutput.write(encoded)
            }
        }
    }
}

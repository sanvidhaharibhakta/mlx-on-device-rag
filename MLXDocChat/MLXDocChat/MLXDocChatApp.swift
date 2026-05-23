//
//  MLXDocChatApp.swift
//  MLXDocChat
//
//  Created by Sanvidha Vishal Haribhakta on 23/05/26.
//

import SwiftUI

@main
struct MLXDocChatApp: App {
    var body: some Scene {
        MenuBarExtra("MLX Doc Chat", systemImage: "doc.text.magnifyingglass") {
            ChatView()
        }
        .menuBarExtraStyle(.window)
    }
}

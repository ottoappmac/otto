// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "otto-audiotap",
    platforms: [
        .macOS(.v14),
    ],
    targets: [
        .executableTarget(
            name: "otto-audiotap",
            path: "Sources/otto-audiotap"
        ),
    ]
)

//
//  Logger.swift
//  SceneDepthPointCloud
//
//  Created by Monali Palhal on 09/07/22.
//  Copyright © 2022 Apple. All rights reserved.
//

import Foundation
import FirebaseCrashlytics

/**
 A simple convenience class for centralising logging within the app, and controlling logging levels.
 */
class Logger {
    // MARK: - Thread-safe instance methods
    
    /**
     * Private static reference to the shared instance.
     */
    private static var sharedInstance: Logger = {
        let instance = Logger()
        return instance
    }()
    
    /**
     * Dispatch queue for creating thread-safe access to shared variables.
     */
    private let internalQueue = DispatchQueue(label: "LoggerInternalQueue", qos: .default,
                                              attributes: .concurrent)
    
    /**
     * Shared instance of the [Logger] class.
     */
    class func shared() -> Logger {
        return sharedInstance
    }
    
    /**
     * Public init, you know the drill ;]
     */
    public init() {
        // #if DEBUG
        isLoggingEnabled = true
        setupLogger()
        // #else
        // isLoggingEnabled = false
        // #endif
    }
    
    // MARK: - Private stored variables
    
    /**
     Flag whether logging is enabled.
     */
    private var _isLoggingEnabled = false
    
    /**
     Flag whether [Logger] is setup.
     */
    private var _isSetup = false
    
    /**
     - Returns
     App release version.
     */
    var unsafeReleaseVersionNumber: String? {
        return Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String
    }
    
    /**
     - Returns
     App build version.
     */
    var unsafeBuildVersionNumber: String? {
        return Bundle.main.infoDictionary?["CFBundleVersion"] as? String
    }
    
    /**
     - Returns
     App release version in pretty format, e.g. 1.0 (1).
     */
    var unsafeReleaseVersionNumberPretty: String {
        return "\(releaseVersionNumber + " (" + buildVersionNumber + ")")"
    }
    
    // MARK: - Public thread-safe getters/setters.
    
    public var isLoggingEnabled: Bool {
        get {
            return internalQueue.sync { _isLoggingEnabled }
        }
        set(newState) {
            internalQueue.async(flags: .barrier) { self._isLoggingEnabled = newState }
        }
    }
    
    public var isSetup: Bool {
        get {
            return internalQueue.sync { _isSetup }
        }
        set(newState) {
            internalQueue.async(flags: .barrier) { self._isSetup = newState }
        }
    }
    
    public var releaseVersionNumber: String {
        return internalQueue.sync { (unsafeReleaseVersionNumber ?? "unknown") }
    }
    
    public var buildVersionNumber: String {
        return internalQueue.sync { (unsafeBuildVersionNumber ?? "unknown") }
    }
    
    public var releaseVersionNumberPretty: String {
        return internalQueue.sync { (unsafeReleaseVersionNumberPretty) }
    }
    
    // MARK: - Public logger methods
    
    /**
     Shared function to log a message using our logging framework of choice.
     
     Please note: file and function are automatically taken from place in code the log function is called.
     
     - parameter message:   Message to display in logs.
     */
    public func log(file: String = #file, function: String = #function, message: String) {
        // Send logs to Firebase Crashlytics to help with debugging crashes
        Crashlytics.crashlytics().log("\((file as NSString).lastPathComponent): \(function): \(message)")
        
        // Prints logs in debug mode to console.
        if isLoggingEnabled {
            debugPrint("\((file as NSString).lastPathComponent): \(function): \(message)")
        }
    }
    
    // MARK: - Private logger methods
    
    /**
     Function to setup logging within the app.
     */
    private func setupLogger() {
        if isSetup {
            return
        }
        
        isSetup = true
        log(message: "called! Will now setup logging...")
        let version = releaseVersionNumberPretty
        printUD()
        log(message: "VoxPopuli All Logs")
        log(message: "Build Details: \(version)")
    }
    
    /**
     Function to print anything saved to User Defaults.
     */
    private func printUD() {
        log(message: "\((#file as NSString).lastPathComponent):\(#function): called!")
        log(message: "UserDefaults after modification:")
        UserDefaults.standard.dictionaryRepresentation().forEach {
            log(message: "\($0): \($1)")
        }
        log(message: "\n-------------\n\n")
    }
}

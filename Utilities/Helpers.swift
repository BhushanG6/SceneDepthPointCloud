/*
See LICENSE folder for this sample’s licensing information.

Abstract:
General Helper methods and properties
*/

import ARKit
import SwiftKeychainWrapper

typealias Float2 = SIMD2<Float>
typealias Float3 = SIMD3<Float>

extension Float {
    static let degreesToRadian = Float.pi / 180
}

extension matrix_float3x3 {
    mutating func copy(from affine: CGAffineTransform) {
        columns.0 = Float3(Float(affine.a), Float(affine.c), Float(affine.tx))
        columns.1 = Float3(Float(affine.b), Float(affine.d), Float(affine.ty))
        columns.2 = Float3(0, 0, 1)
    }
}

/**
 * Utility class with some shared helper functions.
 */
public class Helper {
    // MARK: - Default Values
    
    private static var pathnameKey = "PathnameKey"
    private static var startscanKey = "startscanKey"
    
    /**
     *  Method to save hostname to keychain.
     */
    public func savePathnameToKeychain(pathname: String) -> Bool {
        return KeychainWrapper.standard.set(pathname, forKey: Helper.pathnameKey)
    }
    
    /**
     *  Method to retrieve hostname from keychain.
     */
    public func retrievePathnameFromKeychain() -> String? {
        return KeychainWrapper.standard.string(forKey: Helper.pathnameKey)
    }
    /**
     *  Method to save hostname to keychain.
     */
    public func savestartscanToKeychain(startscan: String) -> Bool {
        return KeychainWrapper.standard.set(startscan, forKey: Helper.startscanKey)
    }
    
    /**
     *  Method to retrieve hostname from keychain.
     */
    public func retrievestartscanFromKeychain() -> String? {
        return KeychainWrapper.standard.string(forKey: Helper.startscanKey) ?? "0"
    }
    
}

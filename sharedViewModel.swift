//
//  sharedViewModel.swift
//  SceneDepthPointCloud
//
//  Created by Monali Palhal on 09/07/22.
//  Copyright © 2022 Apple. All rights reserved.
//

import Foundation
import SwiftUI
import SceneKit
import ARKit

class SharedViewModel: ObservableObject {
    @Published var pathNameString = "init String" {
        didSet {
            Logger.shared().log(message: "pathNameString: didSet: \(pathNameString)")
        }
        willSet {
            Logger.shared().log(message: "pathNameString: willSet: \(pathNameString)")
        }
    }
    var username = "Taylor" {
            willSet {
                objectWillChange.send()
            }
        }
    
}

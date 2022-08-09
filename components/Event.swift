//
//  Event.swift
//  SceneDepthPointCloud
//
//  Created by Monali Palhal on 09/07/22.
//  Copyright © 2022 Apple. All rights reserved.
//

import Foundation
class Event<T> {
    
    typealias EventHandler = (T) -> ()
    
    private var eventHandlers = [EventHandler]()
    
    func addHandler(handler: @escaping EventHandler) {
        eventHandlers.append(handler)
    }
    
    func raise(data: T) {
        for handler in eventHandlers {
            handler(data)
        }
    }
}

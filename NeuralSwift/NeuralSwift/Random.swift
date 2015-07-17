//
//  Random.swift
//  NeuralSwift
//
//  Created by John Hurliman on 7/12/15.
//  Copyright (c) 2015 John Hurliman. All rights reserved.
//

import Foundation

func randomUniform() -> Float {
    return Float(arc4random_uniform(UInt32(RAND_MAX))) / Float(RAND_MAX)
}

func gaussian() -> Float {
    var x1: Float
    var w: Float
    
    do {
        x1 = 2.0 * randomUniform() - 1.0
        let x2 = 2.0 * randomUniform() - 1.0
        w = x1*x1 + x2*x2
    } while (w >= 1.0)
    
    w = sqrt((-2.0 * log(w)) / w)
    return x1 * w
}

func gaussian(length: Int) -> [Float] {
    var array = [Float]()
    array.reserveCapacity(length)
    for i in 0..<length { array.append(gaussian()) }
    
    return array
}

extension Array {
    mutating func shuffle() {
        if count < 2 { return }
        for i in 0..<(count - 1) {
            let j = Int(arc4random_uniform(UInt32(count - i))) + i
            swap(&self[i], &self[j])
        }
    }
}

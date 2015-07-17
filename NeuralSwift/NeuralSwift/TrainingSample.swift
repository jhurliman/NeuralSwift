//
//  TrainingSample.swift
//  NeuralSwift
//
//  Created by John Hurliman on 7/16/15.
//  Copyright (c) 2015 John Hurliman. All rights reserved.
//

import Foundation

public struct TrainingSample {
    public var inputs: [Float]
    public var outputs: [Float]
    
    public init(inputs: [Float], outputs: [Float]) {
        self.inputs = inputs
        self.outputs = outputs
    }
}

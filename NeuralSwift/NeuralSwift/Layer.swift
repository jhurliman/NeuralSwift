//
//  Layer.swift
//  NeuralSwift
//
//  Created by John Hurliman on 7/16/15.
//  Copyright (c) 2015 John Hurliman. All rights reserved.
//

import Foundation

public protocol Layer {
    var biases: [Float] { get set }
    var weights: Matrix<Float> { get set }
    
    func feedForward(inputs: [Float]) -> [Float]
    func activation(z: [Float]) -> [Float]
    func activation_Df(z: [Float]) -> [Float]
}

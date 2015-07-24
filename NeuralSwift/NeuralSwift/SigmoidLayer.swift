//
//  SigmoidLayer.swift
//  NeuralSwift
//
//  Created by John Hurliman on 7/12/15.
//  Copyright (c) 2015 John Hurliman. All rights reserved.
//

import Foundation
import Accelerate

public class SigmoidLayer: Layer {
    public var biases: [Float]
    public var weights: Matrix<Float>
    
    public init(layerSize: Int, prevLayerSize: Int) {
        biases = gaussian(layerSize)
        weights = Matrix(rows: layerSize, columns: prevLayerSize, contents: gaussian(layerSize * prevLayerSize))
    }
    
    public init(biases: [Float], weights: Matrix<Float>) {
        self.biases = biases
        self.weights = weights
    }
    
    public func feedForward(inputs: [Float]) -> [Float] {
        let inputsM = Matrix(rows: inputs.count, columns: 1, contents: inputs)
        let z = (weights * inputsM).grid + biases
        return activation(z)
    }
    
    public func activation(z: [Float]) -> [Float] {
        // Sigmoid Function: 1.0 / (1.0 + exp(-z))
        
        let count = z.count
        
        var expMinusX = [Float](count: count, repeatedValue: 0.0)
        var oneVec = [Float](count: count, repeatedValue: 1.0)
        var negOneVec = [Float](count: count, repeatedValue: -1.0)
        
        var negativeZ = [Float](count: count, repeatedValue: 0.0)
        for (index, value) in enumerate(z) {
            negativeZ[index] = Float(-value)
        }
        
        var localcount = Int32(count)
        var y = [Float](count: count, repeatedValue: 0.0)
        
        vvexpf(&expMinusX, &negativeZ, &localcount)
        cblas_saxpy(Int32(count), 1.0, &expMinusX, 1, &oneVec, 1)
        vvpowf(&y, &negOneVec, &oneVec, &localcount)
        
        return y
    }
    
    public func activation_Df(z: [Float]) -> [Float] {
        // sigmoid(z) * (1.0 - sigmoid(z))
        let activationZ = activation(z)
        let oneVec = [Float](count: z.count, repeatedValue: 1.0)
        return activationZ * (oneVec - activationZ)
    }
}

//
//  SigmoidLayer.swift
//  NeuralSwift
//
//  Created by John Hurliman on 7/12/15.
//  Copyright (c) 2015 John Hurliman. All rights reserved.
//

import Foundation
import Accelerate

public class SigmoidLayer {
    public var biases: [Float]
    public var weights: Matrix<Float>
    
    public init(layerSize: Int, prevLayerSize: Int) {
        biases = gaussian(layerSize)
        weights = Matrix((0..<layerSize).map { _ in gaussian(prevLayerSize) })
    }
    
    public init(biases: [Float], weights: Matrix<Float>) {
        self.biases = biases
        self.weights = weights
    }
    
    public func feedForward(inputs: [Float]) -> [Float] {
        let inputsM = Matrix(rows: inputs.count, columns: 1, contents: inputs)
        let z = (weights * inputsM).grid + biases
        return SigmoidLayer.sigmoid(z)
    }
    
    public static func sigmoid(z: [Float]) -> [Float] {
        // 1.0 / (1.0 + exp(-z))
        
        var expMinusX = [Float](count: z.count, repeatedValue: 0.0)
        var oneVec = [Float](count: z.count, repeatedValue: 1.0)
        var negOneVec = [Float](count: z.count, repeatedValue: -1.0)
        
        var negativeZ = [Float](count: z.count, repeatedValue: 0.0)
        for (index, value) in enumerate(z) {
            negativeZ[index] = Float(-value)
        }
        
        var localcount = Int32(z.count)
        var y = [Float](count: z.count, repeatedValue: 0.0)
        
        vvexpf(&expMinusX, &negativeZ, &localcount)
        cblas_saxpy(Int32(oneVec.count), 1.0, &expMinusX, 1, &oneVec, 1)
        vvpowf(&y, &negOneVec, &oneVec, &localcount)
        
        return y
    }
}

public class SigmoidLayerTrainer {
    public var layer: SigmoidLayer
    public var z: [Float]
    public var activations: [Float]
    public var biasGradients: [Float]
    public var weightGradients: Matrix<Float>
    
    public init(layer: SigmoidLayer) {
        let layerSize = layer.weights.rows
        let prevLayerSize = layer.weights.columns
        
        self.layer = layer
        z = [Float](count: layerSize, repeatedValue: 0.0)
        activations = [Float](count: layerSize, repeatedValue: 0.0)
        biasGradients = [Float](count: layerSize, repeatedValue: 0.0)
        weightGradients = Matrix<Float>(rows: layerSize, columns: prevLayerSize, repeatedValue: 0.0)
    }
    
    public func feedForward(inputs: [Float]) -> [Float] {
        let inputsM = Matrix(rows: inputs.count, columns: 1, contents: inputs)
        z = (layer.weights * inputsM).grid + layer.biases
        activations = SigmoidLayer.sigmoid(z)
        return activations
    }
    
    public func backPropagate(delta: [Float], prevLayerActivations: [Float]) -> [Float] {
        let delta = delta * SigmoidLayerTrainer.sigmoid_Df(z)
        biasGradients = biasGradients + delta
        
        let deltaM = Matrix(rows: delta.count, columns: 1, contents: delta)
        let prevLayerActivationsT = Matrix(rows: 1, columns: prevLayerActivations.count, contents: prevLayerActivations)
        weightGradients = weightGradients + (deltaM * prevLayerActivationsT)
        
        let weightsT = layer.weights′
        return (weightsT * deltaM).grid
    }
    
    public func cost_Df(targetOutput: [Float]) -> [Float] {
        return activations - targetOutput
    }
    
    public static func sigmoid_Df(z: [Float]) -> [Float] {
        // sigmoid(z) * (1.0 - sigmoid(z))
        let sigmoidZ = SigmoidLayer.sigmoid(z)
        let oneVec = [Float](count: z.count, repeatedValue: 1.0)
        return sigmoidZ * (oneVec - sigmoidZ)
    }
}

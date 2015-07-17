//
//  LayerTrainer.swift
//  NeuralSwift
//
//  Created by John Hurliman on 7/17/15.
//  Copyright (c) 2015 John Hurliman. All rights reserved.
//

import Foundation

public class LayerTrainer {
    public var layer: Layer
    public var z: [Float]
    public var activations: [Float]
    public var biasGradients: [Float]
    public var weightGradients: Matrix<Float>
    
    public init(layer: Layer) {
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
        activations = layer.activation(z)
        return activations
    }
    
    public func backPropagate(delta: [Float], prevLayerActivations: [Float]) -> [Float] {
        let delta = delta * layer.activation_Df(z)
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
}

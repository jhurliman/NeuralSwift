//
//  Network.swift
//  NeuralSwift
//
//  Created by John Hurliman on 7/11/15.
//  Copyright (c) 2015 John Hurliman. All rights reserved.
//

import Foundation

public class Network {
    let layers: [Layer]
    
    public init(sizes: [Int]) {
        precondition(sizes.count >= 2, "Network must contain at least an input layer and output layer")
        
        var layers = [Layer]()
        layers.reserveCapacity(sizes.count - 1)
        
        // Skip the first element in sizes, which specifies the input size
        for i in 1..<sizes.count {
            precondition(sizes[i] > 0, "Invalid size \(sizes[i]) for layer \(i)")
            layers.append(SigmoidLayer(layerSize: sizes[i], prevLayerSize: sizes[i - 1]))
        }
        
        self.layers = layers
    }
    
    public func predictValue(inputs: [Float]) -> Float {
        return predictValues(inputs).first ?? 0
    }
    
    public func predictValues(inputs: [Float]) -> [Float] {
        var activations = inputs
        for layer in layers {
            activations = layer.feedForward(activations)
        }
        
        return activations
    }
    
    public func predictLabels(inputs: [Float]) -> [Int] {
        var activations = inputs
        for layer in layers {
            activations = layer.feedForward(activations)
        }
        
        return Network.argmax(activations)
    }
    
    public func train(trainingData: [TrainingSample], epochs: Int, miniBatchSize: Int, eta: Float) {
        var trainingData = trainingData
        
        for e in 0..<epochs {
            //println("Starting epoch \(e+1)/\(epochs)")
            trainingData.shuffle()
            
            let batches = trainingData.count / miniBatchSize
            for b in 0..<batches {
                let batchStart = b * miniBatchSize
                let batchEnd = min(trainingData.count, batchStart + miniBatchSize)
                let batch = trainingData[batchStart..<batchEnd]
                
                updateMiniBatch(batch, eta: eta)
            }
        }
    }
    
    func updateMiniBatch(miniBatch: ArraySlice<TrainingSample>, eta: Float) {
        var trainingNetwork = [LayerTrainer]()
        trainingNetwork.reserveCapacity(layers.count)
        for i in 0..<layers.count { trainingNetwork.append(LayerTrainer(layer: layers[i])) }
        
        for sample in miniBatch {
            self.backPropagate(trainingNetwork, inputs: sample.inputs, outputs: sample.outputs)
        }
        
        let eta_i = eta/Float(miniBatch.count)
        for trainer in trainingNetwork {
            var layer = trainer.layer
            layer.biases = layer.biases - (eta_i*trainer.biasGradients)
            layer.weights = layer.weights - (eta_i*trainer.weightGradients)
        }
    }
    
    func backPropagate(trainingNetwork: [LayerTrainer], inputs: [Float], outputs: [Float]) {
        precondition(trainingNetwork.count == layers.count, "Training network does not match network")
        
        // Forward pass
        var activations = inputs
        for trainer in trainingNetwork {
            activations = trainer.feedForward(activations)
        }
        
        // Compute error
        var delta = trainingNetwork.last!.activations - outputs
        
        // Backward pass
        for i in reverse(0..<trainingNetwork.count) {
            let curTrainer = trainingNetwork[i]
            let curLayer = curTrainer.layer
            let prevLayerActivations = (i > 0) ? trainingNetwork[i - 1].activations : inputs
            
            let backprop = curTrainer.backPropagate(delta, prevLayerActivations: prevLayerActivations)
            delta = backprop * curLayer.activation_Df(curTrainer.z)
        }
    }
    
    static func argmax<T: Comparable>(array: [T]) -> [Int] {
        var maxIndices = [Int]()
        var maxValue: T?
        
        for (index, value) in enumerate(array) {
            if let maxValue_ = maxValue {
                if value >= maxValue_ {
                    maxValue = value
                    maxIndices.append(index)
                }
            } else {
                maxValue = value
                maxIndices.append(index)
            }
        }
        
        return maxIndices
    }
    
    static func runAsync(block: dispatch_block_t) {
        dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), block)
    }
}

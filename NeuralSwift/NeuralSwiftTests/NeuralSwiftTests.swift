//
//  NeuralSwiftTests.swift
//  NeuralSwiftTests
//
//  Created by John Hurliman on 7/11/15.
//  Copyright (c) 2015 John Hurliman. All rights reserved.
//

import XCTest
import NeuralSwift

class ArithmeticTests: XCTestCase {
    func testAdd() {
        let a: [Float] = [1.0, 2.0, 3.0]
        let b: [Float] = [0.0, -5.0, 10.0]
        let c = add(a, b)
        
        XCTAssertEqual(1.0, c[0])
        XCTAssertEqual(-3.0, c[1])
        XCTAssertEqual(13.0, c[2])
    }
    
    func testSubtract() {
        let a: [Float] = [1.0, 2.0, 3.0]
        let b: [Float] = [0.0, -5.0, 10.0]
        let c = sub(a, b)
        
        XCTAssertEqual(1.0, c[0])
        XCTAssertEqual(7.0, c[1])
        XCTAssertEqual(-7.0, c[2])
    }
}

class SigmoidTests: XCTestCase {
    func sigmoid(z: Float) -> Float {
        return 1.0 / (1.0 + exp(-z))
    }
    
    func testSigmoid() {
        // Demonstrating output from the non-vectorized sigmoid function above
        XCTAssertEqual(0.5, sigmoid(0.0))
        XCTAssertEqualWithAccuracy(0.6224593, sigmoid(0.5), 0.000001)
        XCTAssertEqualWithAccuracy(0.7310586, sigmoid(1.0), 0.000001)
        XCTAssertEqualWithAccuracy(0.3775407, sigmoid(-0.5), 0.000001)
        XCTAssertEqualWithAccuracy(0.9999546, sigmoid(10.0), 0.000001)
        
        let sigmoidLayer = SigmoidLayer(layerSize: 1, prevLayerSize: 1)
        let sVec = sigmoidLayer.activation([0.0, 0.5, 1.0, -0.5, 10.0])
        XCTAssertEqual(0.5, sVec[0])
        XCTAssertEqualWithAccuracy(0.6224593, sVec[1], 0.000001)
        XCTAssertEqualWithAccuracy(0.7310586, sVec[2], 0.000001)
        XCTAssertEqualWithAccuracy(0.3775407, sVec[3], 0.000001)
        XCTAssertEqualWithAccuracy(0.9999546, sVec[4], 0.000001)
    }
    
    func testSigmoidFeedForward() {
        let biases: [Float] = [0, 0, 0]
        let weights = Matrix<Float>([ [-1, 1], [-2, 2], [-3, 3] ])
        let layer = SigmoidLayer(biases: biases, weights: weights)
        
        let output = layer.feedForward([1, 1])
        XCTAssertEqual(3, output.count)
        XCTAssertEqual(0.5, output[0])
        XCTAssertEqual(0.5, output[1])
        XCTAssertEqual(0.5, output[2])
    }
    
    func testSigmoidBackProp() {
        let inputs: [Float] = [0]
        let targetActivations: [Float] = [1]
        
        let biases: [Float] = [0]
        let weights = Matrix<Float>([ [0] ])
        var layer = SigmoidLayer(biases: biases, weights: weights)
        
        let initialOutput = layer.feedForward(inputs)
        
        for _ in 1...100 {
            let trainer = LayerTrainer(layer: layer)
            var activations = trainer.feedForward(inputs)
            var delta = trainer.cost_Df(targetActivations)
            var result = trainer.backPropagate(delta, prevLayerActivations: inputs)
            layer.biases = sub(layer.biases, trainer.biasGradients)
            layer.weights = sub(layer.weights, trainer.weightGradients)
            
            let output = layer.feedForward(inputs)
            XCTAssert(output.first! > initialOutput.first!)
        }
        
        let output = layer.feedForward(inputs)
        XCTAssertEqual(targetActivations.first!, round(output.first!))
    }
}

class NeuralSwiftTests: XCTestCase {
    func testXOR() {
        let network = Network(sizes: [2, 3, 1])
        
        let trainingData = [
            TrainingSample(inputs: [1, 1], outputs: [0]),
            TrainingSample(inputs: [0, 1], outputs: [1]),
            TrainingSample(inputs: [1, 0], outputs: [1]),
            TrainingSample(inputs: [0, 0], outputs: [0]),
        ]
        
        measureBlock {
            network.train(trainingData, epochs: 2500, miniBatchSize: 4, eta: 0.95)
        }
        
        let _11 = network.predictValues([1, 1]).first!
        let _01 = network.predictValues([0, 1]).first!
        let _10 = network.predictValues([1, 0]).first!
        let _00 = network.predictValues([0, 0]).first!
        
        XCTAssert(_11 <  0.5, "\(_11) >= 0.5")
        XCTAssert(_01 >= 0.5, "\(_01) < 0.5")
        XCTAssert(_10 >= 0.5, "\(_10) < 0.5")
        XCTAssert(_00 <  0.5, "\(_00) >= 0.5")
    }
}

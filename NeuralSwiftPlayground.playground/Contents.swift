
import NeuralSwift

// Two inputs, three-neuron hidden layer, one output
let network = Network(sizes: [2, 3, 1])

// Truth table for XOR function
let trainingData = [
    TrainingSample(inputs: [1, 1], outputs: [0]),
    TrainingSample(inputs: [0, 1], outputs: [1]),
    TrainingSample(inputs: [1, 0], outputs: [1]),
    TrainingSample(inputs: [0, 0], outputs: [0]),
]

network.train(trainingData, epochs: 5000, miniBatchSize: 4, eta: 0.7)

let _11 = network.predictValues([1, 1]).first!
let _01 = network.predictValues([0, 1]).first!
let _10 = network.predictValues([1, 0]).first!
let _00 = network.predictValues([0, 0]).first!

let predictions = [round(_11), round(_01), round(_10), round(_00)]

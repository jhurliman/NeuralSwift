
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

network.train(trainingData, epochs: 3000, miniBatchSize: 4, eta: 0.8)

let _11 = network.predictValue([1, 1])
let _01 = network.predictValue([0, 1])
let _10 = network.predictValue([1, 0])
let _00 = network.predictValue([0, 0])

let predictions = [round(_11), round(_01), round(_10), round(_00)]

////////////////////////////////////////////////////////////////////////////////

let bundle = NSBundle.mainBundle()
let imagesPath = bundle.pathForResource("MNIST-t10k-images-idx3-ubyte", ofType: nil)!
let labelsPath = bundle.pathForResource("MNIST-t10k-labels-idx1-ubyte", ofType: nil)!

let mnistTrainingData = MNISTLoader(imageFile: imagesPath, labelFile: labelsPath)!

let digitNetwork = Network(sizes: [784, 100, 10])

digitNetwork.train(mnistTrainingData.samples, epochs: 5000, miniBatchSize: 100, eta: 0.7)

let testDigit = mnistTrainingData.samples.first!
let testDigitResult = digitNetwork.predictLabels(testDigit.inputs)

//
//  DecisionTree.swift
//  NeuralSwift
//
//  Created by John Hurliman on 7/28/15.
//  Copyright (c) 2015 John Hurliman. All rights reserved.
//

import Foundation

public class DecisionTree {
    // MARK: - Feature
    
    public enum Feature: Hashable, Printable {
        case Numeric(Double)
        case Category(String)
        
        public var isNumeric: Bool {
            switch (self) {
            case .Numeric(_): return true
            default: return false
            }
        }
        
        public var number: Double {
            switch (self) {
            case .Numeric(let n): return n
            default: return 0
            }
        }
        
        public var category: String {
            switch (self) {
            case .Category(let c): return c
            case .Numeric(let n): return n.description
            }
        }
        
        public var hashValue: Int {
            switch (self) {
            case .Category(let c): return c.hashValue
            case .Numeric(let n): return n.hashValue
            }
        }
        
        public var description: String {
            return self.category
        }
    }
    
    // MARK: - Datum
    
    public struct Datum: Printable {
        public let features: [Feature]
        public let classification: String
        
        public init(features: [Feature], classification: String) {
            self.features = features
            self.classification = classification
        }
        
        public var description: String {
            let featuresStr = ",".join(features.map { $0.description })
            return "[\(featuresStr)] -> \(classification)"
        }
    }
    
    // MARK: - Node
    
    public class Node: Printable {
        let leftChild: Node?
        let rightChild: Node?
        let feature: Feature
        let featureIndex: Int
        let classification: String?
        
        init(feature: Feature, featureIndex: Int, leftChild: Node?, rightChild: Node?) {
            self.leftChild = leftChild
            self.rightChild = rightChild
            self.feature = feature
            self.featureIndex = featureIndex
            self.classification = nil
        }
        
        init(classification: String) {
            self.leftChild = nil
            self.rightChild = nil
            self.feature = .Numeric(0)
            self.featureIndex = -1
            self.classification = classification
        }
        
        public var description: String {
            if let c = classification { return "Classification: \(c)" }
            return "\(feature.description) @ \(featureIndex))"
        }
        
        public func printTree() {
            Node.printTree(self)
        }
        
        static func printTree(node: Node, _ indent: String = "") {
            if let classification = node.classification {
                println(classification)
                return
            }
            
            println("\(node.featureIndex):\(node.feature)")
            
            print("\(indent)L->")
            printTree(node.leftChild!, indent + "  ")
            print("\(indent)R->")
            printTree(node.rightChild!, indent + "  ")
        }
    }
    
    // MARK: - DecisionTree
    
    public let rootNode: Node
    
    public init?(data: [Datum], maxFeatures: Int? = nil) {
        if data.count == 0 {
            rootNode = Node(classification: "")
            return nil
        }
        
        var indexes = [Int](0..<data.first!.features.count)
        rootNode = DecisionTree.createNode(data, featureIndexes: indexes, maxFeatures: maxFeatures)
    }
    
    public func classify(features: [Feature]) -> String? {
        var curNode = rootNode
        
        while true {
            if let classification = curNode.classification { return classification }
            
            let cut = curNode.feature
            if cut.isNumeric {
                let value = features[curNode.featureIndex].number
                curNode = (value >= cut.number) ? curNode.leftChild! : curNode.rightChild!
            } else {
                let category = features[curNode.featureIndex].category
                curNode = (curNode.feature.category == category) ? curNode.leftChild! : curNode.rightChild!
            }
        }
    }
    
    // MARK: - Tree Construction
    
    static func createNode(data: [Datum], var featureIndexes: [Int], maxFeatures: Int?) -> Node {
        if featureIndexes.count == 0 {
            return Node(classification: mostCommonClassification(data))
        }
        
        // Identify best feature via max gain
        let (bestFeatureIndex, bestFeature) = maxGainFeature(data, featureIndexes: featureIndexes, maxFeatures: maxFeatures)
        if bestFeatureIndex == -1 {
            return Node(classification: mostCommonClassification(data))
        }
        
        // Split the data based on the selected optimal feature
        let (left, right) = split(data, featureIndex: bestFeatureIndex, cut: bestFeature)
        
        let leftNode = createNode(left, featureIndexes: featureIndexes, maxFeatures: maxFeatures)
        let rightNode = createNode(right, featureIndexes: featureIndexes, maxFeatures: maxFeatures)
        return Node(feature: bestFeature, featureIndex: bestFeatureIndex, leftChild: leftNode, rightChild: rightNode)
    }
    
    static func mostCommonClassification(data: [Datum]) -> String {
        precondition(data.count > 0)
        var counts = [String: Int]()
        
        for datum in data {
            counts[datum.classification] = (counts[datum.classification] ?? 0) + 1
        }
        
        // Sort dictionary entries by value (occurrences) in descending order
        let sorted = Array(counts).sorted { $0.1 > $1.1 }
        return sorted.first!.0
    }
    
    // MARK: - Gain
    
    static func maxGainFeature(data: [Datum], var featureIndexes: [Int], maxFeatures: Int?) -> (Int, Feature) {
        precondition(featureIndexes.count > 0, "No features indexes passed to maxGainFeature")
        
        let dataEntropy = entropy(data)
        
        if var maxFeatures = maxFeatures {
            // Only consider a random subset of features
            maxFeatures = Swift.min(featureIndexes.count, maxFeatures)
            featureIndexes = sampleWithoutReplacement(featureIndexes, count: maxFeatures)
        }
        
        var bestGain = 0.0
        var bestIndex = -1
        var bestCut = Feature.Numeric(0)
        
        for index in featureIndexes {
            let (curGain, curCut) = gain(data, dataEntropy: dataEntropy, featureIndex: index)
            if curGain > bestGain {
                bestGain = curGain
                bestCut = curCut
                bestIndex = index
            }
        }
        
        return (bestIndex, bestCut)
    }
    
    static func gain(data: [Datum], dataEntropy: Double, featureIndex: Int) -> (Double, Feature) {
        let featureValues = unique(data.map { $0.features[featureIndex] })
        var best = 0.0
        var bestCut = featureValues.first!
        
        for cut in featureValues {
            let (left, right) = split(data, featureIndex: featureIndex, cut: cut)
            if left.count > 0 && right.count > 0 {
                let curGain = dataEntropy - conditionalEntropy(left: left, right: right)
                if curGain > best {
                    best = curGain
                    bestCut = cut
                }
            }
        }
        
        return (best, bestCut)
    }
    
    static func sampleWithoutReplacement(array: [Int], count: Int) -> [Int] {
        precondition(count <= array.count)
        
        var subset = array
        for _ in 0..<(array.count - count) {
            let index = Int(arc4random_uniform(UInt32(subset.count)))
            subset.removeAtIndex(index)
        }
        
        return subset
    }
    
    // MARK: - Entropy
    
    static func unique<T: Hashable>(data: [T]) -> [T] {
        var seen = [T: Bool]()
        return data.filter { seen.updateValue(true, forKey: $0) == nil }
    }
    
    static func uniqueClassifications(data: [Datum]) -> [String] {
        var seen = [String: Bool]()
        return data
            .filter { seen.updateValue(true, forKey: $0.classification) == nil }
            .map { $0.classification }
    }
    
    static func split(data: [Datum], featureIndex: Int, cut: Feature) -> ([Datum], [Datum]) {
        var left = [Datum]()
        var right = [Datum]()
        
        if cut.isNumeric {
            for datum in data {
                (datum.features[featureIndex].number >= cut.number) ? left.append(datum) : right.append(datum)
            }
        } else {
            for datum in data {
                (datum.features[featureIndex].category == cut.category) ? left.append(datum) : right.append(datum)
            }
        }
        
        return (left, right)
    }
    
    static func probability(classification: String, _ data: [Datum]) -> Double {
        var count = 0
        for datum in data {
            if datum.classification == classification { count++ }
        }
        return Double(count) / Double(data.count)
    }
    
    static func entropy(data: [Datum]) -> Double {
        let classifications = uniqueClassifications(data)
        var sum = 0.0
        for classification in classifications {
            let p = probability(classification, data)
            sum += -p * log2(p)
        }
        return sum
    }
    
    static func conditionalEntropy(#left: [Datum], right: [Datum]) -> Double {
        let p = Double(left.count)/Double(left.count + right.count)
        return p*entropy(left) + (1.0-p)*entropy(right)
    }
}

public func ==(lhs: DecisionTree.Feature, rhs: DecisionTree.Feature) -> Bool {
    if lhs.isNumeric != rhs.isNumeric { return false }
    return lhs.isNumeric ? lhs.number == rhs.number : lhs.category == rhs.category
}

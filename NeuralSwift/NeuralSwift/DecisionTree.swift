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
    
    public struct IndexedFeature: Hashable {
        let feature: Feature
        let index: Int
        
        public var hashValue: Int { return index }
    }
    
    // MARK: - Datum
    
    public struct Datum {
        public let features: [Feature]
        public let classification: String
        
        public init(features: [Feature], classification: String) {
            self.features = features
            self.classification = classification
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
    }
    
    // MARK: - DecisionTree
    
    public let rootNode: Node
    
    public init?(data: [Datum]) {
        if data.count == 0 {
            rootNode = Node(classification: "")
            return nil
        }
        
        rootNode = DecisionTree.createNode(data, features: data.first!.features)
    }
    
    public func classify(features: [Feature]) -> String? {
        var curNode = rootNode
        
        while true {
            if let classification = curNode.classification { return classification }
            
            let cut = curNode.feature
            if cut.isNumeric {
                let value = features[curNode.featureIndex].number
                curNode = (value <= cut.number) ? curNode.leftChild! : curNode.rightChild!
            } else {
                let category = features[curNode.featureIndex].category
                curNode = (curNode.feature.category == category) ? curNode.leftChild! : curNode.rightChild!
            }
        }
    }
    
    static func createNode(data: [Datum], var features: [Feature]) -> Node {
        if features.count == 0 {
            return Node(classification: mostCommonClassification(data))
        }
        
        // Identify best feature via max gain, remove from remaining features
        let (bestFeatureIndex, bestFeature) = maxGainFeature(data, features: features)
        features.removeAtIndex(bestFeatureIndex)
        
        // Split the data based on the selected optimal feature
        let (left, right) = split(data, featureIndex: bestFeatureIndex, cut: bestFeature)
        
        if right.count == 0 {
            return Node(classification: mostCommonClassification(left))
        }
        
        let leftNode = createNode(left, features: features)
        let rightNode = createNode(right, features: features)
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
    
    static func maxGainFeature(data: [Datum], features: [Feature]) -> (Int, Feature) {
        precondition(features.count > 0, "Empty features array passed to maxGainFeature")
        
        var best = -Double.infinity
        var bestIndex = -1
        var bestCut = features.first!
        
        for i in 0..<features.count {
            let (curGain, curCut) = gain(data, featureIndex: i)
            if curGain > best {
                best = curGain
                bestCut = curCut
                bestIndex = i
            }
        }
        
        return (bestIndex, bestCut)
    }
    
    static func gain(data: [Datum], featureIndex: Int) -> (Double, Feature) {
        let dataEntropy = entropy(data)
        let features = unique(data.map { $0.features[featureIndex] })
        var best = -Double.infinity
        var bestCut = features.first!
        
        for cut in features {
            let curGain = dataEntropy - conditionalEntropy(data, featureIndex: featureIndex, cut: cut)
            if curGain > best {
                best = curGain
                bestCut = cut
            }
        }
        
        return (best, bestCut)
    }
    
    // MARK: - Entropy
    
    static func unique<T: Hashable>(data: [T]) -> [T] {
        var seen = [T: Bool]()
        return data.filter { seen.updateValue(true, forKey: $0) == nil }
    }
    
    static func unique(data: [Datum]) -> [Datum] {
        var seen = [String: Bool]()
        return data.filter { seen.updateValue(true, forKey: $0.classification) == nil }
    }
    
    static func uniqueFeatures(data: [Datum], featureIndex: Int) -> [Feature] {
        let features = data.map { $0.features[featureIndex] }
        return unique(features)
    }
    
    static func split(data: [Datum], featureIndex: Int, cut: Feature) -> ([Datum], [Datum]) {
        var left = [Datum]()
        var right = [Datum]()
        
        if cut.isNumeric {
            for datum in data {
                (datum.features[featureIndex].number <= cut.number) ? left.append(datum) : right.append(datum)
            }
        } else {
            for datum in data {
                (datum.features[featureIndex].category == cut.category) ? left.append(datum) : right.append(datum)
            }
        }
        
        return (left, right)
    }
    
    static func probability(datum: Datum, _ data: [Datum]) -> Double {
        var count = 0
        for curData in data {
            if curData.classification == datum.classification { count++ }
        }
        return Double(count) / Double(data.count)
    }
    
    static func entropy(data: [Datum]) -> Double {
        let uniqueData = unique(data)
        var sum = 0.0
        for datum in uniqueData {
            let p = probability(datum, data)
            sum += -p * log2(p)
        }
        return sum
    }
    
    static func conditionalEntropy(data: [Datum], featureIndex: Int, cut: Feature) -> Double {
        let (left, right) = split(data, featureIndex: featureIndex, cut: cut)
        let p = Double(left.count)/Double(data.count)
        return p*entropy(left) + (1.0-p)*entropy(right)
    }
}

public func ==(lhs: DecisionTree.Feature, rhs: DecisionTree.Feature) -> Bool {
    if lhs.isNumeric != rhs.isNumeric { return false }
    return lhs.isNumeric ? lhs.number == rhs.number : lhs.category == rhs.category
}

public func ==(lhs: DecisionTree.IndexedFeature, rhs: DecisionTree.IndexedFeature) -> Bool {
    return lhs.index == rhs.index
}

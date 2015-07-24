//
//  MNISTLoader.swift
//  NeuralSwift
//
//  Created by John Hurliman on 7/17/15.
//  Copyright (c) 2015 John Hurliman. All rights reserved.
//

import Foundation

public class MNISTLoader {
    public let samples: [TrainingSample]
    
    public init?(imageFile: String, labelFile: String) {
        // Image loading
        ////////////////////////////////////////////////////////////////////////
        var images = [NSData]()
        var labels = [Int]()
        
        if let imageData = NSData(contentsOfFile: imageFile) {
            let reader = BinaryDataScanner(data: imageData, littleEndian: false, encoding: NSUTF8StringEncoding)
            
            // Magic number
            let magicNum = reader.read32()
            if 2051 != magicNum { self.samples = []; return nil }
            
            let count = Int(reader.read32() ?? 0)
            let rows = Int(reader.read32() ?? 0)
            let columns = Int(reader.read32() ?? 0)
            
            images.reserveCapacity(count)
            
            // Read all of the images
            for i in 0..<count {
                if let imageBytes = reader.readData(rows * columns) {
                    images.append(imageBytes)
                } else {
                    break
                }
            }
        } else {
            self.samples = []; return nil
        }
        
        // Label loading
        ////////////////////////////////////////////////////////////////////////
        if let labelData = NSData(contentsOfFile: labelFile) {
            let reader = BinaryDataScanner(data: labelData, littleEndian: false, encoding: NSUTF8StringEncoding)
            
            // Magic number
            let magicNum = reader.read32()
            if 2049 != magicNum { self.samples = []; return nil }
            
            let count = Int(reader.read32() ?? 0)
            
            labels.reserveCapacity(count)
            
            for i in 0..<count {
                if let labelByte = reader.readByte() {
                    labels.append(Int(labelByte))
                } else {
                    break
                }
            }
        } else {
            self.samples = []; return nil
        }
        
        if images.count != labels.count {
            self.samples = []; return nil
        }
        
        var samples = [TrainingSample]()
        samples.reserveCapacity(images.count)
        for pair in Zip2(images, labels) {
            let imageBytes = pair.0
            let label = pair.1
            
            var pixels = [Float]()
            pixels.reserveCapacity(imageBytes.length)
            
            var current = UnsafePointer<UInt8>(imageBytes.bytes)
            for _ in 0..<imageBytes.length {
                let value = Float(current.memory) / 255.0
                pixels.append(value)
                current = current.successor()
            }
            
            let output = MNISTLoader.makeOutput(label)
            samples.append(TrainingSample(inputs: pixels, outputs: output))
        }
        
        self.samples = samples
    }
    
    static func makeOutput(label: Int) -> [Float] {
        var output = [Float](count: 10, repeatedValue: 0.0)
        output[label] = 1.0
        return output
    }
}

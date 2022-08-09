//
//  ViewControllerStart.swift
//  SceneDepthPointCloud
//
//  Created by Monali Palhal on 26/07/22.
//  Copyright © 2022 Apple. All rights reserved.
//


import UIKit
import SceneKit
import RealityKit
import ARKit
import AVFoundation
var leftshld1 : simd_float3=[0,0,0]
var rightshld1 : simd_float3=[0,0,0]
var leftshld2 : simd_float3=[0,0,0]
var rightshld2 : simd_float3=[0,0,0]
class ViewController: UIViewController, ARSCNViewDelegate , ARSessionDelegate {
    
    @IBOutlet var sceneView: ARSCNView!
    var dist=0
    
    var sx:Float=0
    var sy:Float=0
    var sz:Float=0
    var headJointPosx : Float = 0.0
    var headJointPosy : Float = 0.0
    var headJointPosz : Float = 0.0
    var arrayofallcenter = [simd_float3]()
    var center : simd_float3 = [0,0,0]
    var leftshoulder : simd_float3 = [0,0,0]
    var rightshoulder : simd_float3 = [0,0,0]
    var prevLeft : simd_float3 = [0,0,0]
    var prevRight : simd_float3 = [0,0,0]
    var f : Int = 0
    var player: AVAudioPlayer?
    var event = false
    var eventfront = true
    var eventleft = false
    var trackingstart = false
    override func viewDidLoad() {
        super.viewDidLoad()
        
        
        // Set the view's delegate
        sceneView.delegate = self
        sceneView.session.delegate = self
        
//
//        // Show statistics such as fps and timing information
//        sceneView.showsStatistics = true
//
//        // Create a new scene
//        let scene = SCNScene(named: "art.scnassets/ship.scn")!
//
//        // Set the scene to the view
//        sceneView.scene = scene
    }
    
    
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        
//        let configuration = ARWorldTrackingConfiguration()
        let configuration = ARBodyTrackingConfiguration()
//        configuration.sceneReconstruction = .mesh
        configuration.automaticSkeletonScaleEstimationEnabled = true
       
        
//        configuration.automaticSkeletonScaleEstimationEnabled = true
    

        // Run the view's session/Users/admin/Desktop/उंची मापक/उंची मापक/BodyPoint.swift
            self.sceneView.session.run(configuration , options: [])
//        DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
//            self.playSound(soundname: "started_detection")
//        }
        DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
            self.trackingstart = true
        }
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        // Pause the view's session
        sceneView.session.pause()
    }
    
    @IBAction func showPointCloud(_ sender: Any) {
        let mainStoryBoard = UIStoryboard(name: "Main", bundle: nil)
        let secondViewController = mainStoryBoard.instantiateViewController(withIdentifier: "ShowPointCloudViewController") as! ShowPointCloudViewController
           
        secondViewController.filename = Helper().retrievePathnameFromKeychain() ?? ""
        self.navigationController?.pushViewController(secondViewController, animated: true)
    }
    @IBAction func startScanning(_ sender: Any) {
        var sum : simd_float3 = [0,0,0]
        var avg : simd_float3 = [0,0,0]
        for i in arrayofallcenter {
            sum = sum + i
        }
        var count : Float = Float(arrayofallcenter.count)
        avg.x = sum.x / count
        avg.y = sum.y / count
        avg.z = sum.z / count
        print("avg of center\(avg)")
       let mainStoryBoard = UIStoryboard(name: "Main", bundle: nil)
      let secondViewController = mainStoryBoard.instantiateViewController(withIdentifier: "scanningViewController") as! scanningViewController
        if count == 0 {
            avg = [0, 0, 0]
        }
        
       
//        secondViewController.centerofbody = center
        
            
      self.navigationController?.pushViewController(secondViewController, animated: true)
    }
    func playSound(soundname : String) {
        guard let url = Bundle.main.url(forResource: soundname, withExtension: "mp3") else { return }

        do {
            try AVAudioSession.sharedInstance().setCategory(.playback, mode: .default)
            try AVAudioSession.sharedInstance().setActive(true)

            /* The following line is required for the player to work on iOS 11. Change the file type accordingly*/
            player = try AVAudioPlayer(contentsOf: url, fileTypeHint: AVFileType.mp3.rawValue)

            /* iOS 10 and earlier require the following line:
            player = try AVAudioPlayer(contentsOf: url, fileTypeHint: AVFileTypeMPEGLayer3) */

            guard let player = player else { return }

            player.play()

        } catch let error {
            print(error.localizedDescription)
        }
    }
    
    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
        
        print("started")
        for anchor in anchors {
            if let bodyAncor = anchor as? ARBodyAnchor {
//                print("body updated ")


                let skeleton = bodyAncor.skeleton
                
                let a = AnchorEntity(world: bodyAncor.transform)
//                for (i, joint) in skeleton.definition.jointNames.enumerated() {
////                    print("skeletop join ", i, joint)
//                }
                // zero index for root
//                print("head point \(skeleton.jointModelTransforms[0].columns.3)")
                
                headJointPosy = skeleton.jointModelTransforms[0].columns.3.y
                let root = skeleton.jointModelTransforms[0].columns.3.y
                let lSh = skeleton.jointModelTransforms[20].columns.3
                let rSh = skeleton.jointModelTransforms[64].columns.3
                headJointPosx = skeleton.jointModelTransforms[0].columns.3.x
                headJointPosz = skeleton.jointModelTransforms[0].columns.3.z
//                    print(headJointPosy,"X position")
//                    print(headJointPosx,"Y Position")
//                print("left sholder",lSh)
//                print("right sholder",rSh)
//                print("global left sholder",lSh + anchor.transform.columns.3)
//                print("global right sholder",rSh + anchor.transform.columns.3)
               //  arrayofallcenter.append([skeleton.jointModelTransforms[0].columns.3.x, skeleton.jointModelTransforms[0].columns.3.y,skeleton.jointModelTransforms[0].columns.3.z])
                self.center = [bodyAncor.transform.columns.3.x, bodyAncor.transform.columns.3.y,bodyAncor.transform.columns.3.z]
                let leftarm = lSh + anchor.transform.columns.3
                let rightarm = rSh + anchor.transform.columns.3
                if eventfront {
                    eventfront = false
                    DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
                    
                    leftshld1 = [leftarm.x, leftarm.y, leftarm.z]
                    rightshld1 = [rightarm.x, rightarm.y, rightarm.z]
                    print("turn")
                    print(leftshld1)
                    print(rightshld1)
                    print("-----------------------------------------------")
                
                    }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 6) {
                       
                        self.eventleft = true
                    }
                }
                if eventleft {
                    self.eventleft = false
                    DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
                    leftshld2 = [leftarm.x, leftarm.y, leftarm.z]
                    rightshld2 = [rightarm.x, rightarm.y, rightarm.z]
                    
//                    leftshld2=[ (leftshld2.x * cos(90 * Float(Double.pi) / 180)) + (leftshld2.z * sin(90 * Float(Double.pi) / 180)) , leftshld2.y, (leftshld2.z * cos(90 * Float(Double.pi) / 180)) - (leftshld2.x * sin(90 * Float(Double.pi) / 180))]
                    print("***********************************************")

                    print(leftshld2)
                    print(rightshld2)
                    
                    print("-----------------------------------------------")

                
                    }
                    DispatchQueue.main.asyncAfter(deadline: .now() + 6) {
                       
                       
                        self.event = true
                    }

                   
                }
               
                if event {
                    sx=Float(leftshld1.x-leftshld2.x)
                    sy=Float(leftshld1.y-leftshld2.y)
                    sz=Float(leftshld1.z-leftshld2.z)
                    self.event = false
                    //dist = Double(distance(leftshld1,leftshld2))
                    //print(dist)
                DispatchQueue.main.async {
                   self.playSound(soundname: "human_detected")
                    DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
                        self.sceneView.session.pause()
                        let mainStoryBoard = UIStoryboard(name: "Main", bundle: nil)
                        let secondViewController = mainStoryBoard.instantiateViewController(withIdentifier: "scanningViewController") as! scanningViewController

                        secondViewController.centerofbody = self.center
                        secondViewController.sx=self.sx
                        secondViewController.sy=self.sy
                        secondViewController.sz=self.sz

//                      secondViewController.leftshoulder = leftarm
//                      secondViewController.rightshoulder = rightarm


                   self.navigationController?.pushViewController(secondViewController, animated: true)

                    }
                }
//
//}
////
////
//
//        }
    }
//        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
//            self.playSound(soundname: "started_detection")
        if trackingstart {
        for anchor in anchors {

            guard let bodyAnchor = anchor as? ARBodyAnchor else { continue }

           // print("body anchor \(bodyAnchor.transform.columns.3)")

            // Update the position of the character anchor's position.

           // let bodydistance = abs(bodyAnchor.transform.columns.3.z)+pos

           // var distanceBody = Measurement(value: Double(bodydistance), unit: UnitLength.meters).converted(to: .feet).value
            
           if self.event {
                self.event = false
               DispatchQueue.main.async {
               self.playSound(soundname: "human_detected")
               DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
                    self.sceneView.session.pause()
                    let mainStoryBoard = UIStoryboard(name: "Main", bundle: nil)
                    let secondViewController = mainStoryBoard.instantiateViewController(withIdentifier: "scanningViewController") as! scanningViewController

                    secondViewController.centerofbody = self.center


                    self.navigationController?.pushViewController(secondViewController, animated: true)

             }
//            }
//            }
        }
    }
        
    }
    
    func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
/*
        let speare = SCNSphere(radius: 0.05)



        let headNode = SCNNode()
//        let toesNode = SCNNode()

//                    headNode.position = SCNVector3(0 ,  headJointPosy, 0)
//        headNode.geometry = speare

        let materials = SCNMaterial()
        materials.diffuse.contents = UIColor.red

        speare.materials = [materials]

        headNode.position = SCNVector3(x: headJointPosx, y: headJointPosy    , z: 0)
        headNode.geometry = speare
        node.addChildNode(headNode) */
        
    }
//
         }
        
//                    toesNode.position = SCNVector3(to)
        
        
        
        
        
        
    
}
}
}
}

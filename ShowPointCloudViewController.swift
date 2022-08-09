//
//  ShowPointCloudViewController.swift
//  SceneDepthPointCloud
//
//  Created by Monali Palhal on 09/07/22.
//  Copyright © 2022 Apple. All rights reserved.
//

import UIKit
import SceneKit
import ARKit
import RealityKit
import AVFoundation

// MARK: - App State Management

enum AppState: Int16 {
  case DetectSurface  // Scan surface (Plane Detection On)
  case PointAtSurface // Point at surface to see focus point (Plane Detection Off)
  case TapToStart     // Focus point visible on surface, tap to start
  case Started
}



class ShowPointCloudViewController: UIViewController, ARSCNViewDelegate {

    // MARK: - IB Outlets
    @IBOutlet weak var ScenekitView: SCNView!
    @IBOutlet var sceneView: ARSCNView!
    @IBOutlet weak var statusLabel: UILabel!
    @IBOutlet weak var resetButton: UIButton!
    var player: AVAudioPlayer?
    var currentNode : SCNNode?
    var allpoint = [simd_float3]()
    var filename = ""
    var scanshow = false 
    var currentAngleY: Float = 0.0
    var currentposiotion : CGFloat = 0
    var istouchtube = false
    var istouchavtar = false
    var tube : SCNNode = SCNNode(geometry: SCNTube(innerRadius: 0.06, outerRadius: 0.1, height: 5))
    var ship : SCNNode = SCNNode(geometry: SCNTube(innerRadius: 0.06, outerRadius: 0.1, height: 5))
    
    // MARK: - IB Actions
    @IBAction func resetButtonPressed(_ sender: Any) {
     //  self.resetApp()
        
    }
    
    @IBAction func tapGestureHandler(_ sender: Any) {
      guard appState == .TapToStart else { return }
      self.arPortNode.isHidden = false
      self.focusNode.isHidden = true
      self.arPortNode.position = self.focusNode.position
      appState = .Started
    }
    
    // MARK: - Properties
    var trackingStatus: String = ""
    var statusMessage: String = ""
    var appState: AppState = .DetectSurface
    var focusPoint:CGPoint!
    var focusNode: SCNNode!
    var arPortNode: SCNNode!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = UIColor.white //UIColor(named: "292A33")
        

        // Do any additional setup after loading the view.
       /* let scene = SCNScene(named: "airplane2.ply")
        ScenekitView.scene = scene
        let cameraNode = SCNNode()
        cameraNode.camera = SCNCamera()
        // 3: Place camera
        cameraNode.position = SCNVector3(x: 0, y: 0, z: 20)
        // 4: Set camera on scene
        scene?.rootNode.addChildNode(cameraNode)
        // 5: Adding light to scene
        let lightNode = SCNNode()
        lightNode.light = SCNLight()
        lightNode.light?.type = .omni
        lightNode.position = SCNVector3(x: 0, y: 10, z: 35)
        scene?.rootNode.addChildNode(lightNode)

        // 6: Creating and adding ambien light to scene
        let ambientLightNode = SCNNode()
        ambientLightNode.light = SCNLight()
        ambientLightNode.light?.type = .ambient
        ambientLightNode.light?.color = UIColor.darkGray
        scene?.rootNode.addChildNode(ambientLightNode)
        // Allow user to manipulate camera
        ScenekitView.allowsCameraControl = true

        // Show FPS logs and timming
        // sceneView.showsStatistics = true

        // Set background color
        ScenekitView.backgroundColor = UIColor.white

        // Allow user translate image
        ScenekitView.cameraControlConfiguration.allowsTranslation = false

        // Set scene settings
        ScenekitView.scene = scene */
        
//        if !scanshow {
//            playSound(soundname: "scan_complete")
//        }
        updateStatus()
       
        for swipe in [UISwipeGestureRecognizer.Direction.up,
                             UISwipeGestureRecognizer.Direction.down] {

                   let sw = UISwipeGestureRecognizer(target: self,
                                                     action: #selector(self.dragUpAndDown))
                   sw.direction = swipe
                   self.ScenekitView.addGestureRecognizer(sw)
            print("tube get touched increament")
               }
        let panRecognizer = UIPanGestureRecognizer(target: self,
                                                   action: #selector(self.rotateObject))
        ScenekitView.addGestureRecognizer(panRecognizer)
       
    }
    
    @IBAction func rescnButton(_ sender: Any) {
        let mainStoryBoard = UIStoryboard(name: "Main", bundle: nil)
        let secondViewController = mainStoryBoard.instantiateViewController(withIdentifier: "StartViewController") as! StartViewController
            
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
    /*
    // MARK: - Navigation

    // In a storyboard-based application, you will often want to do a little preparation before navigation
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        // Get the new view controller using segue.destination.
        // Pass the selected object to the new view controller.
    }
    */

}
extension ShowPointCloudViewController {
  
  func initScene() {
    let scene = SCNScene()
    sceneView.scene = scene
    sceneView.delegate = self
    //sceneView.showsStatistics = true
    sceneView.debugOptions = [
      //ARSCNDebugOptions.showFeaturePoints,
      //ARSCNDebugOptions.showWorldOrigin,
      //SCNDebugOptions.showBoundingBoxes,
      //SCNDebugOptions.showWireframe
    ]
    
    let arPortScene = SCNScene(named: "art.scnassets/Scenes/ARPortScene.scn")!
    arPortNode = arPortScene.rootNode.childNode(
      withName: "ARPort", recursively: false)!
    arPortNode.isHidden = true
    sceneView.scene.rootNode.addChildNode(arPortNode)
  }
  
  func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
    DispatchQueue.main.async {
        self.updateStatus()
        print("called")
    }
  }
    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
      DispatchQueue.main.async {
        if let touchLocation = touches.first?.location(in: self.ScenekitView) {
            
          if let hit = self.ScenekitView.hitTest(touchLocation, options: nil).first {
            if hit.node.name == "cloud" {
              print("avatar get touched1")
                self.istouchtube = false
                self.istouchavtar = true
                self.currentNode = hit.node.childNode(withName: "cloud", recursively: false)
            }else if hit.node.name == "tube" {
                //   for swipe in [UISwipeGestureRecognizer.Direction.up,
//                                       UISwipeGestureRecognizer.Direction.down] {
//
//                             let sw = UISwipeGestureRecognizer(target: self,
//                                                               action: #selector(self.dragUpAndDown))
//                             sw.direction = swipe
//                             self.ScenekitView.addGestureRecognizer(sw)
//                      print("tube get touched increament")
//                         }
                self.istouchtube = true
                self.istouchavtar = false
               print("tube get touched1")
                  // self.currentNode = hit.node.childNode(withName: "tube", recursively: false)
            } else {
                self.istouchtube = false
                self.istouchavtar = false
                print("touch outside")
            }
            
          }
        }
      }
        let cameraNode: SCNNode? = ScenekitView.pointOfView
            print(cameraNode?.rotation ?? "There is no camera set to the view")
    }
    
       @objc func dragUpAndDown(recognizer: UISwipeGestureRecognizer) {
           if recognizer.direction == .up {
               tube.position.y += 0.01
               print("increment")
           }
           if recognizer.direction == .down {
               tube.position.y -= 0.01
               print("decrement")
           }
       }
    @objc func rotateObject(_ gesture: UIPanGestureRecognizer) {
print("function call")
         //   guard let nodeToRotate = ship else { return }

        let point = gesture.location(in: ScenekitView)
        if let hit = self.ScenekitView.hitTest(point, options: nil).first {
          if hit.node.name == "cloud" {
         print("avatar get touched")
              self.currentNode = hit.node.childNode(withName: "cloud", recursively: false)
          }
            if hit.node.name == "tube" {
              //   for swipe in [UISwipeGestureRecognizer.Direction.up,
//                                       UISwipeGestureRecognizer.Direction.down] {
//
//                             let sw = UISwipeGestureRecognizer(target: self,
//                                                               action: #selector(self.dragUpAndDown))
//                             sw.direction = swipe
//                             self.ScenekitView.addGestureRecognizer(sw)
//                      print("tube get touched increament")
//                         }
             print("tube get touched")
                // self.currentNode = hit.node.childNode(withName: "tube", recursively: false)
            }
          
        }
        var newAngleY : Float = 0
        let translation = gesture.translation(in: gesture.view!)
       if istouchavtar {
          
            newAngleY = (Float)(translation.x)*(Float)(Double.pi)/180.0
           newAngleY += currentAngleY
        ship.eulerAngles.y = newAngleY

          

           print(ship.eulerAngles)
        }
        if(gesture.state == .ended) { currentAngleY = newAngleY }
        var newposition = translation.y
        if istouchtube {
        
           //  newposition += 0.01
        print("newposition\(newposition)")
       
        tube.position.y = Float(currentposiotion)
        print(tube.position)
        }
        if(gesture.state == .ended) { if newposition > 0 {
            currentposiotion = currentposiotion  - 0.1
        } else { currentposiotion = currentposiotion  + 0.1 }}
   }
    @objc internal  func  updateStatus(){
        print("call update status")
      
        do {
        
       
        
        let paths = NSSearchPathForDirectoriesInDomains(.documentDirectory, .userDomainMask, true)
        let documentsDirectory = paths[0]
        let fileManager = FileManager.default
           
        // var pathname =  Helper().retrievePathnameFromKeychain() ?? "ply_sancfile.scn"
            var pathname = "ply_color_back\(filename).scn"
        let completepath = Helper().retrievePathnameFromKeychain() ?? "/var/mobile/Containers/Data/Application/36569A96-F982-4023-A826-D2288FE9CC9B/Documents/"
        var  path = completepath.deletingPrefix("/var/mobile/Containers/Data/Application/36569A96-F982-4023-A826-D2288FE9CC9B/Documents/")
       
        let imagePAth = (documentsDirectory as NSString).appendingPathComponent(completepath)
            Logger.shared().log(message: "name of file \(pathname) image path \(imagePAth)")
        if fileManager.fileExists(atPath: imagePAth){
           
            print("imagePAth:\(imagePAth)")
            let myURL = URL(fileURLWithPath : imagePAth)
            let scene = try SCNScene(url: myURL as URL, options: nil)
             //setup the camera
//        let scene = SCNScene(named: "ply_color.scn")!
            let camera = SCNCamera();
            camera.usesOrthographicProjection = true
            camera.orthographicScale = 1
            camera.zNear = 0
            camera.zFar = 100;
            // create and add a camera to the scene
            let cameraNode = SCNNode()
            cameraNode.camera = camera
            scene.rootNode.addChildNode(cameraNode)
        cameraNode.position = SCNVector3(x: 0, y: 0, z: 0.5)
        // cameraNode.eulerAngles = SCNVector3Make(180, 0, 180)
            // place the camera
            //cameraNode.position = SCNVector3(x: 0, y: 0, z: 5)
            
//        let box = SCNBox(width: 0.2, height: 0.2, length: 10, chamferRadius: 5);
//                    let boxNode = SCNNode(geometry: box)
//                    scene.rootNode.addChildNode(boxNode)
        //boxNode.position = SCNVector3(x: 0, y: 0, z: -5)
//        let rod = SCNTube(innerRadius: 0.06, outerRadius: 0.1, height: 5)
//        let rodnode = SCNNode(geometry: rod)
//        scene.rootNode.addChildNode(rodnode)
//        rodnode.position = SCNVector3(x: 0, y: 0, z: -0.3)
//        rodnode.eulerAngles = SCNVector3Make(90, 0, 20);
            //        // retrieve the ship node
             ship = scene.rootNode.childNode(withName: "cloud", recursively: true)!
           // scene.rootNode.addChildNode(ship)
             ship.position = SCNVector3(x: 0, y: 0, z: 0)
         ship.eulerAngles = SCNVector3Make(0, 0, 0)
         //tube = scene.rootNode.childNode(withName: "tube", recursively: true)!
        let tubes = SCNTube(innerRadius: 0.06, outerRadius: 0.1, height: 5)
        let tubesnode = SCNNode(geometry: tubes)
       // scene.rootNode.addChildNode(tubesnode)
        tubesnode.position = SCNVector3(x: 0, y: 0, z: 0.23)
        // tubesnode.c
           //  newAngleY * ( 180 / Double.pi)
        tubesnode.eulerAngles = SCNVector3Make(0, 0, 0)
        print("position of tube node \(tubesnode.position)")
        //tube.rotation = SCNVector4(x: 0, y: 1, z: 1, w: 45)
            // put a constraint on the camera1
            let constraint = SCNLookAtConstraint(target: ship)
            
            cameraNode.constraints = [constraint]
            //        let targetNode = SCNLookAtConstraint(target: ship);
            //        //targetNode.gimbalLockEnabled = YES;
            //        cameraNode.constraints = [targetNode];
            
            // create and add a light to the scene
            let lightNode = SCNNode()
            lightNode.light = SCNLight()
            lightNode.light!.type = .omni
            lightNode.position = SCNVector3(x: 0, y: 0, z: 0)
            scene.rootNode.addChildNode(lightNode)
            
            // create and add an ambient light to the scene
            let ambientLightNode = SCNNode()
            ambientLightNode.light = SCNLight()
            ambientLightNode.light!.type = .ambient
            ambientLightNode.light!.color = UIColor.darkGray
            ambientLightNode.position = SCNVector3(x: 0, y: 0, z: 0)
            scene.rootNode.addChildNode(ambientLightNode)
            
            
            
            
            // animate the 3d object
            //ship.runAction(SCNAction.repeatForever(SCNAction.rotateBy(x: 0, y: 2, z: 0, duration: 1)))
            
            // retrieve the SCNView
            //let scnView = SCNView()
            
            
            
            // put a constraint on the camera
            //        let cameraOrbit = SCNNode()
            //        cameraOrbit.addChildNode(cameraNode)
            //        scene.rootNode.addChildNode(cameraOrbit)
            //
            //        // rotate it (I've left out some animation code here to show just the rotation)
            //        cameraOrbit.eulerAngles.x -= Float(CGFloat(M_PI_4))
            //        cameraOrbit.eulerAngles.y -= Float(CGFloat(M_PI_4*3))
            // Allow user to manipulate camera
            ScenekitView.allowsCameraControl = true

            // Show FPS logs and timming
            // sceneView.showsStatistics = true

            // Set background color
//            ScenekitView.backgroundColor = UIColor.white

            // Allow user translate image
            ScenekitView.autoenablesDefaultLighting = true
            ScenekitView.cameraControlConfiguration.allowsTranslation = false
            // ScenekitView.backgroundColor = UIColor(red: 41, green: 42, blue: 51, alpha: 1.0)
            ScenekitView.backgroundColor = UIColor.white
            // Set scene settings
            ScenekitView.scene = scene
            ScenekitView.defaultCameraController.maximumVerticalAngle = 0.001
           
//                SCNTransaction.begin()
//                SCNTransaction.animationDuration = 5
//                scnView.defaultCameraController.translateInCameraSpaceBy(x: 10, y: 10, z: 10)
//                SCNTransaction.commit()
            
           
            
          
            
            // show statistics such as fps and timing information
            ScenekitView.showsStatistics = false
            
            // configure the view
//            ScenekitView.backgroundColor = UIColor.white
            
            // Allow user translate image
            ScenekitView.cameraControlConfiguration.allowsTranslation = false
            
            // scnView.cameraControlConfiguration.rotationSensitivity = true
            let cameraNodes = ScenekitView.pointOfView
            print("cameraNodes:\(cameraNodes)")
            

        } else{
            print("No Image")
        }
        } catch {
            print("error")
            let scnView = SCNView()
        }

    }
}

// MARK: - App Management
/*
extension ShowPointCloudViewController {
  
  func startApp() {
    DispatchQueue.main.async {
      self.arPortNode.isHidden = true
      self.focusNode.isHidden = true
      self.appState = .DetectSurface
    }
  }
  
  func resetApp() {
    DispatchQueue.main.async {
      self.arPortNode.isHidden = true
     
      self.appState = .DetectSurface
    }
  }
}
// MARK: - AR Coaching Overlay

extension ShowPointCloudViewController : ARCoachingOverlayViewDelegate {
    func initARSession() {
      guard ARWorldTrackingConfiguration.isSupported else {
        print("*** ARConfig: AR World Tracking Not Supported")
        return
      }
      
      let config = ARWorldTrackingConfiguration()
      config.worldAlignment = .gravity
      config.providesAudioData = false
      config.planeDetection = .horizontal
      config.isLightEstimationEnabled = true
      config.environmentTexturing = .automatic
      sceneView.session.run(config)
    }
  func initCoachingOverlayView() {
    let coachingOverlay = ARCoachingOverlayView()
    coachingOverlay.session = self.sceneView.session
    coachingOverlay.delegate = self
    coachingOverlay.translatesAutoresizingMaskIntoConstraints = false
    coachingOverlay.activatesAutomatically = true
    coachingOverlay.goal = .horizontalPlane
    self.sceneView.addSubview(coachingOverlay)
    
    NSLayoutConstraint.activate([
      NSLayoutConstraint(item:  coachingOverlay, attribute: .top, relatedBy: .equal,
                         toItem: self.view, attribute: .top, multiplier: 1, constant: 0),
      NSLayoutConstraint(item:  coachingOverlay, attribute: .bottom, relatedBy: .equal,
                         toItem: self.view, attribute: .bottom, multiplier: 1, constant: 0),
      NSLayoutConstraint(item:  coachingOverlay, attribute: .leading, relatedBy: .equal,
                         toItem: self.view, attribute: .leading, multiplier: 1, constant: 0),
      NSLayoutConstraint(item:  coachingOverlay, attribute: .trailing, relatedBy: .equal,
                         toItem: self.view, attribute: .trailing, multiplier: 1, constant: 0)
    ])
  }
  
  func coachingOverlayViewWillActivate(_ coachingOverlayView: ARCoachingOverlayView) {
  }
  
  func coachingOverlayViewDidDeactivate(_ coachingOverlayView: ARCoachingOverlayView) {
    self.startApp()
  }
  
  func coachingOverlayViewDidRequestSessionReset(_ coachingOverlayView: ARCoachingOverlayView) {
    self.resetApp()
  }
}
// MARK: - Scene Management

extension ShowPointCloudViewController {
  
  func initScene() {
    let scene = SCNScene()
    sceneView.scene = scene
    sceneView.delegate = self
    //sceneView.showsStatistics = true
    sceneView.debugOptions = [
      //ARSCNDebugOptions.showFeaturePoints,
      //ARSCNDebugOptions.showWorldOrigin,
      //SCNDebugOptions.showBoundingBoxes,
      //SCNDebugOptions.showWireframe
    ]
    
    let arPortScene = SCNScene(named: "art.scnassets/Scenes/ARPortScene.scn")!
    arPortNode = arPortScene.rootNode.childNode(
      withName: "ARPort", recursively: false)!
    arPortNode.isHidden = true
    sceneView.scene.rootNode.addChildNode(arPortNode)
  }
  
  func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
    DispatchQueue.main.async {
     //  self.updateFocusNode()
     //  self.updateStatus()
    }
  }
  
  override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
    DispatchQueue.main.async {
      if let touchLocation = touches.first?.location(in: self.sceneView) {
        if let hit = self.sceneView.hitTest(touchLocation, options: nil).first {
          if hit.node.name == "Touch" {
            let billboardNode = hit.node.childNode(withName: "Billboard", recursively: false)
            billboardNode?.isHidden = false
          }
          if hit.node.name == "Billboard" {
            hit.node.isHidden = true
          }
        }
      }
    }
  }
  
  func updateStatus() {
    switch appState {
    case .DetectSurface:
      statusMessage = "Scan available flat surfaces..."
    case .PointAtSurface:
      statusMessage = "Point at designated surface first!"
    case .TapToStart:
      statusMessage = "Tap to start."
    case .Started:
      statusMessage = "Tap objects for more info."
    }
    
    self.statusLabel.text = trackingStatus != "" ?
      "\(trackingStatus)" : "\(statusMessage)"
  }
}

// MARK: - Focus Node Management

extension ShowPointCloudViewController {
  
  @objc
  func orientationChanged() {
    focusPoint = CGPoint(x: view.center.x, y: view.center.y  + view.center.y * 0.1)
  }
  
  func initFocusNode() {
    
    let focusScene = SCNScene(named: "art.scnassets/Scenes/FocusScene.scn")!
    focusNode = focusScene.rootNode.childNode(
      withName: "Focus", recursively: false)!
    focusNode.isHidden = true
    sceneView.scene.rootNode.addChildNode(focusNode)
    
    focusPoint = CGPoint(x: view.center.x, y: view.center.y + view.center.y * 0.1)
    NotificationCenter.default.addObserver(self, selector: #selector(ShowPointCloudViewController.orientationChanged), name: UIDevice.orientationDidChangeNotification, object: nil)
  }
  
  func updateFocusNode() {
    
    guard appState != .Started else {
      focusNode.isHidden = true
      return
    }
    
    if let query = self.sceneView.raycastQuery(from: self.focusPoint, allowing: .estimatedPlane, alignment: .horizontal) {
      let results = self.sceneView.session.raycast(query)
      
      if results.count == 1 {
        if let match = results.first {
          let t = match.worldTransform
          self.focusNode.position = SCNVector3(x: t.columns.3.x, y: t.columns.3.y, z: t.columns.3.z)
          self.appState = .TapToStart
          focusNode.isHidden = false
        }
      } else {
        self.appState = .PointAtSurface
        focusNode.isHidden = true
      }
    }
  }
}

*/

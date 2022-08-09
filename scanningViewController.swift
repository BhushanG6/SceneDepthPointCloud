/*
See LICENSE folder for this sample’s licensing information.

Abstract:
Main view controller for the AR experience.
*/

import ARKit
import Combine
import Metal
import MetalKit
import RealityKit
import simd
import SwiftUI
import UIKit
import AVFoundation

final class scanningViewController: UIViewController, ARSessionDelegate {
    
    var sx:Float=0;
    var sy:Float=0;
    var sz:Float=0;
    var displace:simd_float3=[0.0,0.0,0.0]
    private let isUIEnabled = true
    private let confidenceControl = UISegmentedControl(items: ["Low", "Medium", "High"])
    private let rgbRadiusSlider = UISlider()
    let child = SpinnerViewController()
    private let session = ARSession()
    @IBOutlet weak var saveButton: UIButton!
    private var renderer: Renderer!
    var centerofbody : simd_float3 = [0,0,0]
    var leftshoulder : simd_float3 = [0,0,0]
    var rightshoulder : simd_float3 = [0,0,0]
    // MARK: - Properties
    var trackingStatus: String = ""
    var arrayofpointcloud = [0,1,2,3]
    var i = 0
    var date = Date()
    var player: AVAudioPlayer?
    func createSpinnerView() {
        let child = SpinnerViewController()

        // add the spinner view controller
        addChild(child)
        child.view.frame = view.frame
        view.addSubview(child.view)
        child.didMove(toParent: self)

        // wait two seconds to simulate some work happening
        DispatchQueue.main.asyncAfter(deadline: .now() + 3) {
            // then remove the spinner view controller
            child.willMove(toParent: nil)
            child.view.removeFromSuperview()
            child.removeFromParent()
        }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return
        }
        
        session.delegate = self
        displace=[sx,sy,sz]
        // Set the view to use the default device
        if let view = view as? MTKView {
            view.device = device
            
            view.backgroundColor = UIColor.clear
            // we need this to enable depth test
            view.depthStencilPixelFormat = .depth32Float
            view.contentScaleFactor = 1
            view.delegate = self
            
            // Configure the renderer to draw to the view
            renderer = Renderer(session: session, metalDevice: device, renderDestination: view)
            renderer.drawRectResized(size: view.bounds.size)
            renderer.center_of_body = centerofbody
            renderer.rightshoulder = rightshoulder
            renderer.leftshoulder = leftshoulder
            renderer.displ=displace
            DispatchQueue.main.async { [self] in
                self.renderer.writehearderofPly()
            print("center_of_body\(renderer.center_of_body)")
            }
        }
        /*
        // Confidence control
        confidenceControl.backgroundColor = .white
        confidenceControl.selectedSegmentIndex = renderer.confidenceThreshold
        confidenceControl.addTarget(self, action: #selector(viewValueChanged), for: .valueChanged)
        
        // RGB Radius control
        rgbRadiusSlider.minimumValue = 0
        rgbRadiusSlider.maximumValue = 1.5
        rgbRadiusSlider.isContinuous = true
        rgbRadiusSlider.value = renderer.rgbRadius
        rgbRadiusSlider.addTarget(self, action: #selector(viewValueChanged), for: .valueChanged)
        
        let stackView = UIStackView(arrangedSubviews: [confidenceControl, rgbRadiusSlider])
        stackView.isHidden = !isUIEnabled
        stackView.translatesAutoresizingMaskIntoConstraints = false
        stackView.axis = .vertical
        stackView.spacing = 20
        
        view.addSubview(stackView)
        NSLayoutConstraint.activate([
            stackView.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            stackView.bottomAnchor.constraint(equalTo: view.bottomAnchor, constant: -50)
        ])*/
        NotificationCenter.default.addObserver(self, selector: #selector(self.loadshowpointcloudview), name: Notification.Name("reloadviewscnuploaded"), object: nil)
        
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        
        // Create a world-tracking configuration, and
        // enable the scene depth frame-semantic.
        
//        let configuration = ARWorldTrackingConfiguration()
//        configuration.frameSemantics = .sceneDepth
//
//        // Run the view's session
//        session.run(configuration)
        initARSession()
//        let config = ARWorldTrackingConfiguration()
//        config.worldAlignment = .gravity
//        config.providesAudioData = false
//        config.isLightEstimationEnabled = true
//        config.environmentTexturing = .automatic
//        session.run(config)
        // The screen shouldn't dim during AR experiences.
        UIApplication.shared.isIdleTimerDisabled = true
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
    func updatesession(){
        guard let device = MTLCreateSystemDefaultDevice() else {
            print("Metal is not supported on this device")
            return
        }
        
        session.delegate = self
        
        // Set the view to use the default device
        if let view = view as? MTKView {
            view.device = device
            
            view.backgroundColor = UIColor.clear
            // we need this to enable depth test
            view.depthStencilPixelFormat = .depth32Float
            view.contentScaleFactor = 1
            view.delegate = self
            
            // Configure the renderer to draw to the view
            renderer = Renderer(session: session, metalDevice: device, renderDestination: view)
            renderer.drawRectResized(size: view.bounds.size)
        }
    }
    @objc internal func loadshowpointcloudview(){
        DispatchQueue.main.async {
        self.child.willMove(toParent: nil)
        self.child.view.removeFromSuperview()
        self.child.removeFromParent()
    }
        let mainStoryBoard = UIStoryboard(name: "Main", bundle: nil)
        let secondViewController = mainStoryBoard.instantiateViewController(withIdentifier: "ShowPointCloudViewController") as! ShowPointCloudViewController
            secondViewController.allpoint = self.renderer.finalPoints
        secondViewController.filename = self.renderer.filenamecutarm
        self.navigationController?.pushViewController(secondViewController, animated: true)
    }
    @IBAction func saveButtonAction(_ sender: Any) {
        self.renderer.particleBufferIn()
        DispatchQueue.main.async {
           //  self.updatesession()
        self.initARSession()
            
        }
       
       

    }
    
    @IBAction func showpointcloud(_ sender: Any) {
        let mainStoryBoard = UIStoryboard(name: "Main", bundle: nil)
        let secondViewController = mainStoryBoard.instantiateViewController(withIdentifier: "ShowPointCloudViewController") as! ShowPointCloudViewController
            secondViewController.allpoint = self.renderer.finalPoints
        self.navigationController?.pushViewController(secondViewController, animated: true)
    }
    @objc
    private func viewValueChanged(view: UIView) {
        switch view {
            
        case confidenceControl:
            renderer.confidenceThreshold = confidenceControl.selectedSegmentIndex
            
        case rgbRadiusSlider:
            renderer.rgbRadius = rgbRadiusSlider.value
            
        default:
            break
        }
    }
    
    // Auto-hide the home indicator to maximize immersion in AR experiences.
    override var prefersHomeIndicatorAutoHidden: Bool {
        return true
    }
    
    // Hide the status bar to maximize immersion in AR experiences.
    override var prefersStatusBarHidden: Bool {
        return true
    }
    
}

// MARK: - MTKViewDelegate

extension scanningViewController: MTKViewDelegate {
    // Called whenever view changes orientation or layout is changed
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        renderer.drawRectResized(size: size)
    }
    
    // Called whenever the view needs to render
    func draw(in view: MTKView) {
      
        if self.renderer.issessioninitilize {
        print("test draw")
            renderer.draw()
         }
    }
}

// MARK: - RenderDestinationProvider

protocol RenderDestinationProvider {
    var currentRenderPassDescriptor: MTLRenderPassDescriptor? { get }
    var currentDrawable: CAMetalDrawable? { get }
    var colorPixelFormat: MTLPixelFormat { get set }
    var depthStencilPixelFormat: MTLPixelFormat { get set }
    var sampleCount: Int { get set }
}

extension MTKView: RenderDestinationProvider {
    
}

// MARK: - AR Session Management (ARSCNViewDelegate)

extension scanningViewController {
    
  func initARSession() {
      /*
    guard ARWorldTrackingConfiguration.isSupported else {
      print("*** ARConfig: AR World Tracking Not Supported")
      return
    }
    
    let config = ARWorldTrackingConfiguration()
    config.worldAlignment = .gravity
    config.providesAudioData = false
    config.isLightEstimationEnabled = true
    config.environmentTexturing = .automatic
    session.run(config) */
      DispatchQueue.main.async {
          let configuration = ARWorldTrackingConfiguration()
          configuration.frameSemantics = .sceneDepth
          configuration.isAutoFocusEnabled = false
          // Run the view's session
          self.session.run(configuration)
         // self.renderer.particleBufferIn()
          print("session started ")
         
           
         
          DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
              self.renderer.issessioninitilize = true
           Timer.scheduledTimer(timeInterval: 3, target: self, selector: #selector(self.callback), userInfo: nil, repeats: false)
          }
          
      }
     
     
  }
    @objc func callback() {
        print("done")
        self.renderer.issessioninitilize = false
//        self.renderer.particleBufferIn()
      
     //   DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
          
            var num = self.arrayofpointcloud[self.i]
          print("num of data \(num) : \(i)")
       
            self.renderer.removenoisypointnew(num: num)
            
            self.renderer.particleBufferIn()
            self.renderer.isSavingFile = true
            self.session.pause()
            self.i = self.i + 1
          
            
        
      //  self.updatesession()
        if i <= 3 {
            
          //  let skeleton = bodyAncor.skeleton

            print("here 12 ")
        
           
            self.resetARSession()
//            self.renderer.particleBufferIn()
        
        } else {
            print("here ")
            self.renderer.changeoriginnew(num: i)
            DispatchQueue.main.asyncAfter(deadline: .now() + 1) {
            // add the spinner view controller
                self.addChild(self.child)
                self.child.view.frame = self.view.frame
                self.view.addSubview(self.child.view)
                self.child.didMove(toParent: self)
                self.playSound(soundname: "scan_complete")
            }
            
                
            
           
        }
       
    }
  func resetARSession() {
//    let config = session.configuration as!
//      ARWorldTrackingConfiguration
//
//      config.frameSemantics = .sceneDepth
//      config.isAutoFocusEnabled = false
//    session.run(config, options: [.resetTracking, .removeExistingAnchors])
      

      // add the spinner view controller
      
     
      let configuration = ARWorldTrackingConfiguration()
      configuration.frameSemantics = .sceneDepth
      configuration.isAutoFocusEnabled = false
      // Run the view's session
      self.session.run(configuration)
      DispatchQueue.main.asyncAfter(deadline: .now() + 2) {
          self.addChild(self.child)
          self.child.view.frame = self.view.frame
          self.view.addSubview(self.child.view)
          self.child.didMove(toParent: self)
          self.playSound(soundname: "right")
      }
      DispatchQueue.main.asyncAfter(deadline: .now() + 7) {
      self.child.willMove(toParent: nil)
      self.child.view.removeFromSuperview()
      self.child.removeFromParent()
      }
//      DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
//          self.playSound(soundname: "starting_scan")
//      }
      
      DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
         
          self.renderer.particleBufferIn()
          self.renderer.issessioninitilize = true
           
          
           Timer.scheduledTimer(timeInterval: 5, target: self, selector: #selector(self.callback), userInfo: nil, repeats: false)
      }
     
  }
  
  func session(_ session: ARSession, cameraDidChangeTrackingState camera: ARCamera) {
    switch camera.trackingState {
    case .notAvailable:
      self.trackingStatus = "Tracking:  Not available!"
    case .normal:
      self.trackingStatus = ""
    case .limited(let reason):
      switch reason {
      case .excessiveMotion:
        self.trackingStatus = "Tracking: Limited due to excessive motion!"
      case .insufficientFeatures:
        self.trackingStatus = "Tracking: Limited due to insufficient features!"
      case .relocalizing:
        self.trackingStatus = "Tracking: Relocalizing..."
      case .initializing:
        self.trackingStatus = "Tracking: Initializing..."
      @unknown default:
        self.trackingStatus = "Tracking: Unknown..."
      }
    }
  }
  

    func session(_ session: ARSession, didFailWithError error: Error) {
        // Present an error message to the user.
        self.trackingStatus = "AR Session Failure: \(error)"
        guard error is ARError else { return }
        let errorWithInfo = error as NSError
        let messages = [
            errorWithInfo.localizedDescription,
            errorWithInfo.localizedFailureReason,
            errorWithInfo.localizedRecoverySuggestion
        ]
        let errorMessage = messages.compactMap({ $0 }).joined(separator: "\n")
        DispatchQueue.main.async {
            // Present an alert informing about the error that has occurred.
            let alertController = UIAlertController(title: "The AR session failed.", message: errorMessage, preferredStyle: .alert)
            let restartAction = UIAlertAction(title: "Restart Session", style: .default) { _ in
                alertController.dismiss(animated: true, completion: nil)
                if let configuration = self.session.configuration {
                    self.session.run(configuration, options: .resetSceneReconstruction)
                }
            }
            alertController.addAction(restartAction)
            self.present(alertController, animated: true, completion: nil)
        }
    }

  func sessionWasInterrupted(_ session: ARSession) {
    self.trackingStatus = "AR Session Was Interrupted!"
  }
  
  func sessionInterruptionEnded(_ session: ARSession) {
    self.trackingStatus = "AR Session Interruption Ended"
  }
}
// MARK: - Scene Management

extension scanningViewController {
  /**
  func initScene() {
    let scene = SCNScene()
    sceneView.scene = scene
    sceneView.delegate = self
  }
  
  func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
    DispatchQueue.main.async {
      self.updateStatus()
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
  } **/
}

// MARK: - Focus Node Management


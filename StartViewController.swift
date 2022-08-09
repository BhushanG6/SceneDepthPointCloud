//
//  StartViewController.swift
//  SceneDepthPointCloud
//
//  Created by Monali Palhal on 28/07/22.
//  Copyright © 2022 Apple. All rights reserved.
//

import UIKit

class StartViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        

        // Do any additional setup after loading the view.
    }
    
    @IBAction func showPointCloud(_ sender: Any) {
        let mainStoryBoard = UIStoryboard(name: "Main", bundle: nil)
        let secondViewController = mainStoryBoard.instantiateViewController(withIdentifier: "ShowPointCloudViewController") as! ShowPointCloudViewController
        secondViewController.scanshow = true
        secondViewController.filename = Helper().retrievePathnameFromKeychain() ?? ""
        self.navigationController?.pushViewController(secondViewController, animated: true)
    }
    @IBAction func startScanning(_ sender: Any) {
//        var sum : simd_float3 = [0,0,0]
//        var avg : simd_float3 = [0,0,0]
//        for i in arrayofallcenter {
//            sum = sum + i
//        }
//        var count : Float = Float(arrayofallcenter.count)
//        avg.x = sum.x / count
//        avg.y = sum.y / count
//        avg.z = sum.z / count
//        print("avg of center\(avg)")
        let mainStoryBoard = UIStoryboard(name: "Main", bundle: nil)
        let secondViewController = mainStoryBoard.instantiateViewController(withIdentifier: "ViewController") as! ViewController
//        if count == 0 {
//            avg = [0, 0, 0]
//        }
        
       
        // secondViewController.centerofbody = [0, 0, 0]
        
            
        self.navigationController?.pushViewController(secondViewController, animated: true)
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

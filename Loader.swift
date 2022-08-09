//
//  Loader.swift
//  SceneDepthPointCloud
//
//  Created by Monali Palhal on 11/07/22.
//  Copyright © 2022 Apple. All rights reserved.
//
import UIKit
/*
import Foundation
// MARK: - _______________  LOADER   _______________
import SVProgressHUD

/// Show loader
public func showLoader() {
    OperationQueue.main.addOperation {
        SVProgressHUD.setDefaultMaskType(.gradient)
        SVProgressHUD.setBorderColor(UIColor.AppColor.lightingPurpleColor)
        SVProgressHUD.setForegroundColor(UIColor.AppColor.lightingPurpleColor)
        
        SVProgressHUD.show()
    }
}

/// Hide loader
public func hideLoader() {
    OperationQueue.main.addOperation {
        SVProgressHUD.dismiss()
    }
}*/
class SpinnerViewController: UIViewController {
    var spinner = UIActivityIndicatorView(style: .whiteLarge)
    let loadingTextLabel = UILabel()

    override func loadView() {
        view = UIView()
        view.backgroundColor = UIColor(white: 0, alpha: 0.7)

        spinner.translatesAutoresizingMaskIntoConstraints = false
        loadingTextLabel.translatesAutoresizingMaskIntoConstraints = false
        spinner.startAnimating()
        loadingTextLabel.textColor = UIColor.white
                loadingTextLabel.text = "Processing....."
                loadingTextLabel.font = UIFont(name: "Avenir Light", size: 20)
                loadingTextLabel.sizeToFit()
               // loadingTextLabel.center = CGPoint(x: view.center.x, y: view.center.y + 30)
               
        view.addSubview(spinner)
        view.addSubview(loadingTextLabel)

        spinner.centerXAnchor.constraint(equalTo: view.centerXAnchor).isActive = true
        spinner.centerYAnchor.constraint(equalTo: view.centerYAnchor).isActive = true
        loadingTextLabel.centerXAnchor.constraint(equalTo: view.centerXAnchor).isActive = true
        loadingTextLabel.centerYAnchor.constraint(equalTo: view.centerYAnchor,constant: 30).isActive = true
    }
}

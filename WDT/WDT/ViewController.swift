//
//  ViewController.swift
//  WDT
//
//  Created by edison on 2021/9/30.
//

import UIKit

let width = UIScreen.main.bounds.width
let height = UIScreen.main.bounds.height
let background = UIImageView()

class ViewController: UIViewController {
    
    let manualButton = UIButton()
    let autoButton = UIButton()
    let hintLabel = UILabel()
    let fundLabel = UILabel()
    let authorLabel = UILabel()
    
    let labelWeight: CGFloat = 300
    let labelHeight: CGFloat = 50
    let labelTextColor = UIColor.gray

    override func viewDidLoad() {
        super.viewDidLoad()
        setUI()
    }
    
    func setUI() {
        background.frame = CGRect.init(x: 0, y: 0, width: width, height: height)
        background.image = UIImage.init(named: "bg")
        background.contentMode = .scaleAspectFill
        self.view.addSubview(background)
        
        manualButton.frame = CGRect.init(x: (width - labelWeight) / 2, y: 200, width: labelWeight, height: labelHeight)
        manualButton.backgroundColor = .black
        manualButton.layer.cornerRadius = 10
        manualButton.setTitle("detect wheat", for: .normal)
        manualButton.setTitleColor(.white, for: .normal)
        manualButton.addTarget(self, action: #selector(scan), for: .touchDown)
        self.view.addSubview(manualButton)
      
        
        hintLabel.text = "wheat head detector based on WDNetwork"
        hintLabel.backgroundColor = .white
        hintLabel.alpha = 0.1
        hintLabel.layer.cornerRadius = 10
        hintLabel.layer.masksToBounds = true
        hintLabel.textColor = labelTextColor
        hintLabel.sizeToFit()
        hintLabel.textAlignment = .center
        hintLabel.frame = CGRect.init(x: (width - 400) / 2, y: height - labelHeight * 3 - 25, width: 400, height: labelHeight)
        self.view.addSubview(hintLabel)
        
        fundLabel.text = "funded by "
        fundLabel.backgroundColor = .white
        fundLabel.textColor = labelTextColor
        fundLabel.font = UIFont.italicSystemFont(ofSize: 14)
        fundLabel.sizeToFit()
        fundLabel.frame = CGRect.init(x: (width - labelWeight) / 2, y: height - labelHeight * 2 - 25, width: labelWeight, height: labelHeight)
//        self.view.addSubview(fundLabel)
        
        authorLabel.text = "Created by Edison"
        authorLabel.textColor = labelTextColor
        authorLabel.font = UIFont.italicSystemFont(ofSize: 12)
        authorLabel.textAlignment = .center
        authorLabel.sizeToFit()
        authorLabel.textColor = .white
        authorLabel.frame = CGRect.init(x: (width - labelWeight) / 2, y: height - labelHeight - 25, width: labelWeight, height: labelHeight)
        self.view.addSubview(authorLabel)
    }
    
    @objc func scan(_ vc: UIViewController, sender: Any?) {
        let vc = ScanController()
        self.present(vc, animated: false, completion: nil)
    }
}


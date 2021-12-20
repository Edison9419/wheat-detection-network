//
//  ScanController.swift
//  WDT
//
//  Created by edison on 2021/9/30.
//

import UIKit
import AVFoundation

class ScanController: UIViewController, AVCaptureMetadataOutputObjectsDelegate {
    
    var session: AVCaptureSession?
    var device: AVCaptureDevice?
    var input: AVCaptureDeviceInput?
    var output: AVCaptureMetadataOutput?
    var layer: AVCaptureVideoPreviewLayer?
    let hint = UILabel()
    let captureButton = UIButton()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .green
        initSession()
        initContent()
    }
    
    private func initSession() {
        device = AVCaptureDevice.default(for: AVMediaType.video)
        input = try! AVCaptureDeviceInput.init(device: device!)
        output = AVCaptureMetadataOutput.init()
        output?.setMetadataObjectsDelegate(self, queue: DispatchQueue.main)
        session = AVCaptureSession.init()
        session?.sessionPreset = AVCaptureSession.Preset.high
        session?.addInput(input!)
        session?.addOutput(output!)
        output?.metadataObjectTypes = [.qr, .code128, .ean8, .ean13]
        
        layer = AVCaptureVideoPreviewLayer.init(session: session!)
        layer?.videoGravity = AVLayerVideoGravity.resizeAspectFill
        layer?.frame = view.frame
        view.layer.addSublayer(layer!)
        session?.startRunning()
    }
    
    private func initContent() {
        
        let frame = UIImageView()
        frame.frame = .init(x: (width-300)/2, y: 200, width: 300, height: 300)
        frame.contentMode = .scaleAspectFill
        frame.image = UIImage.init(named: "frame.png")
        view.addSubview(frame)
        
        hint.frame = .init(x: (width - 120) / 2, y: 100, width: 120, height: 30)
        hint.text = "capture image"
        hint.textAlignment = .center
        hint.textColor = .white
        hint.font = UIFont.systemFont(ofSize: 15)
        hint.backgroundColor = UIColor.init(white: 0.5, alpha: 0.5)
        hint.layer.borderColor = UIColor.white.cgColor
        hint.layer.cornerRadius = 10
        hint.layer.masksToBounds = true
        view.addSubview(hint)
        
        captureButton.frame = .init(x: (width - 50) / 2, y: 600, width: 50, height: 50)
        captureButton.layer.cornerRadius = 25
        captureButton.backgroundColor = .white
//        captureButton.alpha = 0.5
        captureButton.setTitle("📷", for: .normal)
        captureButton.addTarget(self, action: #selector(capture), for: .touchDown)
        view.addSubview(captureButton)
    }
    
    @objc private func capture() {
        let renderer = UIGraphicsImageRenderer(bounds: view.bounds)
        let image = renderer.image { rendererContext in
                    layer!.render(in: rendererContext.cgContext)
                }
        let url:NSURL! = NSURL(string: "192.0.0.1/wdn.php")
        var request = URLRequest(url: url as URL)
        request.httpBody = image.jpegData(compressionQuality: 1.0)
        let config = URLSessionConfiguration.default
        let session = URLSession(configuration: config)
        let dataTask = session.dataTask(with: request) { (data, response, error) in
            let jsonData:[[Double]] = try! JSONSerialization.jsonObject(with: data!, options: .mutableContainers) as! [[Double]]

            self.drawBoxes(boxes: jsonData)

        }
        dataTask.resume()
    }
    
    func drawBoxes(boxes: [[Double]]) {
        hint.text = String(boxes.count)
        for box in boxes {
            let box = UIView.init(frame: .init(x: box[0], y: box[1], width: box[2], height: box[3]))
            view.addSubview(box)
        }
    }
}

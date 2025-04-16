import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:smart_scan_amr/app/ocr_service.dart';

class MyApp extends StatelessWidget {
  const MyApp({required this.ocrService, super.key});

  final OCRService ocrService;

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'SmartScan-AMR',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: OCRHomePage(ocrService: ocrService),
    );
  }
}

class OCRHomePage extends StatefulWidget {
  const OCRHomePage({required this.ocrService, super.key});

  final OCRService ocrService;

  @override
  State<OCRHomePage> createState() => _OCRHomePageState();
}

class _OCRHomePageState extends State<OCRHomePage> {
  final ImagePicker _picker = ImagePicker();

  Uint8List? _imageBytes;
  String _recognizedText = '';

  Future<void> _takePicture() async {
    final file = await _picker.pickImage(source: ImageSource.camera);
    if (file == null) return;

    final bytes = await file.readAsBytes();
    setState(() {
      _imageBytes = bytes;
      _recognizedText = 'Recognizing...';
    });

    final text = await widget.ocrService.recognize(bytes);
    setState(() => _recognizedText = text);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('SmartScan-AMR')),
      body: Column(
        children: [
          // Top part: image preview
          Expanded(
            flex: 3,
            child: ColoredBox(
              color: Colors.black12,
              child: _imageBytes != null
                  ? Image.memory(_imageBytes!, fit: BoxFit.contain)
                  : const Center(child: Text('No image captured')),
            ),
          ),
          // Bottom part: recognized text
          Expanded(
            flex: 2,
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(16),
              child: Text(
                _recognizedText,
                style: const TextStyle(fontSize: 16),
              ),
            ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _takePicture,
        child: const Icon(Icons.camera_alt),
      ),
    );
  }
}

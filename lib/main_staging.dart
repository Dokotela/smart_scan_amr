import 'package:flutter/material.dart';
import 'package:smart_scan_amr/app/app.dart';
import 'package:smart_scan_amr/app/ocr_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final ocrService = await OCRService.create();
  runApp(MyApp(ocrService: ocrService));
}

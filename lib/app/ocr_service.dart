import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

class OCRService {
  late final OrtSession _encoder;
  late final OrtSession _decoder;
  late final Map<int, String> idToToken;

  /// Initialize ONNX sessions and load tokenizer
  Future<void> init() async {
    OrtEnv.instance.init();
    final opts = OrtSessionOptions();

    // Load encoder ONNX model (ensure correct path)
    final encData = await rootBundle.load('assets/models/TrOCREncoder.onnx');
    _encoder = OrtSession.fromBuffer(encData.buffer.asUint8List(), opts);

    // Load decoder ONNX model (ensure correct path)
    final decData = await rootBundle.load('assets/models/TrOCRDecoder.onnx');
    _decoder = OrtSession.fromBuffer(decData.buffer.asUint8List(), opts);

    // Load vocabulary (token ID → token string)
    final rawVocab = jsonDecode(
      await rootBundle.loadString('assets/models/vocab.json'),
    ) as Map<String, dynamic>;
    if (rawVocab.keys.every((k) => int.tryParse(k) != null)) {
      // Case 1: ID→token (keys are numeric strings)
      idToToken = rawVocab.map((k, v) => MapEntry(int.parse(k), v as String));
    } else {
      // Case 2: token→ID (keys are token strings, values are ints)
      idToToken = {
        for (final e in rawVocab.entries) (e.value as num).toInt(): e.key,
      };
    }
  }

  /// Preprocess raw image bytes to a Float32List of shape [1, 3, 320, 320]
  Float32List _preprocess(Uint8List bytes) {
    final original = img.decodeImage(bytes)!;
    final resized = img.copyResize(original, width: 320, height: 320);
    final buffer = Float32List(1 * 3 * 320 * 320);
    var idx = 0;

    for (var y = 0; y < 320; y++) {
      for (var x = 0; x < 320; x++) {
        final pixel = resized.getPixel(x, y);
        final r = pixel.r.toInt();
        final g = pixel.g.toInt();
        final b = pixel.b.toInt();

        // Normalize pixel values to [-1, 1]
        buffer[idx++] = ((r / 255.0) - 0.5) / 0.5;
        buffer[idx++] = ((g / 255.0) - 0.5) / 0.5;
        buffer[idx++] = ((b / 255.0) - 0.5) / 0.5;
      }
    }
    return buffer;
  }

  /// Perform OCR: runs encoder followed by greedy decoder
  Future<String> recognize(Uint8List imageBytes) async {
    // 1) Encoder inference
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      _preprocess(imageBytes),
      [1, 3, 320, 320],
    );
    final encResults = await _encoder.runAsync(
      OrtRunOptions(),
      {'pixel_values': inputTensor},
    );
    inputTensor.release();
    final encHidden = encResults![0]!;

    // 2) Greedy decoding loop
    const startToken = 2;
    const eosToken = 2;
    final tokenIds = <int>[startToken];

    for (var step = 0; step < 64; step++) {
      final idsTensor = OrtValueTensor.createTensorWithDataList(
        Int64List.fromList(tokenIds),
        [1, tokenIds.length],
      );

      final decResults = await _decoder.runAsync(
        OrtRunOptions(),
        {
          'input_ids': idsTensor,
          'encoder_hidden_states': encHidden,
        },
      );
      idsTensor.release();

      final raw = decResults?[0]?.value;
      late Float32List logits;
      if (raw is Float32List) {
        logits = raw;
      } else if (raw is List) {
        logits = Float32List.fromList(List<double>.from(raw));
      } else {
        throw StateError('Unexpected tensor type: ${raw.runtimeType}');
      }

      final vocabSize = logits.length ~/ tokenIds.length;
      final offset = (tokenIds.length - 1) * vocabSize;
      final slice = logits.sublist(offset, offset + vocabSize);
      final nextId =
          slice.indexWhere((v) => v == slice.reduce((a, b) => a > b ? a : b));

      for (final v in decResults ?? <OrtValue?>[]) {
        v?.release();
      }
      if (nextId == eosToken) break;
      tokenIds.add(nextId);
    }
    encHidden.release();

    final tokens = tokenIds.map((id) => idToToken[id] ?? '').toList();
    final text = tokens.join();
    return text.replaceAll('</s>', '').trim();
  }
}

// lib/ocr_service.dart
// ignore_for_file: avoid_print

import 'dart:async';
import 'dart:convert';
import 'dart:math';
import 'dart:typed_data';

import 'package:collection/collection.dart';
import 'package:crypto/crypto.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

/// Service for optical character recognition using quantized ONNX models
class OCRService {
  OCRService({
    required this.encoder,
    required this.decoder,
    required this.idToToken,
    required this.startTokenId,
    required this.endTokenId,
  });

  final OrtSession encoder;
  final OrtSession decoder;
  final Map<int, String> idToToken;
  final int startTokenId;
  final int endTokenId;
  bool _isDisposed = false;

  bool get isDisposed => _isDisposed;

  /// Debug mode flag for additional logging
  static const bool _debugMode = true;

  /// Static factory method to create and initialize the OCR service
  static Future<OCRService> create({int? numThreads}) async {
    final threadCount = numThreads ?? 2;
    _debugLog('Initializing OCRService with $threadCount threads');

    // Initialize ONNX Runtime
    OrtEnv.instance.init();
    final opts = OrtSessionOptions()
      ..setIntraOpNumThreads(threadCount)
      ..setSessionGraphOptimizationLevel(GraphOptimizationLevel.ortEnableAll);

    // Load quantized encoder & decoder
    _debugLog('Loading encoder model...');
    final encBytes = await rootBundle.load('assets/models/encoder-int8.onnx');
    final encBytesList = Uint8List.sublistView(
      encBytes.buffer
          .asUint8List(encBytes.offsetInBytes, encBytes.lengthInBytes),
    );
    final encoder = OrtSession.fromBuffer(encBytesList, opts);
    _debugLog(
      'Encoder loaded: ${(encBytesList.length / 1024).toStringAsFixed(1)} KB',
    );

    _debugLog('Loading decoder model...');
    final decBytes = await rootBundle.load('assets/models/decoder-int8.onnx');
    final decBytesList = Uint8List.sublistView(
      decBytes.buffer
          .asUint8List(decBytes.offsetInBytes, decBytes.lengthInBytes),
    );
    final decoder = OrtSession.fromBuffer(decBytesList, opts);
    _debugLog(
      'Decoder loaded: ${(decBytesList.length / 1024).toStringAsFixed(1)} KB',
    );

    // Load vocabulary
    _debugLog('Loading vocabulary...');
    final vocabJsonStr =
        await rootBundle.loadString('assets/models/vocab.json');
    final fullVocab = jsonDecode(vocabJsonStr) as Map<String, dynamic>;
    final idToToken = <int, String>{};

    // Hugging Face usually maps token -> id, but sometimes reversed.
    fullVocab.forEach((token, idVal) {
      if (idVal is num) {
        // token → id
        idToToken[idVal.toInt()] = token;
      } else if (idVal is String && int.tryParse(token) != null) {
        // id (string) → token
        idToToken[int.parse(token)] = idVal;
      }
    });

    _debugLog('Loaded full vocab size: ${idToToken.length}');
    if (idToToken.length < 1000) {
      print('⚠️  Looks like vocab.json is too small — '
          'did you bundle the correct file?');
    }

    // Incorporate any added_tokens from tokenizer.json
    try {
      final tokJson =
          await rootBundle.loadString('assets/models/tokenizer.json');
      final tokCfg = jsonDecode(tokJson) as Map<String, dynamic>;
      if (tokCfg['added_tokens'] is List) {
        for (final t in tokCfg['added_tokens'] as List) {
          if (t is Map && t['id'] is num && t['content'] is String) {
            idToToken[(t['id'] as num).toInt()] = t['content'] as String;
          }
        }
        _debugLog('Added ${tokCfg['added_tokens'].length} special tokens');
      }
    } catch (e) {
      _debugLog('Warning: Could not load tokenizer.json - $e');
    }

    // Auto-detect <s> and </s> IDs
    final startTokenId = idToToken.entries
        .firstWhere(
          (e) => e.value == '<s>',
          orElse: () => const MapEntry(-1, ''),
        )
        .key;
    final endTokenId = idToToken.entries
        .firstWhere(
          (e) => e.value == '</s>',
          orElse: () => const MapEntry(-1, ''),
        )
        .key;

    _debugLog('Detected BOS=$startTokenId  EOS=$endTokenId');
    if (startTokenId < 0 || endTokenId < 0) {
      throw Exception('Could not find both "<s>" and "</s>" in vocab.json - '
          'model may not work correctly');
    }

    _debugLog('✅ OCRService initialized successfully');

    return OCRService(
      encoder: encoder,
      decoder: decoder,
      idToToken: idToToken,
      startTokenId: startTokenId,
      endTokenId: endTokenId,
    );
  }

  /// Helper for debug logging
  static void _debugLog(String message) {
    if (_debugMode) {
      print('OCRService: $message');
    }
  }

  /// 1) Decode → resize → normalize → NCHW Float32List → log a sample.
  Float32List _preprocess(Uint8List bytes) {
    final stopwatch = Stopwatch()..start();

    // Decode
    final raw = img.decodeImage(bytes);
    if (raw == null) {
      throw Exception('Failed to decode image data: invalid bytes.');
    }

    // Resize
    final resized = img.copyResize(raw, width: 384, height: 384);

    // ImageNet norm
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    // Allocate [1,3,384,384]
    final out = Float32List(1 * 3 * 384 * 384);
    var i = 0;

    // Fill in channel-first order
    for (var c = 0; c < 3; c++) {
      for (var y = 0; y < 384; y++) {
        for (var x = 0; x < 384; x++) {
          final pixel = resized.getPixel(x, y);
          // Pixel has .r, .g, .b getters
          final r = pixel.r;
          final g = pixel.g;
          final b = pixel.b;
          final channelValue = (c == 0
                  ? r
                  : c == 1
                      ? g
                      : b) /
              255.0;
          out[i++] = (channelValue - mean[c]) / std[c];
        }
      }
    }

    _debugLog('Preprocessing completed in ${stopwatch.elapsedMilliseconds}ms');
    _logPreprocessSample(out);
    return out;
  }

  /// Run encoder + greedy decoder to perform OCR on image bytes,
  /// but on the very first step, if EOS is the top logit, pick the true second‑best instead.
  Future<String> recognize(Uint8List imageBytes) async {
    final stopwatch = Stopwatch()..start();
    if (_isDisposed) throw StateError('OCRService has been disposed.');

    _debugLog('Starting OCR on ${imageBytes.length} bytes');

    // 1) Preprocess & create encoder input
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      _preprocess(imageBytes),
      [1, 3, 384, 384],
    );

    // 2) Run encoder
    _debugLog('Running encoder...');
    final encStart = stopwatch.elapsedMilliseconds;
    final encOut = await encoder.runAsync(
      OrtRunOptions(),
      {'pixel_values': inputTensor},
    );
    _debugLog('Encoder done in ${stopwatch.elapsedMilliseconds - encStart}ms');
    if (encOut == null || encOut.isEmpty || encOut[0] == null) {
      inputTensor.release();
      throw Exception('Encoder produced no output.');
    }
    final OrtValue encHidden = encOut[0]!;

    try {
      // 3) Greedy decode loop
      _debugLog('Starting greedy decode...');
      final tokenIds = <int>[startTokenId];
      const int maxSteps = 64, maxRepeats = 5;

      for (int step = 0; step < maxSteps; step++) {
        // 3a) Build decoder input_ids tensor
        final idsTensor = OrtValueTensor.createTensorWithDataList(
          Int64List.fromList(tokenIds),
          [1, tokenIds.length],
        );

        // 3b) Run decoder
        final decOut = await decoder.runAsync(
          OrtRunOptions(),
          {
            'input_ids': idsTensor,
            'encoder_hidden_states': encHidden,
          },
        );
        if (decOut == null || decOut.isEmpty || decOut[0] == null) {
          idsTensor.release();
          throw Exception('Decoder produced no output at step $step.');
        }

        // 3c) Extract logits
        final rawLogits = decOut[0]!.value;
        final Float32List logits = rawLogits is Float32List
            ? rawLogits
            : _flattenNestedList(rawLogits);

        // 3d) Compute the slice for this timestep
        final int vocabSize = logits.length ~/ tokenIds.length;
        final int offset = (tokenIds.length - 1) * vocabSize;
        final Float32List slice = logits.sublist(offset, offset + vocabSize);

        // 3e) True argmax + second‑best tracking
        int bestId = 0;
        double bestVal = slice[0];
        int secondId = -1;
        double secondVal = double.negativeInfinity;
        for (int i = 1; i < slice.length; i++) {
          final double v = slice[i];
          if (v > bestVal) {
            secondVal = bestVal;
            secondId = bestId;
            bestVal = v;
            bestId = i;
          } else if (v > secondVal && i != bestId) {
            secondVal = v;
            secondId = i;
          }
        }
        int nextId = bestId;

        // 3f) On step 0, if EOS is best, force second‑best instead
        if (step == 0 && nextId == endTokenId && secondId >= 0) {
          _debugLog('Ignored EOS at step 0 (logit=$bestVal); '
              'forcing token $secondId (logit=$secondVal)');
          nextId = secondId;
        }

        _debugLog('Step $step → token $nextId (${idToToken[nextId] ?? "?"})');

        // 3g) Cleanup decoder outputs
        for (final v in decOut) v?.release();
        idsTensor.release();

        // 3h) Stop conditions
        if (nextId == endTokenId) {
          _debugLog('EOS at step $step; breaking');
          break;
        }
        if (tokenIds.length >= maxRepeats &&
            tokenIds
                .sublist(tokenIds.length - maxRepeats)
                .every((t) => t == nextId)) {
          _debugLog('Runaway repeat $nextId; breaking');
          break;
        }

        // 3i) Append and continue
        tokenIds.add(nextId);
      }

      _debugLog('Decoding finished in ${stopwatch.elapsedMilliseconds}ms');

      // 4) Build final string, skipping BOS/EOS
      final result = tokenIds
          .where((id) => id != startTokenId && id != endTokenId)
          .map((id) => idToToken[id] ?? '')
          .join()
          .trim();

      _debugLog('OCR result: "$result"');
      return result;
    } finally {
      // 5) Release encoder resources
      encHidden.release();
      inputTensor.release();
    }
  }

  /// Helper function to check if two lists are equal
  bool listEquals<T>(List<T> a, List<T> b) {
    if (a.length != b.length) return false;
    for (var i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  /// Efficiently flatten a nested list structure into a Float32List
  Float32List _flattenNestedList(dynamic nestedList) {
    final flat = <double>[];

    // Use an iterative approach
    final stack = <dynamic>[nestedList];
    while (stack.isNotEmpty) {
      final current = stack.removeLast();

      if (current is List) {
        // Add items in reverse order to preserve original ordering
        for (var i = current.length - 1; i >= 0; i--) {
          stack.add(current[i]);
        }
      } else if (current is num) {
        flat.add(current.toDouble());
      } else if (current != null) {
        _debugLog(
          'Warning: Unexpected type in model output: ${current.runtimeType}',
        );
      }
    }

    if (flat.isEmpty) {
      throw Exception(
        'Failed to extract values from model output - empty result',
      );
    }

    return Float32List.fromList(flat);
  }

  /// Free resources
  void dispose() {
    if (_isDisposed) return;
    _debugLog('Disposing OCRService resources...');

    try {
      encoder.release();
      _debugLog('Encoder released');
    } catch (e) {
      print('Error releasing encoder: $e');
    }

    try {
      decoder.release();
      _debugLog('Decoder released');
    } catch (e) {
      print('Error releasing decoder: $e');
    }

    _isDisposed = true;
    _debugLog('OCRService disposed');
  }

  /// In OCRService:

  /// 1) Call at the end of `_preprocess`, before returning `out`:
  void _logPreprocessSample(Float32List out) {
    if (_debugMode) {
      print('Preprocess sample [0..9]: ${out.sublist(0, 10)}');
    }
  }

  /// 2) Call immediately after you grab `encHidden`:
  void _logEncoderHiddenSample(OrtValue encHidden) {
    if (!_debugMode) return;
    final v = encHidden.value;
    if (v is Float32List) {
      print(
        'EncHidden flat len=${v.length}, sample[0..9]: ${v.sublist(0, 10)}',
      );
    } else if (v is List) {
      print(
        'EncHidden nested; top length=${v.length}, '
        'first inner sample=${(v.first as List).sublist(0, 10)}',
      );
    } else {
      print('Unexpected encHidden type: ${v.runtimeType}');
    }
  }

  /// 3) Call inside the decode loop, after you compute `slice` and `nextId`:
  void _logDecoderSliceSample(Float32List slice, int nextId) {
    if (_debugMode) {
      print('Logits slice sample [0..9]: ${slice.sublist(0, 10)}');
      print('→ nextId=$nextId');
    }
  }
}

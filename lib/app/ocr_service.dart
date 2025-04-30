import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as img;
import 'package:onnxruntime/onnxruntime.dart';

class OCRService {
  // Private named constructor
  OCRService._(this._encoderSession, this._decoderSession, this.id2token);

  final OrtSession _encoderSession;
  final OrtSession _decoderSession;
  final Map<int, String> id2token;

  // ONE THING: async factory to load models from assets
  static Future<OCRService> create() async {
    final encData = await rootBundle.load('assets/models/encoder_model.onnx');
    final decData = await rootBundle.load('assets/models/decoder_model.onnx');
    final encoderSession = OrtSession.fromBuffer(
      encData.buffer.asUint8List(),
      OrtSessionOptions(),
    );

    final decoderSession = OrtSession.fromBuffer(
      decData.buffer.asUint8List(),
      OrtSessionOptions(),
    );
    print('Encoder inputs: ${encoderSession.inputNames}');
    print('Decoder inputs: ${decoderSession.inputNames}');

    final vocabJson = await rootBundle.loadString('assets/models/vocab.json');
    final vocabMap = jsonDecode(vocabJson) as Map<String, dynamic>;
    // build id→token map
    final id2token = <int, String>{
      for (final e in vocabMap.entries) (e.value as int): e.key,
    };
    return OCRService._(encoderSession, decoderSession, id2token);
  }

  Future<String> recognize(Uint8List imageBytes) async {
    // 1) Decode & resize to 384×384
    final image = img.decodeImage(imageBytes)!;
    final resized = img.copyResize(image, width: 384, height: 384);
    print('⚙️ Resized to ${resized.width}×${resized.height}');

// Assume you already have `resized` (384×384) and have imported `dart:typed_data`

    final w = resized.width, h = resized.height;
    final pixelCount = w * h;

// 1) Allocate three planes
    final rPlane = Float32List(pixelCount);
    final gPlane = Float32List(pixelCount);
    final bPlane = Float32List(pixelCount);

// 2) Fill & normalize each plane into [-1,1]
    var idx = 0;
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        final px = resized.getPixel(x, y);
        // cast .r/.g/.b (num) to double
        final rn = ((px.r.toDouble() / 255.0) - 0.5) / 0.5;
        final gn = ((px.g.toDouble() / 255.0) - 0.5) / 0.5;
        final bn = ((px.b.toDouble() / 255.0) - 0.5) / 0.5;

        rPlane[idx] = rn;
        gPlane[idx] = gn;
        bPlane[idx] = bn;
        idx++;
      }
    }

// 3) Concatenate into one [1,3,384,384] data buffer
    final inputData = Float32List(pixelCount * 3);
    inputData.setRange(0, pixelCount, rPlane);
    inputData.setRange(pixelCount, 2 * pixelCount, gPlane);
    inputData.setRange(2 * pixelCount, 3 * pixelCount, bPlane);

// 4) Then feed `inputData` into your encoder:
    final encInput = OrtValueTensor.createTensorWithDataList(
      inputData,
      [1, 3, h, w],
    );

    print('⚙️ Running encoder…');
    final encOutputs = await _encoderSession.runAsync(
      OrtRunOptions(),
      {'pixel_values': encInput},
    );
    final encoderStates = encOutputs![0]! as OrtValueTensor;
    final ev = encoderStates.value as List<List<List<double>>>;
    print(
        '⚙️ Encoder output shape = [${ev.length}, ${ev[0].length}, ${ev[0][0].length}]');
    print('⚙️ encoderStates[0][0][0..4] = ${ev[0][0].sublist(0, 5)}');

    // 5) Autoregressive decoding
    final bosId = id2token.entries.firstWhere((e) => e.value == '<s>').key;
    final eosId = id2token.entries.firstWhere((e) => e.value == '</s>').key;
    final decoded = <int>[bosId];
    final result = StringBuffer();
    const maxLen = 128;

    for (var step = 0; step < maxLen; step++) {
      final inputIdsTensor = OrtValueTensor.createTensorWithDataList(
        Int64List.fromList(decoded),
        [1, decoded.length],
      );
      final decOutputs = await _decoderSession.runAsync(
        OrtRunOptions(),
        {
          'input_ids': inputIdsTensor,
          'encoder_hidden_states': encoderStates,
        },
      );
      final nested =
          (decOutputs![0]! as OrtValueTensor).value as List<List<List<double>>>;
      final lastLogits = nested[0][decoded.length - 1];

      // pick highest logit
      var nextId = 0;
      for (var i = 1; i < lastLogits.length; i++) {
        if (lastLogits[i] > lastLogits[nextId]) nextId = i;
      }
      if (nextId == eosId) break;

      decoded.add(nextId);
      result.write(id2token[nextId]);
    }

    print('✅ Recognized: ${result.toString()}');
    print('🔍 Raw decoded IDs: $decoded');
    print('🔍 Raw tokens: ${decoded.map((i) => id2token[i]).toList()}');

    final finalText = decodeTokens(decoded, id2token);
    print('✅ Final text: $finalText');
    return finalText;
  }

  String decodeTokens(List<int> ids, Map<int, String> id2token) {
    final tokens = ids
        .map((i) => id2token[i]!)
        .where((t) => t != '<s>' && t != '</s>' && t != '<pad>')
        .toList();

    final buf = StringBuffer();
    String? lastWord;
    for (final t in tokens) {
      // Strip the leading Ġ marker, and treat it as a new word
      final isSpace = t.startsWith('Ġ');
      final word = isSpace ? t.substring(1) : t;

      // Skip if it’s the same as the last word (de-dup)
      if (word == lastWord) continue;

      if (isSpace) buf.write(' ');
      buf.write(word);
      lastWord = word;
    }

    return buf.toString().trim();
  }
}

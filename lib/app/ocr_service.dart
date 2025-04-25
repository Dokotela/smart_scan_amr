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
    final image = img.decodeImage(imageBytes)!;
    final resized = img.copyResize(image, width: 384, height: 384);
    final rgbBytes = resized.getBytes(order: img.ChannelOrder.rgb);
    final inputData =
        Float32List.fromList(rgbBytes.map((b) => b / 255.0).toList());
    final inputTensor = OrtValueTensor.createTensorWithDataList(
      inputData,
      [1, 3, 384, 384],
    );
    final runOptions = OrtRunOptions();
    final encoderOutputs = await _encoderSession.runAsync(
      runOptions,
      {'pixel_values': inputTensor},
    );
    if (encoderOutputs == null || encoderOutputs.isEmpty) {
      throw Exception('Encoder did not return any outputs');
    }
    final decoderOutputs = await _decoderSession.runAsync(
      runOptions,
      {'encoder_hidden_states': encoderOutputs[0]!},
    );
    final outputTensor = decoderOutputs![0]! as OrtValueTensor;
    final logits = outputTensor.value as Float32List;
    // assume logits shape is [1, seqLen, vocabSize]
    final vocabSize = id2token.length;
    final seqLen = logits.length ~/ vocabSize;

    // greedy decode
    final tokenIds = List<int>.generate(seqLen, (i) {
      final base = i * vocabSize;
      var maxVal = logits[base];
      var maxIdx = 0;
      for (var j = 1; j < vocabSize; j++) {
        if (logits[base + j] > maxVal) {
          maxVal = logits[base + j];
          maxIdx = j;
        }
      }
      return maxIdx;
    });

    // map IDs to tokens and concatenate
    final result = tokenIds.map((id) => id2token[id] ?? '').join();
    return result;
  }
}

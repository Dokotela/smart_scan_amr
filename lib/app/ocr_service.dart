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
    // 1) Image → encoder hidden states (unchanged)
    final image = img.decodeImage(imageBytes)!;
    final resized = img.copyResize(image, width: 384, height: 384);
    final rgbBytes = resized.getBytes(order: img.ChannelOrder.rgb);
    final inputData =
        Float32List.fromList(rgbBytes.map((b) => b / 255.0).toList());
    final encInput = OrtValueTensor.createTensorWithDataList(
      inputData,
      [1, 3, 384, 384],
    );
    final encOutputs = await _encoderSession.runAsync(
      OrtRunOptions(),
      {'pixel_values': encInput},
    );
    final encoderStates = encOutputs![0]! as OrtValueTensor;

    // 2) Prepare for decoder loop
    // Find your special token IDs (adjust if your vocab uses different strings)
    final bosId = id2token.entries.firstWhere((e) => e.value == '<s>').key;
    final eosId = id2token.entries.firstWhere((e) => e.value == '</s>').key;

    final decoded = [bosId];
    final result = StringBuffer();
    const maxLen = 128;

    for (var step = 0; step < maxLen; step++) {
      // 2a) Create input_ids tensor for current prefix
      final inputIdsTensor = OrtValueTensor.createTensorWithDataList(
        Int64List.fromList(decoded),
        [1, decoded.length],
      );

      // 2b) Run decoder with both inputs
      final decOutputs = await _decoderSession.runAsync(
        OrtRunOptions(),
        {
          'input_ids': inputIdsTensor,
          'encoder_hidden_states': encoderStates,
        },
      );
      final logitsTensor = decOutputs![0]! as OrtValueTensor;
      // 1) Cast down to the 3-D nested list
      final nested = logitsTensor.value as List<List<List<double>>>;

      // 2) Grab batch 0, then time-step (decoded.length-1)
      final lastLogits = nested[0][decoded.length - 1];
      // lastLogits is a List<double> of length == vocabSize

      // 3) Do your arg-max on that List<double>
      var nextId = 0;
      var maxLogit = lastLogits[0];
      for (var i = 1; i < lastLogits.length; i++) {
        if (lastLogits[i] > maxLogit) {
          maxLogit = lastLogits[i];
          nextId = i;
        }
      }

      if (nextId == eosId) break; // done
      decoded.add(nextId); // append to prefix
      result.write(id2token[nextId]); // append token text
    }

    print('Recognized: $result');
    return result.toString();
  }
}

import '../models/audio_file.dart';

class MockAudioData {
  static List<AudioFile> getSampleAudioFiles() {
    return [
      AudioFile(
        id: '1',
        name: 'Classical Symphony No. 1',
        length: const Duration(minutes: 4, seconds: 32),
        losslessMetadata: AudioMetadata(
          sampleRate: 48000,
          fileSizeBytes: 52.4 * 1024 * 1024, // 52.4 MB
          bitDepth: 24,
          bitRate: 1536,
          streamingTime: const Duration(milliseconds: 245),
        ),
        lossyMetadata: AudioMetadata(
          sampleRate: 44100,
          fileSizeBytes: 4.2 * 1024 * 1024, // 4.2 MB
          bitDepth: 16,
          bitRate: 320,
          streamingTime: const Duration(milliseconds: 89),
        ),
        enhancedMetadata: AudioMetadata(
          sampleRate: 48000,
          fileSizeBytes: 38.7 * 1024 * 1024, // 38.7 MB
          bitDepth: 24,
          bitRate: 1152,
          streamingTime: const Duration(milliseconds: 312),
        ),
      ),
      AudioFile(
        id: '2',
        name: 'Jazz Piano Improvisation',
        length: const Duration(minutes: 3, seconds: 18),
        losslessMetadata: AudioMetadata(
          sampleRate: 44100,
          fileSizeBytes: 34.8 * 1024 * 1024, // 34.8 MB
          bitDepth: 16,
          bitRate: 1411,
          streamingTime: const Duration(milliseconds: 198)
        ),
        lossyMetadata: AudioMetadata(
          sampleRate: 44100,
          fileSizeBytes: 3.1 * 1024 * 1024, // 3.1 MB
          bitDepth: 16,
          bitRate: 256,
          streamingTime: const Duration(milliseconds: 67)
        ),
        enhancedMetadata: AudioMetadata(
          sampleRate: 44100,
          fileSizeBytes: 28.9 * 1024 * 1024, // 28.9 MB
          bitDepth: 16,
          bitRate: 1152,
          streamingTime: const Duration(milliseconds: 287)
        ),
      ),
      AudioFile(
        id: '3',
        name: 'Rock Guitar Solo',
        length: const Duration(minutes: 2, seconds: 45),
        losslessMetadata: AudioMetadata(
          sampleRate: 96000,
          fileSizeBytes: 63.2 * 1024 * 1024, // 63.2 MB
          bitDepth: 24,
          bitRate: 2304,
          streamingTime: const Duration(milliseconds: 356)
        ),
        lossyMetadata: AudioMetadata(
          sampleRate: 44100,
          fileSizeBytes: 2.7 * 1024 * 1024, // 2.7 MB
          bitDepth: 16,
          bitRate: 320,
          streamingTime: const Duration(milliseconds: 72)
        ),
        enhancedMetadata: AudioMetadata(
          sampleRate: 48000,
          fileSizeBytes: 29.5 * 1024 * 1024, // 29.5 MB
          bitDepth: 24,
          bitRate: 1152,
          streamingTime: const Duration(milliseconds: 423)
        ),
      ),
      AudioFile(
        id: '4',
        name: 'Ambient Electronic',
        length: const Duration(minutes: 5, seconds: 12),
        losslessMetadata: AudioMetadata(
          sampleRate: 48000,
          fileSizeBytes: 59.8 * 1024 * 1024, // 59.8 MB
          bitDepth: 24,
          bitRate: 1536,
          streamingTime: const Duration(milliseconds: 287),
        ),
        lossyMetadata: AudioMetadata(
          sampleRate: 44100,
          fileSizeBytes: 4.9 * 1024 * 1024, // 4.9 MB
          bitDepth: 16,
          bitRate: 256,
          streamingTime: const Duration(milliseconds: 94),
        ),
        enhancedMetadata: AudioMetadata(
          sampleRate: 48000,
          fileSizeBytes: 44.3 * 1024 * 1024, // 44.3 MB
          bitDepth: 24,
          bitRate: 1152,
          streamingTime: const Duration(milliseconds: 367),
        ),
      ),
    ];
  }
}
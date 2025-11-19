class AudioFile {
  final String id;
  final String name;
  final Duration length;
  final AudioMetadata losslessMetadata;
  final AudioMetadata lossyMetadata;
  final AudioMetadata enhancedMetadata;

  AudioFile({
    required this.id,
    required this.name,
    required this.length,
    required this.losslessMetadata,
    required this.lossyMetadata,
    required this.enhancedMetadata,
  });

  String get formattedLength {
    final minutes = length.inMinutes;
    final seconds = length.inSeconds % 60;
    return '${minutes.toString().padLeft(2, '0')}:${seconds.toString().padLeft(2, '0')}';
  }
}

class AudioMetadata {
  final int sampleRate;
  final double fileSizeBytes;
  final int bitDepth;
  final int bitRate;
  final Duration? streamingTime;

  AudioMetadata({
    required this.sampleRate,
    required this.fileSizeBytes,
    required this.bitDepth,
    required this.bitRate,
    this.streamingTime,
  });

  String get formattedFileSize {
    if (fileSizeBytes < 1024) {
      return '${fileSizeBytes.toStringAsFixed(1)} B';
    } else if (fileSizeBytes < 1024 * 1024) {
      return '${(fileSizeBytes / 1024).toStringAsFixed(1)} KB';
    } else {
      return '${(fileSizeBytes / (1024 * 1024)).toStringAsFixed(1)} MB';
    }
  }

  String get formattedSampleRate {
    if (sampleRate >= 1000) {
      return '${(sampleRate / 1000).toStringAsFixed(1)} kHz';
    } else {
      return '$sampleRate Hz';
    }
  }

  String get formattedBitRate {
    if (bitRate >= 1000) {
      return '${(bitRate / 1000).toStringAsFixed(0)} kbps';
    } else {
      return '$bitRate bps';
    }
  }

  String? get formattedStreamingTime {
    if (streamingTime == null) return null;
    final ms = streamingTime!.inMilliseconds;
    if (ms < 1000) {
      return '${ms}ms';
    } else {
      return '${(ms / 1000).toStringAsFixed(2)}s';
    }
  }
}
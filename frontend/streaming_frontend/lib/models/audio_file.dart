class AudioFile {
  final String id;
  final String name;
  final Duration length;
  final AudioMetadata losslessMetadata;
  final AudioMetadata lossyMetadata;
  final AudioMetadata? enhancedMetadata;

  AudioFile({
    required this.id,
    required this.name,
    required this.length,
    required this.losslessMetadata,
    required this.lossyMetadata,
    required this.enhancedMetadata,
  });

  // Factory constructor to create AudioFile from API data
  factory AudioFile.fromApiData(Map<String, dynamic> apiData) {
    return AudioFile(
      id: apiData['_id'] ?? '',
      name: apiData['name'] ?? 'Unknown Track',
      length: _parseDuration(apiData['length']),
      losslessMetadata: AudioMetadata.fromApiData(apiData['lossless'] ?? {}),
      lossyMetadata: AudioMetadata.fromApiData(apiData['lossy'] ?? {}),
      enhancedMetadata: null, // Will be null for now
    );
  }

  static Duration _parseDuration(String? durationStr) {
    if (durationStr == null || durationStr.isEmpty) {
      return const Duration();
    }
    
    try {
      // Parse format like "00:02:07" (hours:minutes:seconds)
      final parts = durationStr.split(':');
      if (parts.length == 3) {
        final hours = int.parse(parts[0]);
        final minutes = int.parse(parts[1]);
        final seconds = int.parse(parts[2]);
        return Duration(hours: hours, minutes: minutes, seconds: seconds);
      } else if (parts.length == 2) {
        // Format like "02:07" (minutes:seconds)
        final minutes = int.parse(parts[0]);
        final seconds = int.parse(parts[1]);
        return Duration(minutes: minutes, seconds: seconds);
      }
    } catch (e) {
      print('Error parsing duration: $durationStr');
    }
    
    return const Duration();
  }

  String get formattedLength {
    final minutes = length.inMinutes;
    final seconds = length.inSeconds % 60;
    return '${minutes.toString().padLeft(2, '0')}:${seconds.toString().padLeft(2, '0')}';
  }
}

class AudioMetadata {
  final int sampleRate;
  final double fileSize;
  final int bitDepth;
  final int bitRate;

  AudioMetadata({
    required this.sampleRate,
    required this.fileSize,
    required this.bitDepth,
    required this.bitRate,
  });

  // Factory constructor to create AudioMetadata from API data
  factory AudioMetadata.fromApiData(Map<String, dynamic> apiData) {
    return AudioMetadata(
      sampleRate: _parseIntSafely(apiData['sample_rate']),
      fileSize: _parseFileSizeSafely(apiData['size']),
      bitDepth: _parseBitDepthSafely(apiData['bit_depth']),
      bitRate: _parseBitRateSafely(apiData['bit_rate']),
    );
  }

  static int _parseIntSafely(dynamic value) {
    if (value is int) return value;
    if (value is String) {
      try {
        return int.parse(value);
      } catch (e) {
        return 0;
      }
    }
    return 0;
  }

  static double _parseFileSizeSafely(dynamic value) {
    if (value == null) return 0.0;
    
    String sizeStr = value.toString().toLowerCase();
    
    // Extract numbers from strings like "91 MB", "1.89 MB", "937 KB"
    RegExp regex = RegExp(r'([\d.]+)\s*(mb|kb|gb|b)?');
    Match? match = regex.firstMatch(sizeStr);
    
    if (match != null) {
      double number = double.tryParse(match.group(1) ?? '0') ?? 0;
      String? unit = match.group(2);
      
      switch (unit?.toLowerCase()) {
        case 'gb':
          return number * 1024 * 1024 * 1024;
        case 'mb':
          return number * 1024 * 1024;
        case 'kb':
          return number * 1024;
        case 'b':
        default:
          return number;
      }
    }
    
    return 0.0;
  }

  static int _parseBitDepthSafely(dynamic value) {
    if (value == null) return 16; // Default bit depth
    
    String bitDepthStr = value.toString().toLowerCase();
    
    // Extract numbers from strings like "16 bit", "32 bit"
    RegExp regex = RegExp(r'(\d+)');
    Match? match = regex.firstMatch(bitDepthStr);
    
    if (match != null) {
      return int.tryParse(match.group(1) ?? '16') ?? 16;
    }
    
    return 16;
  }

  static int _parseBitRateSafely(dynamic value) {
    if (value == null) return 0;
    
    String bitRateStr = value.toString().toLowerCase();
    
    // Extract numbers from strings like "6144kbps", "124kbps"
    RegExp regex = RegExp(r'([\d.]+)\s*(kbps|bps|k)?');
    Match? match = regex.firstMatch(bitRateStr);
    
    if (match != null) {
      double number = double.tryParse(match.group(1) ?? '0') ?? 0;
      String? unit = match.group(2);
      
      switch (unit?.toLowerCase()) {
        case 'kbps':
        case 'k':
          return (number * 1000).round();
        case 'bps':
        default:
          return number.round();
      }
    }
    
    return 0;
  }

  String get formattedFileSize {
    if (fileSize < 1024) {
      return '${fileSize.toStringAsFixed(1)} B';
    } else if (fileSize < 1024 * 1024) {
      return '${(fileSize / 1024).toStringAsFixed(1)} KB';
    } else {
      return '${(fileSize / (1024 * 1024)).toStringAsFixed(1)} MB';
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

  /*String? get formattedStreamingTime {
    if (streamingTime == null) return null;
    final ms = streamingTime!.inMilliseconds;
    if (ms < 1000) {
      return '${ms}ms';
    } else {
      return '${(ms / 1000).toStringAsFixed(2)}s';
    }
  }*/
}
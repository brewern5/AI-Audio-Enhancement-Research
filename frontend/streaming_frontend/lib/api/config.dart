class AppConfig {
  static const String baseUrl = 'http://localhost:8000';
  static const Duration httpTimeout = Duration(seconds: 30);
  
  // Audio streaming endpoints
  static String getLossyStreamUrl(String songName) {
    final encodedName = Uri.encodeComponent(songName);
    return '$baseUrl/tracks/stream/lossy?name=$encodedName';
  }
  
  static String getLosslessStreamUrl(String songName) {
    final encodedName = Uri.encodeComponent(songName);
    return '$baseUrl/tracks/stream/lossless?name=$encodedName';
  }
  
  static String getEnhancedStreamUrl(String songName) {
    final encodedName = Uri.encodeComponent(songName);
    return '$baseUrl/tracks/stream/enhanced?name=$encodedName'; // Future endpoint
  }
}
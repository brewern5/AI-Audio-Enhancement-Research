import '../api/client.dart' as api;
import '../models/audio_file.dart';

class AudioMetadataManager {
  static List<AudioFile> getSampleAudioFiles = [];

  

  Future<List<dynamic>?> getAllAudioFiles() async {
    try {
      final response = await api.fetchNewest();

      if(response == null){
        return null;
      }
      
      List<dynamic> responseList = [];
      
      // Return the raw API response for now
      // You can transform this into AudioFile objects later if needed
      return response;
      
    } catch (e) {
      print('Error in getAllAudioFiles: $e');
      return null;
    }
  }
}
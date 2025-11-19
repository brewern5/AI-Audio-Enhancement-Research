import 'package:flutter/material.dart';
import '../data/audio_metadata_manager.dart';
import '../api/client.dart';
import '../models/audio_file.dart';
import 'audio_view_page.dart';

class LandingPage extends StatefulWidget {
  const LandingPage({super.key});
  
  @override
  State<LandingPage> createState() => _LandingPageState();
}

class _LandingPageState extends State<LandingPage> {
  List<dynamic>? _apiData;
  bool _isLoading = true;
  String? _errorMessage;
  
  @override
  void initState() {
    super.initState();
    _loadNewestData();
  }
  
  void _loadNewestData() async {
    try {
      setState(() {
        _isLoading = true;
        _errorMessage = null;
      });
      
      final result = await fetchNewest();
      if (result != null) {
        setState(() {
          _apiData = result;
          _isLoading = false;
        });
        print('Successfully loaded ${result.length} tracks from API');
      } else {
        setState(() {
          _isLoading = false;
          _errorMessage = 'No data received from API';
        });
      }
    } catch (e) {
      setState(() {
        _isLoading = false;
        _errorMessage = 'Error fetching data: $e';
      });
      print('Error fetching newest data: $e');
    }
  }
  
  List<dynamic> _getAudioFiles() {
    // If we have API data, use it; otherwise return empty list
    if (_apiData != null && _apiData!.isNotEmpty) {
      return _apiData!;
    } else {
      return [];
    }
  }
  
  @override
  Widget build(BuildContext context) {
    final audioFiles = _getAudioFiles();

    return Scaffold(
      appBar: AppBar(
        title: const Text('Audio Enhancement Research'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        elevation: 0,
      ),
      body: Column(
        children: [
          // Header section
          Container(
            width: double.infinity,
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              gradient: LinearGradient(
                colors: [
                  Theme.of(context).colorScheme.primary.withOpacity(0.1),
                  Theme.of(context).colorScheme.secondary.withOpacity(0.1),
                ],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Audio Quality Comparison',
                  style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  'Compare lossless, lossy, and AI-enhanced audio files',
                  style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                    color: Colors.grey[700],
                  ),
                ),
              ],
            ),
          ),
          // Content section
          Expanded(
            child: _buildContent(audioFiles),
          ),
        ],
      ),
    );
  }
  
  Widget _buildContent(List<dynamic> audioFiles) {
    if (_isLoading) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            CircularProgressIndicator(),
            SizedBox(height: 16),
            Text('Loading tracks...'),
          ],
        ),
      );
    }
    
    if (_errorMessage != null) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.error_outline,
              size: 48,
              color: Colors.red[400],
            ),
            const SizedBox(height: 16),
            Text(
              _errorMessage!,
              style: TextStyle(color: Colors.red[600]),
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 16),
            ElevatedButton(
              onPressed: _loadNewestData,
              child: const Text('Retry'),
            ),
          ],
        ),
      );
    }
    
    if (audioFiles.isEmpty) {
      return const Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(
              Icons.music_off,
              size: 48,
              color: Colors.grey,
            ),
            SizedBox(height: 16),
            Text('No tracks available'),
          ],
        ),
      );
    }
    
    return ListView.builder(
      padding: const EdgeInsets.symmetric(vertical: 16),
      itemCount: audioFiles.length,
      itemBuilder: (context, index) {
        final audioFile = audioFiles[index];
        return _buildTrackCard(audioFile, index);
      },
    );
  }
  
  Widget _buildTrackCard(dynamic trackData, int index) {
    // Handle both API data and AudioFile objects
    String trackName;
    String trackId;
    
    if (trackData is Map<String, dynamic>) {
      // API data
      trackName = trackData['name'] ?? 'Unknown Track';
      trackId = trackData['_id'] ?? index.toString();
    } else {
      // AudioFile object
      trackName = trackData.name ?? 'Unknown Track';
      trackId = trackData.id ?? index.toString();
    }
    
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
      child: ListTile(
        leading: const CircleAvatar(
          child: Icon(Icons.audiotrack),
        ),
        title: Text(trackName),
        subtitle: Text('Track ID: $trackId'),
        trailing: const Icon(Icons.play_arrow),
        onTap: () {
          try {
            // Convert API data to AudioFile if needed
            AudioFile audioFile;
            
            if (trackData is Map<String, dynamic>) {
              audioFile = AudioFile.fromApiData(trackData);
            } else if (trackData is AudioFile) {
              audioFile = trackData;
            } else {
              print('Unknown data type for track: $trackData');
              return;
            }
            
            // Navigate to audio view page
            Navigator.push(
              context,
              MaterialPageRoute(
                builder: (context) => AudioViewPage(audioFile: audioFile),
              ),
            );
          } catch (e) {
            print('Error navigating to audio view: $e');
            // Show error message to user
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text('Error loading track: $e'),
                backgroundColor: Colors.red,
              ),
            );
          }
        },
      ),
    );
  }
}
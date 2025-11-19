import 'package:flutter/material.dart';
import '../models/audio_file.dart';
import 'spectrogram_page.dart';

enum AudioType { lossless, lossy, enhanced }

class AudioViewPage extends StatefulWidget {
  final AudioFile audioFile;

  const AudioViewPage({super.key, required this.audioFile});

  @override
  State<AudioViewPage> createState() => _AudioViewPageState();
}

class _AudioViewPageState extends State<AudioViewPage> {
  AudioType? currentlyPlaying;
  bool isLoading = false;

  void _playAudio(AudioType type) {
    setState(() {
      if (currentlyPlaying == type) {
        currentlyPlaying = null; // Stop playing
      } else {
        isLoading = true;
        currentlyPlaying = type;
        // Simulate loading time
        Future.delayed(const Duration(milliseconds: 500), () {
          setState(() {
            isLoading = false;
          });
        });
      }
    });
  }

  AudioMetadata _getMetadata(AudioType type) {
    switch (type) {
      case AudioType.lossless:
        return widget.audioFile.losslessMetadata;
      case AudioType.lossy:
        return widget.audioFile.lossyMetadata;
      case AudioType.enhanced:
        return widget.audioFile.enhancedMetadata;
    }
  }

  String _getTypeLabel(AudioType type) {
    switch (type) {
      case AudioType.lossless:
        return 'Original Lossless';
      case AudioType.lossy:
        return 'Lossy Compressed';
      case AudioType.enhanced:
        return 'AI Enhanced';
    }
  }

  Color _getTypeColor(AudioType type) {
    switch (type) {
      case AudioType.lossless:
        return Colors.green;
      case AudioType.lossy:
        return Colors.orange;
      case AudioType.enhanced:
        return Colors.purple;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.audioFile.name),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Audio file header
            Card(
              elevation: 4,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      widget.audioFile.name,
                      style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        Icon(Icons.access_time, size: 16, color: Colors.grey[600]),
                        const SizedBox(width: 4),
                        Text(
                          'Duration: ${widget.audioFile.formattedLength}',
                          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                            color: Colors.grey[600],
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),
            
            // Audio player sections
            ...AudioType.values.map((type) => _buildAudioPlayerSection(type)),
            
            const SizedBox(height: 24),
            
            // Spectrogram button
            SizedBox(
              width: double.infinity,
              height: 56,
              child: ElevatedButton.icon(
                onPressed: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => SpectrogramPage(audioFile: widget.audioFile),
                    ),
                  );
                },
                icon: const Icon(Icons.graphic_eq),
                label: const Text('View Spectrograms'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Theme.of(context).colorScheme.secondary,
                  foregroundColor: Colors.white,
                  textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildAudioPlayerSection(AudioType type) {
    final metadata = _getMetadata(type);
    final isCurrentlyPlaying = currentlyPlaying == type;
    final typeColor = _getTypeColor(type);
    
    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header with play button
            Row(
              children: [
                Expanded(
                  child: Text(
                    _getTypeLabel(type),
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                      color: typeColor,
                    ),
                  ),
                ),
                if (isLoading && isCurrentlyPlaying)
                  const SizedBox(
                    width: 24,
                    height: 24,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                else
                  IconButton(
                    onPressed: () => _playAudio(type),
                    icon: Icon(
                      isCurrentlyPlaying ? Icons.pause_circle : Icons.play_circle,
                      size: 32,
                      color: typeColor,
                    ),
                  ),
              ],
            ),
            const SizedBox(height: 16),
            
            // Metadata grid
            GridView.count(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              crossAxisCount: 2,
              childAspectRatio: 2.5,
              crossAxisSpacing: 16,
              mainAxisSpacing: 12,
              children: [
                _buildMetadataItem('Sample Rate', metadata.formattedSampleRate),
                _buildMetadataItem('File Size', metadata.formattedFileSize),
                _buildMetadataItem('Bit Depth', '${metadata.bitDepth} bit'),
                _buildMetadataItem('Bit Rate', metadata.formattedBitRate),
                if (metadata.formattedStreamingTime != null)
                  _buildMetadataItem('Streaming Time', metadata.formattedStreamingTime!),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildMetadataItem(String label, String value) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: Theme.of(context).textTheme.bodySmall?.copyWith(
            color: Colors.grey[600],
            fontWeight: FontWeight.w500,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          value,
          style: Theme.of(context).textTheme.bodyMedium?.copyWith(
            fontWeight: FontWeight.bold,
          ),
        ),
      ],
    );
  }
}
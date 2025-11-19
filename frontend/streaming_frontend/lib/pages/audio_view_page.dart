import 'package:flutter/material.dart';
import '../models/audio_file.dart';
import '../widgets/audio_player.dart';
import 'spectrogram_page.dart';

enum AudioType { lossless, lossy, enhanced }

class AudioViewPage extends StatefulWidget {
  final AudioFile audioFile;

  const AudioViewPage({super.key, required this.audioFile});

  @override
  State<AudioViewPage> createState() => _AudioViewPageState();
}

class _AudioViewPageState extends State<AudioViewPage> {
  AudioMetadata? _getMetadata(AudioType type) {
    switch (type) {
      case AudioType.lossless:
        return widget.audioFile.losslessMetadata;
      case AudioType.lossy:
        return widget.audioFile.lossyMetadata;
      case AudioType.enhanced:
        return widget.audioFile.enhancedMetadata; // Can be null
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
            ...AudioType.values
                .where((type) => type != AudioType.enhanced || widget.audioFile.enhancedMetadata != null)
                .map((type) => _buildAudioPlayerSection(type)),
            
            // Enhanced audio section (if available)
            if (widget.audioFile.enhancedMetadata != null)
              _buildEnhancedAudioSection(),
            
            // Show placeholder for enhanced audio if not available
            if (widget.audioFile.enhancedMetadata == null)
              _buildEnhancedPlaceholder(),
            
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
    final typeColor = _getTypeColor(type);
    
    // Return empty container if metadata is null (should not happen after filtering)
    if (metadata == null) {
      return const SizedBox.shrink();
    }
    
    // Convert AudioType to string for API
    String audioTypeString = type == AudioType.lossless ? 'lossless' : 'lossy';
    
    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header
            Text(
              _getTypeLabel(type),
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
                color: typeColor,
              ),
            ),
            const SizedBox(height: 16),
            
            // Audio Player
            AudioPlayerWidget(
              songName: widget.audioFile.name,
              audioType: audioTypeString,
              totalDuration: widget.audioFile.length,
              themeColor: typeColor,
            ),
            
            const SizedBox(height: 8),
            
            // Metadata grid
            GridView.count(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              crossAxisCount: 4,
              childAspectRatio: 3.5,
              crossAxisSpacing: 4,
              mainAxisSpacing: 2,
              children: [
                _buildMetadataItem('Sample Rate', metadata.formattedSampleRate),
                _buildMetadataItem('File Size', metadata.formattedFileSize),
                _buildMetadataItem('Bit Depth', '${metadata.bitDepth} bit'),
                _buildMetadataItem('Bit Rate', metadata.formattedBitRate),
              ]
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
          style: Theme.of(context).textTheme.labelSmall?.copyWith(
            color: Colors.grey[600],
            fontWeight: FontWeight.w500,
            fontSize: 10,
          ),
        ),
        const SizedBox(height: 1),
        Text(
          value,
          style: Theme.of(context).textTheme.bodySmall?.copyWith(
            fontWeight: FontWeight.bold,
            fontSize: 12,
          ),
        ),
      ],
    );
  }

  Widget _buildEnhancedAudioSection() {
    final metadata = widget.audioFile.enhancedMetadata;
    if (metadata == null) return const SizedBox.shrink();
    
    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Header with sparkle icon
            Row(
              children: [
                Expanded(
                  child: Text(
                    'AI-Enhanced Audio',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                      color: Colors.purple,
                    ),
                  ),
                ),
                Icon(
                  Icons.auto_awesome,
                  color: Colors.purple,
                  size: 24,
                ),
              ],
            ),
            const SizedBox(height: 16),
            
            // Audio Player
            AudioPlayerWidget(
              songName: widget.audioFile.name,
              audioType: 'enhanced', // This might need backend support
              totalDuration: widget.audioFile.length,
              themeColor: Colors.purple,
            ),
            
            const SizedBox(height: 8),
            
            // Metadata grid
            GridView.count(
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              crossAxisCount: 2,
              childAspectRatio: 3.5,
              crossAxisSpacing: 4,
              mainAxisSpacing: 2,
              children: [
                _buildMetadataItem('Sample Rate', metadata.formattedSampleRate),
                _buildMetadataItem('File Size', metadata.formattedFileSize),
                _buildMetadataItem('Bit Depth', '${metadata.bitDepth} bit'),
                _buildMetadataItem('Bit Rate', metadata.formattedBitRate),
              ]
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildEnhancedPlaceholder() {
    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Expanded(
                  child: Text(
                    'AI-Enhanced Audio',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                      color: Colors.grey[600],
                    ),
                  ),
                ),
                Icon(
                  Icons.auto_awesome_outlined,
                  color: Colors.grey[400],
                  size: 32,
                ),
              ],
            ),
            const SizedBox(height: 16),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                color: Colors.grey[100],
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.grey[300]!),
              ),
              child: Column(
                children: [
                  Icon(
                    Icons.construction_outlined,
                    size: 48,
                    color: Colors.grey[400],
                  ),
                  const SizedBox(height: 12),
                  Text(
                    'AI-Enhanced Version Not Available',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      color: Colors.grey[600],
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Text(
                    'This track has not been processed through our AI enhancement pipeline yet.',
                    style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: Colors.grey[600],
                    ),
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
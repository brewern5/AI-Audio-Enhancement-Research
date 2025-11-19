import 'package:flutter/material.dart';
import '../widgets/song_widget.dart';
import '../data/mock_audio_data.dart';
import 'audio_view_page.dart';

class LandingPage extends StatelessWidget {
  const LandingPage({super.key});

  @override
  Widget build(BuildContext context) {
    final audioFiles = MockAudioData.getSampleAudioFiles();

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
          // Songs list
          Expanded(
            child: ListView.builder(
              padding: const EdgeInsets.symmetric(vertical: 16),
              itemCount: audioFiles.length,
              itemBuilder: (context, index) {
                final audioFile = audioFiles[index];
                return SongWidget(
                  audioFile: audioFile,
                  onTap: () {
                    Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (context) => AudioViewPage(audioFile: audioFile),
                      ),
                    );
                  },
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}
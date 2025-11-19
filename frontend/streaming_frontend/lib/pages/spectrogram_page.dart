import 'package:flutter/material.dart';
import '../models/audio_file.dart';

class SpectrogramPage extends StatefulWidget {
  final AudioFile audioFile;

  const SpectrogramPage({super.key, required this.audioFile});

  @override
  State<SpectrogramPage> createState() => _SpectrogramPageState();
}

class _SpectrogramPageState extends State<SpectrogramPage> with TickerProviderStateMixin {
  late TabController _tabController;
  bool isLoading = true;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
    // Simulate loading spectrograms
    Future.delayed(const Duration(seconds: 2), () {
      setState(() {
        isLoading = false;
      });
    });
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('${widget.audioFile.name} - Spectrograms'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        bottom: TabBar(
          controller: _tabController,
          labelColor: Theme.of(context).primaryColor,
          unselectedLabelColor: Colors.grey,
          indicatorColor: Theme.of(context).primaryColor,
          tabs: const [
            Tab(text: 'Original Lossless'),
            Tab(text: 'Lossy Compressed'),
            Tab(text: 'AI Enhanced'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          _buildSpectrogramView('lossless', Colors.green),
          _buildSpectrogramView('lossy', Colors.orange),
          _buildSpectrogramView('enhanced', Colors.purple),
        ],
      ),
    );
  }

  Widget _buildSpectrogramView(String type, Color color) {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Spectrogram container
          Container(
            width: double.infinity,
            height: 300,
            decoration: BoxDecoration(
              border: Border.all(color: Colors.grey[300]!),
              borderRadius: BorderRadius.circular(8),
              color: Colors.grey[50],
            ),
            child: isLoading
                ? const Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        CircularProgressIndicator(),
                        SizedBox(height: 16),
                        Text('Loading spectrogram...'),
                      ],
                    ),
                  )
                : _buildSpectrogramContent(type, color),
          ),
          const SizedBox(height: 24),
          
          // Frequency analysis section
          Card(
            elevation: 2,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.analytics, color: color),
                      const SizedBox(width: 8),
                      Text(
                        'Frequency Analysis',
                        style: Theme.of(context).textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 16),
                  _buildAnalysisGrid(type),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          
          // Interactive controls
          Card(
            elevation: 2,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Spectrogram Controls',
                    style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 16),
                  _buildControlsSection(),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSpectrogramContent(String type, Color color) {
    return Stack(
      children: [
        // Simulated spectrogram pattern
        Container(
          width: double.infinity,
          height: double.infinity,
          child: CustomPaint(
            painter: SpectrogramPainter(color: color, type: type),
          ),
        ),
        // Frequency labels (Y-axis)
        Positioned(
          left: 8,
          top: 16,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildFrequencyLabel('20kHz'),
              const SizedBox(height: 40),
              _buildFrequencyLabel('15kHz'),
              const SizedBox(height: 40),
              _buildFrequencyLabel('10kHz'),
              const SizedBox(height: 40),
              _buildFrequencyLabel('5kHz'),
              const SizedBox(height: 40),
              _buildFrequencyLabel('1kHz'),
              const SizedBox(height: 40),
              _buildFrequencyLabel('100Hz'),
            ],
          ),
        ),
        // Time labels (X-axis)
        Positioned(
          bottom: 8,
          left: 40,
          right: 16,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              _buildTimeLabel('0s'),
              _buildTimeLabel('1s'),
              _buildTimeLabel('2s'),
              _buildTimeLabel('3s'),
              _buildTimeLabel('4s'),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildFrequencyLabel(String frequency) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.7),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        frequency,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 10,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }

  Widget _buildTimeLabel(String time) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(0.7),
        borderRadius: BorderRadius.circular(4),
      ),
      child: Text(
        time,
        style: const TextStyle(
          color: Colors.white,
          fontSize: 10,
          fontWeight: FontWeight.bold,
        ),
      ),
    );
  }

  Widget _buildAnalysisGrid(String type) {
    // Mock data based on audio type
    Map<String, String> analysisData;
    switch (type) {
      case 'lossless':
        analysisData = {
          'Frequency Range': '20Hz - 20kHz',
          'Dynamic Range': '96dB',
          'Peak Frequency': '2.5kHz',
          'Harmonic Content': 'Rich',
        };
        break;
      case 'lossy':
        analysisData = {
          'Frequency Range': '20Hz - 16kHz',
          'Dynamic Range': '72dB',
          'Peak Frequency': '2.1kHz',
          'Harmonic Content': 'Reduced',
        };
        break;
      case 'enhanced':
        analysisData = {
          'Frequency Range': '20Hz - 22kHz',
          'Dynamic Range': '108dB',
          'Peak Frequency': '2.7kHz',
          'Harmonic Content': 'Enhanced',
        };
        break;
      default:
        analysisData = {};
    }

    return GridView.count(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      crossAxisCount: 2,
      childAspectRatio: 2.5,
      crossAxisSpacing: 16,
      mainAxisSpacing: 12,
      children: analysisData.entries.map((entry) {
        return _buildAnalysisItem(entry.key, entry.value);
      }).toList(),
    );
  }

  Widget _buildAnalysisItem(String label, String value) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.grey[100],
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisAlignment: MainAxisAlignment.center,
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
      ),
    );
  }

  Widget _buildControlsSection() {
    return Column(
      children: [
        Row(
          children: [
            Expanded(
              child: ElevatedButton.icon(
                onPressed: () {
                  // Implement zoom functionality
                },
                icon: const Icon(Icons.zoom_in, size: 18),
                label: const Text('Zoom In'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue[100],
                  foregroundColor: Colors.blue[800],
                ),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: ElevatedButton.icon(
                onPressed: () {
                  // Implement zoom out functionality
                },
                icon: const Icon(Icons.zoom_out, size: 18),
                label: const Text('Zoom Out'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.blue[100],
                  foregroundColor: Colors.blue[800],
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: () {
              // Implement export functionality
            },
            icon: const Icon(Icons.download, size: 18),
            label: const Text('Export Spectrogram'),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.green[100],
              foregroundColor: Colors.green[800],
            ),
          ),
        ),
      ],
    );
  }
}

class SpectrogramPainter extends CustomPainter {
  final Color color;
  final String type;

  SpectrogramPainter({required this.color, required this.type});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint();
    
    // Create a gradient effect for the spectrogram
    final gradient = LinearGradient(
      begin: Alignment.topCenter,
      end: Alignment.bottomCenter,
      colors: [
        color.withOpacity(0.8),
        color.withOpacity(0.3),
        Colors.transparent,
      ],
    );

    final rect = Rect.fromLTWH(0, 0, size.width, size.height);
    paint.shader = gradient.createShader(rect);

    // Draw simulated frequency bands
    for (int i = 0; i < 50; i++) {
      final x = (i / 50) * size.width;
      final height = (i % 10 + 1) * size.height / 15;
      final y = size.height - height;
      
      // Vary intensity based on type
      final intensity = type == 'enhanced' ? 1.2 : (type == 'lossy' ? 0.7 : 1.0);
      paint.color = color.withOpacity(0.3 * intensity);
      
      canvas.drawRect(
        Rect.fromLTWH(x, y, size.width / 60, height),
        paint,
      );
    }

    // Add some noise pattern for realism
    paint.color = Colors.black.withOpacity(0.1);
    for (int i = 0; i < 100; i++) {
      canvas.drawCircle(
        Offset(
          (i * 17) % size.width,
          (i * 23) % size.height,
        ),
        1,
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
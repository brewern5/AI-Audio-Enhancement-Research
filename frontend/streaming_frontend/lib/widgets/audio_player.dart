import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import '../api/config.dart';
import 'dart:async';
// ignore: avoid_web_libraries_in_flutter
import 'dart:html' as html;

class AudioPlayerWidget extends StatefulWidget {
  final String songName;
  final String audioType; // 'lossy' or 'lossless'
  final Duration totalDuration;
  final Color themeColor;

  const AudioPlayerWidget({
    Key? key,
    required this.songName,
    required this.audioType,
    required this.totalDuration,
    required this.themeColor,
  }) : super(key: key);

  @override
  State<AudioPlayerWidget> createState() => _AudioPlayerWidgetState();
}

class _AudioPlayerWidgetState extends State<AudioPlayerWidget> {
  html.AudioElement? _audioElement;
  bool _isPlaying = false;
  bool _isLoading = false;
  Duration _currentPosition = Duration.zero;
  Duration _totalDuration = Duration.zero;
  Timer? _positionTimer;

  @override
  void initState() {
    super.initState();
    _totalDuration = widget.totalDuration;
    if (kIsWeb) {
      _setupWebAudio();
    }
  }

  void _setupWebAudio() {
    _audioElement = html.AudioElement();
    _audioElement!.preload = 'metadata';
    
    // Set up event listeners
    _audioElement!.onLoadedMetadata.listen((_) {
      if (mounted) {
        setState(() {
          final duration = _audioElement!.duration;
          if (duration != null && duration.isFinite) {
            _totalDuration = Duration(seconds: duration.round());
          }
          _isLoading = false;
        });
      }
    });

    _audioElement!.onTimeUpdate.listen((_) {
      if (mounted) {
        setState(() {
          final currentTime = _audioElement!.currentTime;
          if (currentTime != null && currentTime.isFinite) {
            _currentPosition = Duration(seconds: currentTime.round());
          }
        });
      }
    });

    _audioElement!.onPlay.listen((_) {
      if (mounted) {
        setState(() {
          _isPlaying = true;
        });
      }
    });

    _audioElement!.onPause.listen((_) {
      if (mounted) {
        setState(() {
          _isPlaying = false;
        });
      }
    });

    _audioElement!.onEnded.listen((_) {
      if (mounted) {
        setState(() {
          _isPlaying = false;
          _currentPosition = Duration.zero;
        });
      }
    });

    _audioElement!.onLoadedData.listen((_) {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    });

    _audioElement!.onWaiting.listen((_) {
      if (mounted) {
        setState(() {
          _isLoading = true;
        });
      }
    });

    _audioElement!.onError.listen((event) {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
        _showErrorSnackBar('Failed to load audio stream');
      }
    });
  }

  String _getStreamUrl() {
    switch (widget.audioType) {
      case 'lossy':
        return AppConfig.getLossyStreamUrl(widget.songName);
      case 'lossless':
        return AppConfig.getLosslessStreamUrl(widget.songName);
      case 'enhanced':
        return AppConfig.getEnhancedStreamUrl(widget.songName);
      default:
        return AppConfig.getLossyStreamUrl(widget.songName);
    }
  }

  Future<void> _togglePlayPause() async {
    try {
      if (kIsWeb && _audioElement != null) {
        if (_isPlaying) {
          _audioElement!.pause();
        } else {
          if (_audioElement!.src.isEmpty) {
            final streamUrl = _getStreamUrl();
            print('Attempting to load audio from: $streamUrl');
            if (mounted) {
              setState(() {
                _isLoading = true;
              });
            }
            _audioElement!.src = streamUrl;
          }
          await _audioElement!.play();
        }
      }
    } catch (e) {
      print('Error playing audio: $e');
      print('Audio element src: ${_audioElement?.src}');
      _showErrorSnackBar('Failed to play audio: $e');
    }
  }

  Future<void> _seek(Duration position) async {
    if (kIsWeb && _audioElement != null) {
      final seekTime = position.inSeconds.toDouble();
      if (seekTime.isFinite && seekTime >= 0) {
        _audioElement!.currentTime = seekTime;
      }
    }
  }

  void _showErrorSnackBar(String message) {
    if (mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(message),
          backgroundColor: Colors.red,
        ),
      );
    }
  }

  String _formatDuration(Duration duration) {
    String twoDigits(int n) => n.toString().padLeft(2, "0");
    String twoDigitMinutes = twoDigits(duration.inMinutes.remainder(60));
    String twoDigitSeconds = twoDigits(duration.inSeconds.remainder(60));
    
    if (duration.inHours > 0) {
      return "${twoDigits(duration.inHours)}:$twoDigitMinutes:$twoDigitSeconds";
    }
    return "$twoDigitMinutes:$twoDigitSeconds";
  }

  @override
  Widget build(BuildContext context) {
    // Show web-only message if not on web
    if (!kIsWeb) {
      return Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.grey[100],
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.grey[300]!),
        ),
        child: Column(
          children: [
            Icon(
              Icons.info_outline,
              color: Colors.grey[600],
              size: 48,
            ),
            const SizedBox(height: 8),
            Text(
              'Audio streaming is currently only available on web',
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                color: Colors.grey[600],
              ),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      );
    }

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.grey[50],
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.grey[200]!),
      ),
      child: Column(
        children: [
          // Play/Pause Button and Track Info
          Row(
            children: [
              // Play/Pause Button
              Container(
                decoration: BoxDecoration(
                  color: widget.themeColor,
                  borderRadius: BorderRadius.circular(25),
                ),
                child: _isLoading
                    ? Container(
                        width: 50,
                        height: 50,
                        padding: const EdgeInsets.all(12),
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                        ),
                      )
                    : IconButton(
                        onPressed: _togglePlayPause,
                        icon: Icon(
                          _isPlaying ? Icons.pause : Icons.play_arrow,
                          color: Colors.white,
                          size: 30,
                        ),
                      ),
              ),
              const SizedBox(width: 16),
              // Track Info
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      widget.songName,
                      style: Theme.of(context).textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '${widget.audioType.toUpperCase()} Quality',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: widget.themeColor,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ),
              // Time Display
              Text(
                '${_formatDuration(_currentPosition)} / ${_formatDuration(_totalDuration)}',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: Colors.grey[600],
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          // Progress Bar
          Column(
            children: [
              SliderTheme(
                data: SliderTheme.of(context).copyWith(
                  trackHeight: 4,
                  thumbShape: RoundSliderThumbShape(enabledThumbRadius: 8),
                  overlayShape: RoundSliderOverlayShape(overlayRadius: 16),
                  activeTrackColor: widget.themeColor,
                  inactiveTrackColor: Colors.grey[300],
                  thumbColor: widget.themeColor,
                  overlayColor: widget.themeColor.withOpacity(0.2),
                ),
                child: Slider(
                  value: _totalDuration.inSeconds > 0
                      ? _currentPosition.inSeconds.toDouble().clamp(
                          0.0, _totalDuration.inSeconds.toDouble())
                      : 0.0,
                  max: _totalDuration.inSeconds.toDouble(),
                  onChanged: (value) {
                    _seek(Duration(seconds: value.round()));
                  },
                ),
              ),
              // Time markers
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 24),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      _formatDuration(_currentPosition),
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Colors.grey[600],
                      ),
                    ),
                    Text(
                      _formatDuration(_totalDuration),
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Colors.grey[600],
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          // Stream URL info (for debugging)
          if (true) // Set to true for debugging
            Text(
              'Stream URL: ${_getStreamUrl()}',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                color: Colors.grey[500],
                fontSize: 10,
              ),
            ),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _positionTimer?.cancel();
    if (kIsWeb && _audioElement != null) {
      // Stop the audio and clear all event listeners
      _audioElement!.pause();
      _audioElement!.src = '';
      _audioElement!.load(); // This helps clear the audio element state
    }
    super.dispose();
  }
}
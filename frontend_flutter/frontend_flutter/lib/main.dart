import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui; // 需要引入 ui 库以使用 ui.Image
import 'dart:async'; // 需要引入 async 库以使用 Completer

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:path_provider/path_provider.dart';
import 'package:share_plus/share_plus.dart';

// ===================================================================
// 1. 数据模型 (Data Models)
// ===================================================================

class BoundingBox {
  final double x1, y1, x2, y2;
  final double confidence;

  BoundingBox({
    required this.x1,
    required this.y1,
    required this.x2,
    required this.y2,
    required this.confidence,
  });

  factory BoundingBox.fromJson(Map<String, dynamic> json) {
    return BoundingBox(
      x1: (json['x1'] ?? 0).toDouble(),
      y1: (json['y1'] ?? 0).toDouble(),
      x2: (json['x2'] ?? 0).toDouble(),
      y2: (json['y2'] ?? 0).toDouble(),
      confidence: (json['conf'] ?? 0).toDouble(),
    );
  }
}

class ImageSlice {
  final String imageUrl;
  final List<BoundingBox> bboxes;

  ImageSlice({required this.imageUrl, required this.bboxes});

  factory ImageSlice.fromJson(Map<String, dynamic> json) {
    var bboxesList = (json['bboxes'] as List?)?.map((bboxJson) => BoundingBox.fromJson(bboxJson)).toList() ?? [];
    return ImageSlice(
      imageUrl: json['image_url'] ?? '',
      bboxes: bboxesList,
    );
  }
}

class DetectionResult {
  final List<ImageSlice> slices;
  final String report;

  DetectionResult({required this.slices, required this.report});

  factory DetectionResult.fromJson(Map<String, dynamic> json) {
    var slicesList = (json['slices'] as List?)?.map((sliceJson) => ImageSlice.fromJson(sliceJson)).toList() ?? [];
    return DetectionResult(
      slices: slicesList,
      report: json['report'] ?? '无检测报告数据',
    );
  }
}


// ===================================================================
// 2. 主程序入口
// ===================================================================

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '肺结节智能检测',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
        useMaterial3: true,
      ),
      home: const UploadPage(),
    );
  }
}

// ===================================================================
// 3. 上传页面 (UploadPage)
// ===================================================================

class UploadPage extends StatefulWidget {
  const UploadPage({super.key});

  @override
  State<UploadPage> createState() => _UploadPageState();
}

class _UploadPageState extends State<UploadPage> {
  final ImagePicker _picker = ImagePicker();
  List<XFile> _imageFiles = [];
  bool _isLoading = false;
  String? _errorMessage;

  Future<void> _pickImages() async {
    try {
      final pickedFiles = await _picker.pickMultiImage(imageQuality: 80);
      if (pickedFiles.isNotEmpty) {
        setState(() {
          _imageFiles = pickedFiles;
          _errorMessage = null;
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = '选择图片失败: $e';
      });
    }
  }

  Future<void> _submit() async {
    if (_imageFiles.isEmpty) {
      setState(() {
        _errorMessage = '请先选择图片';
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      var uri = Uri.parse('http://localhost:5000/api/detect');
      var request = http.MultipartRequest('POST', uri);

      for (final file in _imageFiles) {
        if (kIsWeb) {
          request.files.add(http.MultipartFile.fromBytes(
            'images',
            await file.readAsBytes(),
            filename: file.name,
          ));
        } else {
          request.files.add(await http.MultipartFile.fromPath(
            'images',
            file.path,
            filename: file.name,
          ));
        }
      }

      final streamedResponse = await request.send();
      
      if (streamedResponse.statusCode == 200) {
        final responseBody = await streamedResponse.stream.bytesToString();
        final data = json.decode(responseBody);
        final result = DetectionResult.fromJson(data);

        if (mounted) {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => ResultPage(result: result),
            ),
          );
        }
      } else {
        throw Exception('服务器错误: ${streamedResponse.statusCode}');
      }
    } catch (e) {
      setState(() {
        _errorMessage = '提交失败: $e';
      });
    } finally {
      if(mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('上传 CT 切片')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            ElevatedButton.icon(
              icon: const Icon(Icons.photo_library),
              label: const Text('选择CT切片图片'),
              onPressed: _isLoading ? null : _pickImages,
              style: ElevatedButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 12)),
            ),
            const SizedBox(height: 10),
            if (_errorMessage != null)
              Text(_errorMessage!, style: const TextStyle(color: Colors.red), textAlign: TextAlign.center),
            Expanded(
              child: _imageFiles.isEmpty
                  ? const Center(child: Text('请选择CT切片图片进行检测'))
                  : GridView.builder(
                      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                        crossAxisCount: 3,
                        crossAxisSpacing: 8,
                        mainAxisSpacing: 8,
                      ),
                      itemCount: _imageFiles.length,
                      itemBuilder: (context, index) {
                        final file = _imageFiles[index];
                        return kIsWeb
                            ? Image.network(file.path, fit: BoxFit.cover)
                            : Image.file(File(file.path), fit: BoxFit.cover);
                      },
                    ),
            ),
            if (_imageFiles.isNotEmpty)
              ElevatedButton(
                onPressed: _isLoading ? null : _submit,
                style: ElevatedButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 16)),
                child: _isLoading
                    ? const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(color: Colors.white, strokeWidth: 3))
                    : const Text('开始肺结节检测'),
              ),
          ],
        ),
      ),
    );
  }
}

// ===================================================================
// 4. 结果页面 (ResultPage)
// ===================================================================

class ResultPage extends StatefulWidget {
  final DetectionResult result;
  const ResultPage({required this.result, super.key});

  @override
  State<ResultPage> createState() => _ResultPageState();
}

class _ResultPageState extends State<ResultPage> {
  late final PageController _pageController;
  int _currentIndex = 0;
  bool _isGeneratingSuggestion = false;
  String? _suggestionText;

  @override
  void initState() {
    super.initState();
    _pageController = PageController();
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }
  
  Future<void> _getAiSuggestion() async {
    setState(() {
      _isGeneratingSuggestion = true;
      _suggestionText = null;
    });
    try {
      await Future.delayed(const Duration(seconds: 2));
      const suggestion = "模拟AI建议：根据检测报告，建议您咨询呼吸科专家，并进行定期复查。请保持健康的生活方式。";
      setState(() {
        _suggestionText = suggestion;
      });
    } catch (e) {
      setState(() {
        _suggestionText = "获取AI建议失败: $e";
      });
    } finally {
      setState(() {
        _isGeneratingSuggestion = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final slices = widget.result.slices;
    if (slices.isEmpty) {
      return Scaffold(
        appBar: AppBar(title: const Text('检测结果')),
        body: const Center(child: Text('未获取到有效的检测数据')),
      );
    }
    final currentSlice = slices[_currentIndex];
    final screenHeight = MediaQuery.of(context).size.height;

    return Scaffold(
      appBar: AppBar(
        title: Text('检测结果 (${_currentIndex + 1}/${slices.length})'),
        actions: [
          IconButton(
            icon: const Icon(Icons.share),
            onPressed: () => Share.share(widget.result.report, subject: '肺结节检测报告'),
          ),
        ],
      ),
      body: Column(
        children: [
          // ==========================================================
          //
          //              ↓↓↓ 这里是核心修改 ↓↓↓
          //
          //  1. 用 Padding 增加边距
          //  2. 用 SizedBox 限制高度为屏幕高度的 40%
          //  3. 用 Card 增加卡片阴影效果，使其看起来像预览卡片
          //
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: SizedBox(
              height: screenHeight * 0.4, // <-- 你可以在这里调整大小 (比如 0.35, 0.5)
              child: Card(
                clipBehavior: Clip.antiAlias, // 让图片圆角生效
                elevation: 4.0, // 卡片阴影
                child: PageView.builder(
                  controller: _pageController,
                  itemCount: slices.length,
                  onPageChanged: (index) {
                    setState(() {
                      _currentIndex = index;
                    });
                  },
                  itemBuilder: (context, index) {
                    return BoundingBoxImage(slice: slices[index]);
                  },
                ),
              ),
            ),
          ),
          //
          //              ↑↑↑ 这里是核心修改 ↑↑↑
          //
          // ==========================================================
          
          Expanded(
            child: DefaultTabController(
              length: 2,
              child: Column(
                children: [
                   const TabBar(
                    tabs: [
                      Tab(text: '结节详情'),
                      Tab(text: '综合报告'),
                    ],
                  ),
                  Expanded(
                    child: TabBarView(
                      children: [
                        currentSlice.bboxes.isEmpty
                          ? const Center(child: Text('当前切片未检测到肺结节'))
                          : ListView.builder(
                              padding: const EdgeInsets.all(8.0),
                              itemCount: currentSlice.bboxes.length,
                              itemBuilder: (context, index) {
                                final box = currentSlice.bboxes[index];
                                return Card(
                                  child: ListTile(
                                    leading: CircleAvatar(child: Text('${index + 1}')),
                                    title: Text('置信度: ${(box.confidence * 100).toStringAsFixed(1)}%'),
                                    subtitle: Text('坐标: (${box.x1.toInt()}, ${box.y1.toInt()})'),
                                  ),
                                );
                              },
                            ),
                        SingleChildScrollView(
                          padding: const EdgeInsets.all(16.0),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.stretch,
                            children: [
                              Text("检测报告", style: Theme.of(context).textTheme.titleMedium),
                              const SizedBox(height: 8),
                              Text(widget.result.report),
                              const SizedBox(height: 24),
                              ElevatedButton.icon(
                                icon: _isGeneratingSuggestion 
                                  ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2)) 
                                  : const Icon(Icons.lightbulb_outline),
                                label: const Text("获取智能建议 (AI)"),
                                onPressed: _isGeneratingSuggestion ? null : _getAiSuggestion,
                              ),
                              if(_suggestionText != null) ...[
                                const SizedBox(height: 16),
                                Container(
                                  padding: const EdgeInsets.all(12),
                                  decoration: BoxDecoration(
                                    color: Colors.blue.shade50,
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  child: Text(_suggestionText!),
                                )
                              ]
                            ],
                          ),
                        ),
                      ],
                    ),
                  )
                ],
              ),
            )
          )
        ],
      ),
    );
  }
}

// ===================================================================
// 5. 独立的绘图组件 (BoundingBoxImage)
// ===================================================================

class BoundingBoxImage extends StatefulWidget {
  final ImageSlice slice;

  const BoundingBoxImage({required this.slice, super.key});

  @override
  State<BoundingBoxImage> createState() => _BoundingBoxImageState();
}

class _BoundingBoxImageState extends State<BoundingBoxImage> {
  ui.Image? _imageInfo;

  @override
  void initState() {
    super.initState();
    _loadImage();
  }

  @override
  void didUpdateWidget(covariant BoundingBoxImage oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.slice.imageUrl != oldWidget.slice.imageUrl) {
      _loadImage();
    }
  }

  Future<void> _loadImage() async {
    final completer = Completer<ui.Image>();
    final stream = NetworkImage(widget.slice.imageUrl).resolve(const ImageConfiguration());
    final listener = ImageStreamListener(
      (ImageInfo info, bool _) {
        if (!completer.isCompleted) {
          completer.complete(info.image);
        }
      },
      onError: (exception, stackTrace) {
        if (!completer.isCompleted) {
          completer.completeError(exception);
        }
      },
    );
    stream.addListener(listener);
    try {
      final image = await completer.future;
      if (mounted) {
        setState(() {
          _imageInfo = image;
        });
      }
    } catch(e) {
      print("Failed to load image for painter: $e");
    } finally {
      stream.removeListener(listener);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      color: Colors.black87,
      child: InteractiveViewer(
        minScale: 1.0,
        maxScale: 4.0,
        child: Stack(
          fit: StackFit.expand,
          alignment: Alignment.center,
          children: [
            Image.network(
              widget.slice.imageUrl,
              fit: BoxFit.contain,
              loadingBuilder: (context, child, loadingProgress) {
                if (loadingProgress == null) return child;
                return const Center(child: CircularProgressIndicator());
              },
              errorBuilder: (context, error, stackTrace) {
                return const Center(child: Icon(Icons.error, color: Colors.red));
              },
            ),
            if (_imageInfo != null)
              CustomPaint(
                size: Size.infinite,
                painter: BoxPainter(
                  bboxes: widget.slice.bboxes,
                  image: _imageInfo!,
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class BoxPainter extends CustomPainter {
  final List<BoundingBox> bboxes;
  final ui.Image image;

  BoxPainter({required this.bboxes, required this.image});

  @override
  void paint(Canvas canvas, Size size) {
    final imageSize = Size(image.width.toDouble(), image.height.toDouble());
    final fittedSizes = applyBoxFit(BoxFit.contain, imageSize, size);
    final sourceSize = fittedSizes.source;
    final destinationRect = Alignment.center.inscribe(sourceSize, Rect.fromLTWH(0, 0, size.width, size.height));

    final paint = Paint()
      ..color = Colors.red
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;
      
    final scaleX = destinationRect.width / image.width;
    final scaleY = destinationRect.height / image.height;

    for (final box in bboxes) {
      final rect = Rect.fromLTRB(
        destinationRect.left + box.x1 * scaleX,
        destinationRect.top + box.y1 * scaleY,
        destinationRect.left + box.x2 * scaleX,
        destinationRect.top + box.y2 * scaleY,
      );
      canvas.drawRect(rect, paint);

      final textPainter = TextPainter(
        text: TextSpan(
          text: '${(box.confidence * 100).toStringAsFixed(1)}%',
          style: const TextStyle(
            color: Colors.white,
            backgroundColor: Colors.red,
            fontSize: 12,
          ),
        ),
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      textPainter.paint(canvas, rect.topLeft.translate(0, -textPainter.height));
    }
  }

  @override
  bool shouldRepaint(covariant BoxPainter oldDelegate) {
    return bboxes != oldDelegate.bboxes || image != oldDelegate.image;
  }
}
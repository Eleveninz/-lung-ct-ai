import 'package:flutter/material.dart';
import 'dart:io';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:share_plus/share_plus.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '肺结节检测',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const UploadPage(),
    );
  }
}

class UploadPage extends StatefulWidget {
  const UploadPage({super.key});

  @override
  State<UploadPage> createState() => _UploadPageState();
}

class _UploadPageState extends State<UploadPage> {
  final ImagePicker _picker = ImagePicker();
  List<XFile>? _images;
  bool _loading = false;

  Future pickImages() async {
    final imgs = await _picker.pickMultiImage();
    setState(() => _images = imgs);
    }

  Future submit() async {
    if (_images == null || _images!.isEmpty) return;
    
    setState(() => _loading = true);
    
    var uri = Uri.parse('http://localhost:5000/api/detect');
    var req = http.MultipartRequest('POST', uri);
    
    for (var img in _images!) {
      req.files.add(await http.MultipartFile.fromPath('images', img.path));
    }
    
    var res = await req.send();
    var body = await res.stream.bytesToString();
    var data = json.decode(body);
    
    setState(() => _loading = false);
    
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => ResultPage(data: data),  // 补充 context 参数
),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('上传 CT 切片')),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            ElevatedButton(
              onPressed: pickImages,
              child: const Text('选择图片'),
            ),
            const SizedBox(height: 10),
            Expanded(
              child: _images == null 
                ? const Center(child: Text('尚未选择图片')) 
                : GridView.count(
                    crossAxisCount: 3,
                    children: _images!
                      .map((f) => Image.file(File(f.path)))
                      .toList(),
                  ),
            ),
            if (_images != null && _images!.isNotEmpty)
              ElevatedButton(
                onPressed: _loading ? null : submit,
                child: _loading 
                  ? const CircularProgressIndicator() 
                  : const Text('开始检测'),
              ),
          ],
        ),
      ),
    );
  }
}

class ResultPage extends StatefulWidget {
  final Map data;
  const ResultPage({required this.data, super.key});

  @override
  State<ResultPage> createState() => _ResultPageState();
}

class _ResultPageState extends State<ResultPage> {
  int _index = 0;

  Future _downloadReport() async {
    final dir = await getApplicationDocumentsDirectory();
    final file = File('${dir.path}/report.txt');
    await file.writeAsString(widget.data['report']);
    
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('报告已保存：${file.path}')),
    );
  }

  @override
  Widget build(BuildContext context) {
    List slices = widget.data['slices'];
    String report = widget.data['report'];
    List bboxes = slices[_index]['bboxes'];
    String imgUrl = slices[_index]['image_url'];

    return Scaffold(
      appBar: AppBar(title: const Text('检测结果')),
      body: Column(
        children: [
          Expanded(
            child: PageView.builder(
              itemCount: slices.length,
              onPageChanged: (i) => setState(() => _index = i),
              itemBuilder: (context, i) {
                var slice = slices[i];
                return Stack(
                  children: [
                    Image.network(slice['image_url'], fit: BoxFit.contain),
                    ...slice['bboxes'].map<Widget>((box) {
                      return Positioned(
                        left: box['x1'].toDouble(),
                        top: box['y1'].toDouble(),
                        width: box['x2'] - box['x1'],
                        height: box['y2'] - box['y1'],
                        child: Container(
                          decoration: BoxDecoration(
                            border: Border.all(color: Colors.red, width: 2),
                          ),
                          child: Text(
                            '${(box['conf'] * 100).toStringAsFixed(1)}%',
                            style: const TextStyle(
                                color: Colors.red,
                                backgroundColor: Colors.white70),
                          ),
                        ),
                      );
                    }).toList(),
                  ],
                );
              },
            ),
          ),
          Expanded(
            child: ListView(
              children: bboxes.map<Widget>((box) {
                return ListTile(
                  title: Text('置信度: ${(box['conf'] * 100).toStringAsFixed(1)}%'),
                  subtitle: Text('位置: ${box['leaf']}, 大小: ${box['diameter_mm']} mm'),
                );
              }).toList(),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Text('报告：\n$report'),
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              ElevatedButton(
                onPressed: _downloadReport,
                child: const Text('下载报告'),
              ),
              ElevatedButton(
                onPressed: () => Share.share(report),
                child: const Text('分享'),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

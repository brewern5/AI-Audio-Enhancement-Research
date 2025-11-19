import 'package:http/http.dart' as http;
import 'dart:convert'; // Json conversion

final String baseurl = 'http://localhost:8000';

Future<List<dynamic>?> fetchNewest() async {
  try {

    Uri uri = Uri.parse('$baseurl/tracks/newest');

    final response = await http.get(uri);

    if(response.statusCode == 200) {
      final jsonData = jsonDecode(response.body);
      print('data received: $jsonData');
      return List<dynamic>.from(jsonData);

    } else {
      print('Error:  ${response.statusCode}');
      return null;
    }
  } catch (e) {
    print('Exception during API call: $e');
    return null;
  }
} 
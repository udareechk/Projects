/*
 *  This sketch demonstrates how to set up a simple HTTP-like server.
 *  The server will set a GPIO pin depending on the request
 *    http://server_ip/gpio/0 will set the GPIO2 low,
 *    http://server_ip/gpio/1 will set the GPIO2 high
 *  server_ip is the IP address of the ESP8266 module, will be 
 *  printed to Serial when the module is connected.
 */

#include <ESP8266WiFi.h>

const char* ssid = "Xperia";
const char* password = "abcd1234";

// Create an instance of the server
// specify the port to listen on as an argument
WiFiServer server(80);

// Match the request
  int val1, val2, val3, val4;

void setup() {
  Serial.begin(115200);
  delay(10);

  // prepare GPIO2
  pinMode(4, OUTPUT);
  pinMode(5, OUTPUT);
  pinMode(0, OUTPUT);
  pinMode(2, OUTPUT);
  digitalWrite(4, 0);
  digitalWrite(5, 0);
  digitalWrite(0, 0);
  digitalWrite(2, 0);
  
  // Connect to WiFi network
  Serial.println();
  Serial.println();
  Serial.print("Connecting to ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.println("WiFi connected");
  
  // Start the server
  server.begin();
  Serial.println("Server started");

  // Print the IP address
  Serial.println(WiFi.localIP());
}

void loop() {
  // Check if a client has connected
  WiFiClient client = server.available();
  if (!client) {
    return;
  }
  
  // Wait until the client sends some data
  Serial.println("new client");
  while(!client.available()){
    delay(1);
  }
  
  // Read the first line of the request
  String req = client.readStringUntil('\r');
  Serial.println(req);
  client.flush();
  
  if (req.indexOf("/d1off") != -1)
    val1 = 0;
    
  else if (req.indexOf("/d1on") != -1)
    val1 = 1;

  else if (req.indexOf("/d2off") != -1)
    val2 = 0;
  
  else if (req.indexOf("/d2on") != -1)
    val2 = 1;

  else if (req.indexOf("/d3off") != -1)
    val3 = 0;
  
  else if (req.indexOf("/d3on") != -1)
    val3 = 1;

  else if (req.indexOf("/d4off") != -1)
    val4 = 0;
  
  else if (req.indexOf("/d4on") != -1)
    val4 = 1;
  
  else {
    Serial.println("invalid request");
    client.stop();
    return;
  }

  // Set GPIO2 according to the request
  
  digitalWrite(5, val1);
  digitalWrite(4, val2);
  digitalWrite(0, val3);
  digitalWrite(2, val4);
  
  client.flush();

  // Prepare the response
  String s = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<!DOCTYPE HTML>\r\n<html>\r\n Plug 1: ";
  s += (val1)?"high":"low";
  s += "<br> Plug 2: ";
  s += (val2)?"high":"low";
  s += "<br> Plug 3: ";
  s += (val3)?"high: ":"low";
  s += "<br> Plug 4: ";
  s += (val4)?"high":"low";
  s += "</html>\n";

  // Send the response to the client
  client.print(s);
  delay(1);
  Serial.println("Client disonnected");

  // The client will actually be disconnected 
  // when the function returns and 'client' object is detroyed
}


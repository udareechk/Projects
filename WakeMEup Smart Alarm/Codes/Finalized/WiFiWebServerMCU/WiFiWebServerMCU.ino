/*
 *  This sketch demonstrates how to set up a simple HTTP-like server.
 *  The server will set a GPIO pin depending on the request
 *    http://server_ip/gpio/0 will set the GPIO2 low,
 *    http://server_ip/gpio/1 will set the GPIO2 high
 *  server_ip is the IP address of the ESP8266 module, will be 
 *  printed to Serial when the module is connected.
 */

#include <ESP8266WiFi.h>

const char* ssid = "WiFi iData_469C";
const char* password = "12345678";

// Create an instance of the server
// specify the port to listen on as an argument
WiFiServer server(80);

// Match the request
  int val[5];
  bool plugState[5];

void setup() {
  Serial.begin(115200);
  delay(10);

  // prepare GPIO2
  pinMode(16, OUTPUT);
  pinMode(4, OUTPUT);
  pinMode(2, OUTPUT);
  pinMode(14, OUTPUT);
  pinMode(2, OUTPUT);
  digitalWrite(16, 0);
  digitalWrite(4, 0);
  digitalWrite(2, 0);
  digitalWrite(14, 0);
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

//  if (req.indexOf("/MultiPlug") != -1){

    char i;

    for( i=0 ; i<5; i++){
      
      String chk = "/";
      char x = i + '0';
      chk.concat(x);
      
      Serial.println(chk);
      if (req.indexOf(chk  +"on") != -1){
          plugState[i] = true;
          val[i] = 1;
      } else if (req.indexOf(chk +"off") != -1){
          plugState[i] = false;
          val[i] = 0;
      }else{
        Serial.println("invalid request");
        client.stop();
        return;
      }
      }
  
//    if (req.indexOf("/0off") != -1)
//      val0 = 0;
//      
//    if (req.indexOf("/0on") != -1)
//      val0 = 1;
//  
//    if (req.indexOf("/1off") != -1)
//      val1 = 0;
//    
//    if (req.indexOf("/1on") != -1)
//      val1 = 1;
//  
//    if (req.indexOf("/2off") != -1)
//      val2 = 0;
//    
//    if (req.indexOf("/2on") != -1)
//      val2 = 1;
//  
//    if (req.indexOf("/3off") != -1)
//      val3 = 0;
//    
//    if (req.indexOf("/3on") != -1)
//      val3 = 1;
//      
//    if (req.indexOf("/4off") != -1)
//      val4 = 0;
//    
//    if (req.indexOf("/4on") != -1)
//      val4 = 1;
    
//  } else {
//    Serial.println("invalid request");
//    client.stop();
//    return;
//  }

  // Set GPIO2 according to the request
  digitalWrite(16, val[0]);
  digitalWrite(4, val[1]);
  digitalWrite(2, val[2]);
  digitalWrite(14, val[3]);
  digitalWrite(2, val[4]);
  
  client.flush();

  // Prepare the response
//  String s = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n<!DOCTYPE HTML>\r\n<html>\r\n Plug 1: ";
//  String s = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n";
//  s += (val1)?"high":"low";
//  s += "<br> Plug 2: ";
//  s += (val2)?"high":"low";
//  s += "<br> Plug 3: ";
//  s += (val3)?"high: ":"low";
//  s += "<br> Plug 4: ";
//  s += (val4)?"high":"low";
//  s += "</html>\n";

  // Send the response to the client
//  client.print(s);
  delay(1);
  Serial.println("Client disonnected");

  // The client will actually be disconnected 
  // when the function returns and 'client' object is detroyed
}


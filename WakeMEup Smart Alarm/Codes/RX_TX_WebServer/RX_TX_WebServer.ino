#include <ESP8266WiFi.h>
#include <WiFiClient.h> 
#include <ESP8266WebServer.h>

const char* ssid = "WiFi iData_469C";
const char* password = "12345678";
 
WiFiServer server(80);

const int ledPin = D12; // the pin that the LED is attached to
const int configPin = D5;
//int incomingByte;      // a variable to read incoming serial data into

//Variables to be sent to Mega
char setHours = 11, setMins = 48, setMonths = 4, setDays = 22;
int setYears = 2017;
char alarmHour = 19, alarmMin = 15;
char volume = 15;
char snoozeHour, snoozeMin, snoozeTime = 1;

//Functions
void webServerInit();
void webServerLoop();
bool readPinState(int pin);
void handleSubmit();
void sendConfigData();

void setup() {
  // initialize serial communication:
  Serial.begin(9600);
  // initialize the LED pin as an output:
  pinMode(ledPin, OUTPUT);
  pinMode(configPin, INPUT);
  
  digitalWrite(ledPin, LOW);
  digitalWrite(configPin, LOW);

  webServerInit();
}

void loop() {

//if(readPinState(configPin)){
if(digitalRead(configPin) == HIGH){
//  webServerInit();
//digitalWrite(ledPin, HIGH);
  webServerLoop();
//  digitalWrite(ledPin, LOW);
  }

}

void webServerInit(){
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
  Serial.print("Use this URL : ");
  Serial.print("http://");
  Serial.print(WiFi.localIP());
  Serial.println("/");
}

void webServerLoop(){

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
  String request = client.readStringUntil('\r');
  Serial.println(request);
  client.flush();
 
  // Match the request
 
  int value = LOW;
  if (request.indexOf("/ALARM=ON") != -1) {
    digitalWrite(ledPin, HIGH);
    Serial.end();
    Serial.begin(11500);
    Serial.print('H');
    delay(10000);
    Serial.end();
    value = HIGH;
  } 
  if (request.indexOf("/ALARM=OFF") != -1){
    digitalWrite(ledPin, LOW);
    Serial.end();
    Serial.begin(11500);
    Serial.print('L');
    delay(10000);
    Serial.end();
    value = LOW;
  }

  if (request.indexOf("/submit") != -1){
//    handleSubmit();
  
  }


//  server.on("/submit", handleSubmit);



  
  // Return the response
  client.println("HTTP/1.1 200 OK");
  client.println("Content-Type: text/html");
  client.println(""); //  do not forget this one
  client.println("<!DOCTYPE HTML>");
  client.println("<html>");
  client.println("<head>");
  client.println("<title>WakeMEup</title>");
  client.println("</head>");
  client.println("<body>");
  client.println("<h1>WakeMEup</h1>");
  client.println("<p>Alarm State:</p>");
 
  if(value == HIGH) {
    client.print("ON");  
  } else {
    client.print("OFF");
  }
  client.println("<br><br>");
  client.println("Click <a href=\"/ALARM=ON\">here</a> turn the ALARM on pin 13 ON<br>");
  client.println("Click <a href=\"/ALARM=OFF\">here</a> turn the ALARM on pin 13 OFF<br>");
  client.println("<form action = \"/submit\" method = \"POST\">First name:<br><input type=\"text\" name=\"firstname\"><br></form>");
  client.println("</html>");
 
  delay(1);
  Serial.begin(9600);
  Serial.println("Client disconnected");
  Serial.println("");

  client.stop();
  sendConfigData();

  
 
  }

  bool readPinState(int pin){

  return (digitalRead(pin) == HIGH);
  
  }

  void sendConfigData(){
    Serial.end();
    Serial.begin(11500);
    Serial.print(setHours);
    delay(5000);
    Serial.print(setMins);
    delay(5000);
    Serial.print(setMonths);
    delay(5000);
    Serial.print(setDays);
    delay(5000);
    Serial.print(setYears);
    delay(5000);
    Serial.print(alarmHour);
    delay(5000);
    Serial.print(alarmMin);
    delay(5000);
    Serial.print(volume);
    delay(5000);
    Serial.print(snoozeHour);
    delay(5000);
    Serial.print(snoozeMin);
    delay(5000);
    Serial.print(snoozeTime);
    delay(5000);
    Serial.end();
    Serial.begin(9600);
  
    }
//
//  void handleSubmit(){

//    if (server.args() > 0 ) {
//    for ( uint8_t i = 0; i < server.args(); i++ ) {
//      if (server.argName(i) == "firstname") {
//
//        Serial.println(server.arg(i));
//         // do something here with value from server.arg(i);
//      }
//   }
//}
    
    
//  }

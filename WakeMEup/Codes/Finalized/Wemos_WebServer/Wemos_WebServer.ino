#include <ESP8266WiFi.h>
#include <WiFiClient.h> 
#include <ESP8266WiFiMulti.h>

const char* ssid = "WiFi iData_469C";
const char* password = "12345678";
 
WiFiServer server(80);
ESP8266WiFiMulti WiFiMulti;

const int ledPin = D12; // the pin that the LED is attached to
const int configPin = D5;
bool initConfigMode = false;
bool configMode = false;
bool initMultiplugMode = false;
bool multiplugMode = false;

//Variables to be sent to Mega
char setHours = 11, setMins = 48, setMonths = 4, setDays = 22;
int setYears = 2017;
char alarmHour = 19, alarmMin = 15;
char volume = 15;
char snoozeHour, snoozeMin, snoozeTime = 1;
int changes[26];
String request;

//Functions
void webServerInit();
void webServerLoop();
bool readPinState(int pin);
void clientInit();
void clientLoop();

void setup() {
  // initialize serial communication:
  Serial.begin(9600);
  // initialize the LED pin as an output:
  pinMode(ledPin, OUTPUT);
  pinMode(configPin, INPUT);
  
  digitalWrite(ledPin, LOW);
  digitalWrite(configPin, HIGH);

  initConfigMode = readPinState(configPin);
//  initConfigMode = true;
}

void loop() {
  // Alarm configure intialization
  if (initConfigMode){
    webServerInit();
    initConfigMode = false;
    configMode = true;
  }
  
  // Alarm configuration
  if (configMode){
    webServerLoop();
    delay(100);
  //  configMode = false;
//    initMultiplugMode = true; 
  }

  // multiplug connection initialization
  if (initMultiplugMode){
    clientInit();
    initMultiplugMode = false;
    multiplugMode = true;
  }
   // sending data to multiplug
  if (multiplugMode){
    clientLoop();
    multiplugMode = false; 
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
  request = client.readStringUntil('\r');
  Serial.println(request);
  client.flush();
 
  // Match the request
  int value = LOW;
  if (request.indexOf("/ALARM=ON") != -1) {
    digitalWrite(ledPin, HIGH);
    strSplit(request);
    value = HIGH;
//    delay(1);
    configMode = false; 
    initMultiplugMode = true; 
  } 
  if (request.indexOf("/ALARM=OFF") != -1){
    digitalWrite(ledPin, LOW);
    value = LOW;
  }
 
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
  client.println("Click <a href=\"/ALARM=ON/?a=2&b=3&c=8&d=8000&e=90\">here</a> turn the ALARM on pin 13 ON<br>");
  client.println("Click <a href=\"/ALARM=OFF\">here</a> turn the ALARM on pin 13 OFF<br>");
  client.println("<form action = \"/submit\" method = \"POST\">First name:<br><input type=\"text\" name=\"firstname\"><br></form>");
  client.println("</html>");
 
  delay(1);
  Serial.begin(9600);
  Serial.println("Client disconnected");
  Serial.println("");

//  client.stop();
  
  }

  bool readPinState(int pin){
    return (digitalRead(pin) == HIGH);
  }

  void strSplit(String request){
  
//  String urlRest = "/?a=2&b=3&c=8&d=8000&e=90 HTTP/1.1";// request
  String urlRest = request;
  char strStart = char(urlRest.indexOf("?")) + 1;      // Start val
  char strEnd = char(urlRest.indexOf("HTTP"));   // end val
  
  char buff[100];
  
  Serial.println();
  Serial.println(strEnd, DEC);
  Serial.println(buff);
  Serial.println();

  urlRest.toCharArray(buff, strEnd+1);
  
  char i;
  char var[10];
  int val[10];
  String temp;
  char varCount = 0;
  bool isVar = true;
  
  for (i = strStart; i<=strEnd; i++){
    if (isVar) {
      var[varCount] = buff[i];
      isVar = false;
      i++;
    } else {
      temp = "";
      while (i <= strEnd && buff[i] != '&'){
        temp += buff[i];
        i++;
      }
      val[varCount] = temp.toInt();      
      isVar = true;
      varCount++;
    }
  }
  
  for (i=0; i<varCount; i++){

    int index = var[i] - 'a';
    changes[index] = val[i];
    
    Serial.print(var[i]);
    Serial.print("=");
    Serial.println(changes[index]);
  }
}

  void clientInit(){
    Serial.begin(9600);
    delay(10);

    // We start by connecting to a WiFi network
    WiFiMulti.addAP("WiFi iData_469C", "12345678");

    Serial.println();
    Serial.println();
    Serial.print("Wait for WiFi... ");

    while(WiFiMulti.run() != WL_CONNECTED) {
        Serial.print(".");
        delay(500);
    }

    Serial.println("");
    Serial.println("WiFi connected");
    Serial.println("IP address: ");
    Serial.println(WiFi.localIP());

    delay(500);
  }
  
  void clientLoop(){
    const uint16_t port = 80;
    const char * host = "192.168.169.100"; // ip or dns
    
    Serial.print("connecting to ");
    Serial.println(host);

    // Use WiFiClient class to create TCP connections
    WiFiClient client;

    if (!client.connect(host, port)) {
        Serial.println("connection failed");
        Serial.println("wait 5 sec...");
        delay(5000);
        return;
  
   }

     // This will send the request to the server
    String url = "/0on/1on/2on/3on/4on";
    client.print(String("GET ") + url + " HTTP/1.1\r\n");

    //read back one line from server
    String line = client.readStringUntil('\r');
    client.println(line);

    Serial.println("closing connection");
    client.stop();
    
    Serial.println("wait 5 sec...");
    delay(5000);

  }


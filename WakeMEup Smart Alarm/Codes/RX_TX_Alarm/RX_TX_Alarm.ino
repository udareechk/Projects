const int ledPin = 13; // the pin that the LED is attached to
int incomingByte;      // a variable to read incoming serial data into

  void setup() {
  Serial.begin(11500);
// initialize the LED pin as an output:
  pinMode(ledPin, OUTPUT);
}

void loop() {
//  Serial.print('H');
//  delay(5000);
//  Serial.print('L');
//  delay(10000);

  if (Serial.available() > 0) {
    // read the oldest byte in the serial buffer:
    incomingByte = Serial.read();
   
    // if it's a capital H (ASCII 72), turn on the LED:
    if (incomingByte == 'H') {
      digitalWrite(ledPin, HIGH);
    }
    // if it's an L (ASCII 76) turn off the LED:
    if (incomingByte == 'L') {
      digitalWrite(ledPin, LOW);
    }

     Serial.print(incomingByte);
  }

  
}


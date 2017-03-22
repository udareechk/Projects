#include "Arduino.h"
#include "SoftwareSerial.h"
#include "DFRobotDFPlayerMini.h"
#include <LiquidCrystal.h>
#include <DS3231.h>

SoftwareSerial mySoftwareSerial(10, 11); // RX, TX
DFRobotDFPlayerMini myDFPlayer;

// initialize the library with the numbers of the interface pins
LiquidCrystal lcd(9, 8, 7, 6, 5, 4);

// Init the DS3231 using the hardware interface
DS3231  rtc(SDA, SCL);

// Functions
void displayInfo();
void ringAlarm();
bool timeEquals();
void toggleAlarm();
void snoozeAlarm();
void checkAlarm();
void updateTimeRTC();
void setDateTime();
void checkBed();


// Static Variables

char hours, mins, secs;
char months, days;
int years;
float temperature;

char setHours = 11, setMins = 48, setMonths = 4, setDays = 22;
int setYears = 2017;
bool isSetTime = false;

char alarmHour = 19, alarmMin = 15;
char volume = 15;
char snoozeHour, snoozeMin, snoozeTime = 1;
char alarmState = 0;
bool alarmRinging = false;

bool body = false, head = false;
char bodyPin = 13, headPin = 12;

void setup() {

  // Initialize the rtc object
  rtc.begin();

  //------------------Configure MP3--------------------------------------------
  mySoftwareSerial.begin(9600);
  Serial.begin(115200);
  
  Serial.println();
  Serial.println(F("DFRobot DFPlayer Mini Demo"));
  Serial.println(F("Initializing DFPlayer ... (May take 3~5 seconds)"));
  
  if (!myDFPlayer.begin(mySoftwareSerial)) {  //Use softwareSerial to communicate with mp3.
    Serial.println(F("Unable to begin:"));
    Serial.println(F("1.Please recheck the connection!"));
    Serial.println(F("2.Please insert the SD card!"));
    while(true);
  }
  Serial.println(F("DFPlayer Mini online."));
  
  //----------------------------------------------------------------------------

  // Interrupt pins

  pinMode(18, INPUT_PULLUP);
  attachInterrupt(5, toggleAlarm, RISING);

  pinMode(bodyPin, INPUT);
  pinMode(headPin, INPUT);

}

void loop() {

  if (isSetTime){
    setDateTime();
  }
  
  interrupts();
  displayInfo();

  updateTimeRTC();
  checkAlarm();

  checkBed();


  Serial.println(body);
  Serial.println(head);
  Serial.println();

  delay(1000);

}

void ringAlarm(){
  myDFPlayer.volume(volume);  //Set volume value. From 0 to 30
  myDFPlayer.play(1);  //Play the first mp3

  alarmRinging = true;
}

void stopAlarm(){
  myDFPlayer.pause();
  alarmRinging = false;
}

void displayInfo(){
  // set up the LCD's number of columns and rows:
  lcd.begin(20, 4);
  // Print a message to the LCD.

  char strTime[20], strAlarm[20], strDate[20];
  sprintf(strTime, "* Time : %02d:%02d:%02d", hours, mins, secs);
  sprintf(strDate, "* Date : %04d:%02d:%02d", years, months, days);
  sprintf(strAlarm, "* Alarm : %02d:%02d", snoozeHour, snoozeMin);
  
  lcd.print(strTime);
  lcd.setCursor(0,1);
  lcd.print(strDate);
  lcd.setCursor(0,2);
  lcd.print(strAlarm);
  lcd.setCursor(0,3);

  if (alarmState == 0){
    lcd.print("* OFF!");
  } else if (alarmState == 1){
    lcd.print("* ON!");
  } else {
    lcd.print("* SNOOZE!");
  }

  lcd.setCursor(13,3);
  lcd.print(temperature);
  lcd.setCursor(18,3);
  lcd.print("'C");
}

bool timeEquals(){
  return (mins == snoozeMin && hours == snoozeHour);
}

void toggleAlarm(){
  if (alarmState == 0 && !alarmRinging){
    alarmState = 1;
    snoozeHour = alarmHour;
    snoozeMin = alarmMin;
    
  } else if (alarmState == 1 && alarmRinging){
    alarmState = 2;
    
  } else if (alarmState == 1 && !alarmRinging){
    alarmState = 0;
  }
}

void snoozeAlarm(){
  snoozeMin = mins + snoozeTime;
  if (snoozeMin > 59){
    snoozeHour++;
    snoozeMin -= 60;
  }
  if (snoozeHour > 23){
    snoozeHour = 0;
  }
}

void checkAlarm(){
  
  if (timeEquals() && !alarmRinging && alarmState != 0 && body){
    ringAlarm();
    alarmState = 1;
  }

  if (timeEquals() && !alarmRinging && alarmState != 0 && !body){
    alarmState = 0;
  }

  if (alarmRinging && alarmState == 2){
    stopAlarm();
    snoozeAlarm();
  }
}

void updateTimeRTC(){

  // Get Time
  String rtcTime = rtc.getTimeStr();
  String tmpHours = rtcTime.substring(0,2);
  hours = tmpHours.toInt();
  String tmpMins = rtcTime.substring(3,5);
  mins = tmpMins.toInt();
  String tmpSecs = rtcTime.substring(6,8);
  secs = tmpSecs.toInt();

  // Get Date
  String rtcDate = rtc.getDateStr();
  String tmpYears = rtcDate.substring(6,10);
  years = tmpYears.toInt();
  String tmpMonths = rtcDate.substring(3,5);
  months = tmpMonths.toInt();
  String tmpDays = rtcDate.substring(0,2);
  days = tmpDays.toInt();

  // Get Temperature
  temperature = rtc.getTemp();
}

void setDateTime(){
  rtc.setTime(setHours, setMins, 0);     // Set the time to 12:00:00 (24hr format)
  rtc.setDate(setDays, setMonths, setYears);   // Set the date to January 1st, 2014
  isSetTime = false;    // Done setting time
}

void checkBed(){
  if (digitalRead(bodyPin)){
    body = false;
  } else {
    body = true;
  }

  if (digitalRead(headPin)){
    head = false;
  } else {
    head = true;
  }
}


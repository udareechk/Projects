#include "Arduino.h"
#include "SoftwareSerial.h"
#include "DFRobotDFPlayerMini.h"
#include <LiquidCrystal.h>
#include <DS3231.h>

#define PLUGS 4

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
void strSplit();


// Static Variables

char hours, mins, secs;
char months, days;
int years;
float temperature;

// Set Clock
char setHours = 12, setMins = 30, setMonths = 4, setDays = 2;
int setYears = 2017;
bool isSetTime = false, isSetDate = false;

// Alarm
char alarmHour = 14, alarmMin = 5;
char volume = 25, sound = 1;
char snoozeHour, snoozeMin, snoozeTime = 1, snoozeCount = 3;
char alarmState = 0;
bool alarmRinging = false;
int sleepCount;

// Buttons
bool body = false, head = false;
char bodyPin = 13, headPin = 12;

// Configure
int changes[26];

// Multiplug
bool plugState[PLUGS] = {false}, plugStart[PLUGS], plugEnd[PLUGS], plugToggle[PLUGS];

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
    configTime();
  }

  if (isSetDate){
    configDate();
  }
  
  interrupts();
  displayInfo();

  updateTimeRTC();
  checkAlarm();

  checkBed();


//  Serial.println(body);
//  Serial.println(head);
//  Serial.println();

  delay(1000);

}

void ringAlarm(){
  myDFPlayer.volume(volume);  //Set volume value. From 0 to 30
  myDFPlayer.play(sound);  //Play the first mp3

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

void configTime(){
  rtc.setTime(setHours, setMins, 0);     // Set the time to 12:00:00 (24hr format)
  isSetTime = false;    // Done setting time
}

void configDate(){
  rtc.setDate(setDays, setMonths, setYears);   // Set the date to January 1st, 2014
  isSetDate = false;  // Done setting date
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

void strSplit(){
  
}

void configure(){
  for (int i = 0; i<26; i++){
    changes[i] = -1;
  }
  strSplit();

  if (changes[0] != -1){
    alarmHour = changes[0]/100;
    alarmMin = changes[0]%100;
  }
  if (changes[1] != -1){
    snoozeTime = changes[1];
  }
  if (changes[2] != -1){
    snoozeCount = changes[2];
  }
  if (changes[3] != -1){
    volume = changes[3];
  }
  if (changes[4] != -1){
    sound = changes[4];
  }
  if (changes[5] != -1){
    setHours = changes[5]/100;
    setMins = 30%100;
    isSetTime = true;
  }
  if (changes[6] != -1){
    setYears = changes[6]/10000;
    setMonths = (changes[6]%10000)/100;
    setDays = changes[6]%100;
    isSetDate = true;
  }
}

void operateMultiplug(){
  int tempTime = hours * mins;
  for (int i=0; i<PLUGS; i++){
    if (tempTime == plugStart[i]){
      plugState[i] = true;
    }
    if (tempTime == plugEnd[i]){
      plugState[i] = false;
    }
    
  }
}




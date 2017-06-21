void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
}

void loop() {
  // put your main code here, to run repeatedly:
  
  strSplit();
  delay(1000);
  
  
}

void strSplit(){
  
  String urlRest = "?a=2&b=3&c=8&d=8000&e=90";    // request
  char strStart = 1;      // Start val
  char strEnd = urlRest.length();   // end val

  
  
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
    Serial.print(var[i]);
    Serial.print("=");
    Serial.println(val[i]);
  }
}


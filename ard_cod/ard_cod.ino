#include <Servo.h>

Servo sortingServo;
int currentAngle = 0;  // Current servo position
unsigned long lastActivation = 0;
const int RESET_DELAY = 2000;  // 2 seconds to reset

void setup() {
  Serial.begin(115200);
  sortingServo.attach(9);  // Servo on pin 9
  sortingServo.write(0);   // Initial position
}

void loop() {
  // Reset to neutral position after delay
  if (millis() - lastActivation > RESET_DELAY && currentAngle != 0) {
    sortingServo.write(0);
    currentAngle = 0;
  }

  // Read angle from Python
  if (Serial.available() > 0) {
    int newAngle = Serial.parseInt();
    
    if (newAngle >= 0 && newAngle <= 180) {
      sortingServo.write(newAngle);
      currentAngle = newAngle;
      lastActivation = millis();
    }
  }
}
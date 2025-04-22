#include <ESP8266WiFi.h>
#include <ESP8266WebServer.h>
#include <Servo.h>

// Wi-Fi credentials
const char* ssid = "your-SSID";
const char* password = "your-PASSWORD";

// Create a server on port 80
ESP8266WebServer server(80);

// Create a Servo object
Servo myservo;
int servoPin = D1;  // Adjust the pin as per your setup

void setup() {
  // Start serial communication
  Serial.begin(115200);
  
  // Connect to Wi-Fi
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.println("Connecting to WiFi...");
  }
  Serial.println("Connected to WiFi");

  // Attach the servo to the pin
  myservo.attach(servoPin);

  // Define route to trigger servo
  server.on("/trigger_servo", HTTP_POST, triggerServo);

  // Start the server
  server.begin();
}

void loop() {
  server.handleClient();  // Handle incoming requests
}

void triggerServo() {
  // Trigger the servo to move (you can adjust this as per your requirement)
  myservo.write(90);  // Move to 90 degrees
  delay(1000);        // Hold for 1 second
  myservo.write(0);   // Move back to 0 degrees

  // Respond to the Flask server
  server.send(200, "text/plain", "Servo triggered");
}
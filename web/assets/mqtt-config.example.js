// Copy to mqtt-config.js and configure a browser MQTT user with subscribe-only
// access to topgun/coffee/results. Browser configuration is visible to visitors.
window.TOPGUN_MQTT_CONFIG = {
  url: "wss://YOUR-BROKER:8084/mqtt",
  username: "YOUR-WEB-SUBSCRIBER",
  password: "REPLACE-ME"
};

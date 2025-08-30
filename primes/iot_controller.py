import paho.mqtt.client as mqtt
import json
import torch

# Team Instruction: Implement IoT controller for API data and SPACE HVAC monitoring.
# Use MQTT for real-time control, inspired by Emeagwali’s distributed coordination.
class IoTController:
    def __init__(self, broker: str = "mqtt://localhost:1883"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client = mqtt.Client()
        self.client.connect(broker.split("://")[1])
        self.modes = ["full", "restricted", "cutoff"]

    def on_message(self, client, userdata, message):
        payload = json.loads(message.payload.decode())
        mode = payload.get("mode", "full")
        if mode in self.modes:
            print(f"Set IoT mode: {mode} for API data and SPACE HVAC")

    def set_mode(self, mode: str, power_level: float, api_enabled: bool):
        """Sets IoT mode for API data syncing and SPACE HVAC control."""
        if mode not in self.modes:
            raise ValueError(f"Invalid mode: {mode}")
        payload = {"mode": mode, "power_level": power_level, "api_enabled": api_enabled}
        self.client.publish("infinity/iot/control", json.dumps(payload))

# Example usage
if __name__ == "__main__":
    controller = IoTController()
    controller.set_mode("full", 0.9, True)
    print("IoT mode set to full with 90% power and API syncing enabled")
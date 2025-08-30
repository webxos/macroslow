import paho.mqtt.client as mqtt
import json
import torch

# Team Instruction: Implement SPACE HVAC controller for GLASTONBURY 2048.
# Ensure IoT reliability for medical AI environments (e.g., Nigerian clinics).
class SPACEHVACController:
    def __init__(self, broker: str = "mqtt://localhost:1883"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client = mqtt.Client()
        self.client.connect(broker.split("://")[1])
        self.client.subscribe("infinity/hvac/control")

    def on_message(self, client, userdata, message):
        """Handles SPACE HVAC control signals for temperature and air quality."""
        payload = json.loads(message.payload.decode())
        temp = payload.get("temperature", 22.0)
        air_quality = payload.get("air_quality", 0.9)
        print(f"SPACE HVAC set: {temp}°C, Air Quality: {air_quality}")

    def set_environment(self, temperature: float, air_quality: float):
        """Sets SPACE HVAC parameters for IoT environment control."""
        payload = {"temperature": temperature, "air_quality": air_quality}
        self.client.publish("infinity/hvac/control", json.dumps(payload))

# Example usage
if __name__ == "__main__":
    hvac = SPACEHVACController()
    hvac.set_environment(22.0, 0.95)
    hvac.client.loop_start()
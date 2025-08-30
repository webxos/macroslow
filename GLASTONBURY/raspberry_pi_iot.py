import time
import json
import torch
from src.glastonbury_2048.iot_mqtt_bridge import IoTMQTTBridge

# Team Instruction: Implement Raspberry Pi IoT integration for GLASTONBURY 2048.
# Simulate sensor data (e.g., medical vitals) for real-time API syncing.
class RaspberryPiIoT:
    def __init__(self, mqtt_broker: str = "mqtt://localhost:1883"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mqtt_bridge = IoTMQTTBridge(mqtt_broker)

    def collect_sensor_data(self) -> torch.Tensor:
        """Simulates Raspberry Pi sensor data (e.g., heart rate, temperature)."""
        sensor_data = {
            "heart_rate": 75.0 + time.time() % 10,
            "temperature": 36.5 + (time.time() % 5) / 10,
            "timestamp": time.time()
        }
        return torch.tensor(list(sensor_data.values()), device=self.device)

    def sync_to_api(self):
        """Syncs sensor data to INFINITY UI via MQTT."""
        data = self.collect_sensor_data()
        self.mqtt_bridge.publish_iot_data(data)

# Example usage
if __name__ == "__main__":
    pi = RaspberryPiIoT()
    while True:
        pi.sync_to_api()
        time.sleep(5)  # Simulate sensor data every 5 seconds
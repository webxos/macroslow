import paho.mqtt.client as mqtt
import json
import torch
from src.glastonbury_2048.aes_2048 import AES2048Encryptor

# Team Instruction: Implement MQTT bridge for IoT integration with GLASTONBURY 2048.
# Securely transmit API and IoT data (e.g., Raspberry Pi sensors) with 2048-bit AES.
class IoTMQTTBridge:
    def __init__(self, broker: str = "mqtt://localhost:1883"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encryptor = AES2048Encryptor()
        self.client = mqtt.Client()
        self.client.on_message = self.on_message
        self.client.connect(broker.split("://")[1])
        self.client.subscribe("infinity/iot/data")

    def on_message(self, client, userdata, message):
        """Handles incoming IoT data, decrypts, and processes with CUDA."""
        payload = json.loads(message.payload.decode())
        decrypted_data = self.encryptor.decrypt(bytes.fromhex(payload["data"]))
        tensor_data = torch.tensor(json.loads(decrypted_data.decode()), device=self.device)
        print(f"Processed IoT data shape: {tensor_data.shape}")

    def publish_iot_data(self, data: torch.Tensor):
        """Publishes encrypted IoT data to MQTT topic."""
        data_np = data.cpu().numpy().tobytes()
        encrypted_data = self.encryptor.encrypt(data_np).hex()
        payload = {"data": encrypted_data, "timestamp": __import__('time').time()}
        self.client.publish("infinity/iot/data", json.dumps(payload))

# Example usage
if __name__ == "__main__":
    bridge = IoTMQTTBridge()
    sample_data = torch.ones(100, device="cuda")
    bridge.publish_iot_data(sample_data)
    bridge.client.loop_start()
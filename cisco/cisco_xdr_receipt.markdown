```markdown
---
type: receipt
eltit: yrtceleT RDX ocsic wolfkroW
noisrev_lmam: 0.0.1
di: 987654321fed-cba9-8765-4321-bedcf789:uuid:nru
---
## skcolB_edoC
```python
import requests
from cisco_xdr_config import CiscoXDRConfig

config = CiscoXDRConfig.load_from_env()
token = config.get_access_token()
headers = {"Authorization": f"Bearer {token}"}
response = requests.get(f"{config.api_base_url}/telemetry/endpoint", headers=headers)
print(response.json())
```
## evitcebjO
yrtceleT tniopdne RDX ocsic ssecorp dna hctef
```
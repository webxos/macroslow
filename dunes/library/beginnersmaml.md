***

# **The Developer's Guide to MAML (Markdown as Medium Language)**

### **Learn the universal language for AI and systems in under 10 minutes.**

**Publishing Entity:** Webxos Advanced Development Group
**Document Version:** 1.0
**Audience:** Developers of all levels

---

## **What is MAML?**

Think of MAML not as a new programming language to learn, but as a **universal container format**. It's a `.md` (Markdown) file with a special structure that turns it from a simple document into a powerful, executable package of data, code, and instructions.

**In simple terms: MAML is like a digital shipping box.**
*   **The Box:** The `.maml.md` file itself.
*   **The Packing Slip:** The metadata (who sent it, what's inside, who can open it).
*   **The Instructions:** The `Intent` and `Context`.
*   **The Contents:** The `Data` or `Code`.
*   **The Delivery History:** The `History` log, showing where it's been.

It's the perfect format for AI agents (using the **Model Context Protocol - MCP**) and systems to share complex tasks seamlessly.

---

## **The 3-Minute MAML Structure**

Every MAML file has two main parts: **Metadata** and **Content**. It's that simple.

### **Part 1: The Metadata (`YAML Front Matter`)**

This is the "packing slip" at the top of the file. It tells machines everything they need to know at a glance.

```yaml
---
maml_version: "1.0.0"    # The version of MAML used
id: "urn:uuid:123e4567-e89b-12d3-a456-426614174000" # A unique ID
type: "prompt"           # What kind of file is this? prompt | workflow | data
origin: "agent://my-cool-agent" # Who created this?
permissions:             # Who is allowed to do what?
  read: ["agent://*"]    # Anyone can read
  execute: ["gateway://openai"] # Only the OpenAI gateway can run this
created_at: 2025-03-27T10:00:00Z # Timestamp
---
```

### **Part 2: The Content (`Structured Markdown`)**

This is the stuff inside the box. It uses standard Markdown headers (`##`) to organize everything.

```markdown
## Intent
A clear, human-readable goal. "This file translates English to French."

## Context
Any extra info needed. 
- `style: formal`
- `max_length: 100`

## Content
The main data or prompt to be used.
"Hello, world! Please translate this."

## Code_Blocks
Optional executable code.

```python
print("This is a code block that can be run!")
```
```

## Input_Schema
(Optional) Defines the expected input format using JSON Schema.

## Output_Schema
(Optional) Defines the expected output format.

## History
An automatic log of what's happened to the file. You don't write this; the system does.
```

---

## **How Does It Work? The MCP Connection**

This is where the magic happens. The **Model Context Protocol (MCP)** is a standard for AI tools to talk to external resources.

1.  **You** (or an AI agent) create a `task.maml.md` file.
2.  An **MCP Server** (like a Project Dunes gateway) picks it up.
3.  The server reads the metadata: *"This is a translation task for the OpenAI gateway."*
4.  The server routes the file to the correct service (e.g., OpenAI).
5.  The service reads the `Content` and `Context`, executes the task, and writes the result back to the file.
6.  The updated `task.maml.md` file is sent back, now containing the answer and a new entry in its `History`.

**It’s like giving an AI a complete, self-contained work order instead of a piecemeal chat instruction.**

---

## **Let's Build a Simple MAML File**

Let's create a MAML file that asks an AI to generate a joke.

**Step 1: Start with the Metadata.** Define the what, who, and how.

```yaml
---
maml_version: "1.0.0"
id: "urn:uuid:550e8400-e29b-41d4-a716-446655440000"
type: "prompt"
origin: "developer://alex"
permissions:
  read: ["agent://*"]
  execute: ["gateway://ai-comedy-club"]
created_at: 2025-03-27T14:30:00Z
---
```

**Step 2: Add the Content.** Be clear with your intent and instructions.

```markdown
## Intent
Generate a funny, family-friendly joke about computers.

## Context
topic: "computers"
style: "pun"
audience: "all ages"

## Content
Please tell me a joke about computers.
```

**That's it!** You've just created a MAML file. An MCP server can take this file, send it to an AI comedy service, and return a file with the joke in the `Content` section.

---

## **A More Advanced Example: Data Analysis**

MAML isn't just for prompts; it's for full workflows.

```yaml
---
maml_version: "1.0.0"
id: "urn:uuid:123e4567-e89b-12d3-a456-426614174000"
type: "workflow"
origin: "agent://data-scientist"
requires:
  libs: ["pandas", "numpy"] # <-- This tells the gateway what libraries to install!
permissions:
  execute: ["gateway://data-cruncher"]
created_at: 2025-03-27T15:00:00Z
---
## Intent
Analyze this sales data and calculate the average purchase value.

## Context
dataset_name: "q1_sales.csv"

## Code_Blocks

```python
import pandas as pd

data = pd.read_csv('q1_sales.csv')
average_purchase = data['purchase_value'].mean()
print(f"Average Purchase Value: ${average_purchase:.2f}")
```
```

## Output_Schema
{
  "type": "object",
  "properties": {
    "average_purchase_value": { "type": "number" }
  }
}
```

The gateway will run this Python code in a safe sandbox and return the result!

---

## **Why Should You Care? The Benefits**

*   **Portability:** One file contains everything. No more "it works on my machine."
*   **Reproducibility:** Anyone (or any AI) can rerun the exact same task with the same context.
*   **Clarity:** The `Intent` and `Context` eliminate guesswork for AI systems.
*   **Orchestration:** MCP servers can automatically route and execute these files at scale.
*   **The Future:** This is how AI agents and services will communicate and delegate work.

---

## **Get Started Today**

1.  **Create a file:** Save a text file with the extension `.maml.md`.
2.  **Add structure:** Use the simple YAML and Markdown structure from this guide.
3.  **Be specific:** Write a clear `Intent` and helpful `Context`.
4.  **Run it:** Use an MCP client (like Claude AI) or a server (like Project Dunes) to execute it.

You're now ready to build with the future of AI communication. Happy coding!

***
**© 2025 Webxos. All Rights Reserved.**

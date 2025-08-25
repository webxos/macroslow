from crewai import Agent
import os

class VideoScribeAgent(Agent):
    def __init__(self):
        super().__init__(role="Video Scribe", goal="Generate training videos", backstory="Video synthesis expert", llm="openai/gpt-4o")

    def generate_video(self, context):
        report_url = context.get("report_url", "default_report.txt")
        video_path = f"videos/training_{os.urandom(8).hex()}.mp4"
        with open(report_url, "r") as f:
            content = f.read()
        return f"Video generated at URL: {video_path} from {content[:50]}..."

agent = VideoScribeAgent()

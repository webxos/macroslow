from jinja2 import Environment, FileSystemLoader
from datetime import datetime
import os

env = Environment(loader=FileSystemLoader('../frontend/templates/'))
template = env.get_template('index.html.j2')

html_output = template.render(
    generated_date=datetime.utcnow().isoformat(),
    page_title="annot8 - Collaborative Data Annotation Hub"
)

with open('../frontend/static/index.html', 'w') as f:
    f.write(html_output)

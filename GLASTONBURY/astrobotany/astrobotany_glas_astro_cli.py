import click
import json
from .data_ingestor import AstrobotanyIngestor
from .bio_verifier import BioVerifier

@click.group()
def AstrobotanyCLI():
    pass

@AstrobotanyCLI.command()
@click.option("--source", default="NASA", help="Data source (NASA, SpaceX, Citizen)")
@click.option("--data-file", help="Path to data file")
def ingest(source, data_file):
    """Ingest astrobotany data."""
    with open(data_file, "r") as f:
        data = json.load(f)
    ingestor = AstrobotanyIngestor()
    result = ingestor.ingest_data(source, data)
    click.echo(f"Ingested: {result}")

@AstrobotanyCLI.command()
@click.option("--data-file", help="Path to data file")
def verify(data_file):
    """Verify astrobotany data."""
    with open(data_file, "r") as f:
        data = json.load(f)
    verifier = BioVerifier()
    result = verifier.verify_biodata(data)
    click.echo(f"Verification: {'Valid' if result else 'Invalid'}")
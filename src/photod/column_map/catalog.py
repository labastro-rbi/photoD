"""
Column map for catalog data.
"""

from photod.column_map.base import mapper_from_glossary
from pathlib import Path

glossary_path = Path(__file__).parent / "glossary.yaml"
m = mapper_from_glossary("CatalogColumnMap", "Reference catalog columns", glossary_path)

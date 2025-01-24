"""
Base class of any column map.
"""

from dataclasses import dataclass
import yaml


@dataclass
class ColumnMap:

    name: str = "Name of column map"
    purpose: str = "Purpose of map"

    def __init__(self, name: str, purpose: str):
        self.name = name
        self.purpose = purpose


def mapper_from_glossary(class_name: str, purpose: str, glossary_yaml: str) -> ColumnMap:
    """
    Given a class name and purpose, creates a class of that name as a specialization of
    ColumnMap whose attributes are read-only properties informed by the elements of the
    glossary that specify a variable and an id.
    """
    with open(glossary_yaml) as f_in:
        glossary = yaml.safe_load(f_in)
    clazz_attrs = {'__init__': lambda self: ColumnMap.__init__(self, class_name, purpose)}
    clazz_attrs.update({x['variable']: property(fget=lambda s, value=x['id']: value, doc=x['description']) for x in glossary if 'variable' in x})
    clazz = type(class_name, (ColumnMap,), clazz_attrs)
    return clazz()

"""
Entity containment hierarchy resolution.

This module resolves parent-child relationships between entities in a trial.
When an entity is "contained in" another entity, it inherits properties
from its parent. This module handles that inheritance, type conversion,
and attribute spreading (e.g., splitting a ``location`` property into
``mapName``, ``latitude``, and ``longitude``).
"""

from copy import deepcopy

def key_from_name(named_entity):
    """
    Build a unique key string from an entity's type and name.

    Parameters
    ----------
    named_entity : dict
        An entity dict with ``deviceTypeName`` and ``deviceItemName`` keys.

    Returns
    -------
    str or None
        A key in the format ``"typeName : itemName"``, or None if either
        field is missing.
    """
    t = named_entity.get("deviceTypeName", None)
    n = named_entity.get("deviceItemName", None)
    if t is None or n is None:
        return None
    return t + " : " + n


def get_parent(xref_entities, named_entity):
    """
    Look up the parent entity of a contained entity.

    Parameters
    ----------
    xref_entities : dict[str, dict]
        A cross-reference dictionary mapping entity keys (from
        :func:`key_from_name`) to entity dicts.
    named_entity : dict
        The child entity dict, which may have a ``containedIn`` field
        with ``deviceTypeName`` and ``deviceItemName``.

    Returns
    -------
    dict or None
        The parent entity dict, or None if the entity has no parent
        or the parent is not found in the cross-reference.
    """
    containedIn = named_entity.get("containedIn", {})
    key = key_from_name(containedIn)
    if key is None:
        return None
    return xref_entities.get(key, None)


def get_attrs(entity):
    """
    Extract the list of named attributes from an entity.

    Filters out any attribute entries that don't have a ``name`` field.

    Parameters
    ----------
    entity : dict
        An entity dict with an optional ``attributes`` list.

    Returns
    -------
    list[dict]
        A list of attribute dicts that have a non-None ``name`` key.
    """
    attrsList =  [x for x in entity.get("attributes", []) if x.get("name", None) is not None]
    return attrsList


def spread_attributes(entity):
    """
    Flatten complex attributes into top-level entity fields.

    Performs the following transformations on the entity dict in-place:

    - ``location`` dict is spread into ``mapName``, ``latitude``, ``longitude``
    - ``attributes`` list is spread into individual top-level keys
    - ``containedIn`` dict is spread into ``containedInType`` and ``containedIn`` (name only)

    Parameters
    ----------
    entity : dict
        The entity dict to transform. Modified in-place.
    """
    if "location" in entity:
        loc = entity["location"]
        entity["mapName"] = loc["name"]
        entity["latitude"] = loc["coordinates"][1]
        entity["longitude"] = loc["coordinates"][0]
        del entity["location"]
    if "attributes" in entity:
        entity_attrs = get_attrs(entity)
        for attr in entity_attrs:
            entity[attr["name"]] = attr["value"]
        del entity["attributes"]

    if "containedIn"  in entity:
        entity["containedInType"] = entity["containedIn"]["deviceTypeName"]
        entity["containedIn"] = entity["containedIn"]["deviceItemName"]

def handle_String(value):
    """
    Type handler for String properties (pass-through).

    Parameters
    ----------
    value : str
        The string value.

    Returns
    -------
    str
        The value unchanged.
    """
    return value

def handle_Number(value):
    """
    Type handler for Number properties (converts to float).

    Parameters
    ----------
    value : str or numeric
        The value to convert.

    Returns
    -------
    float
        The value as a float.
    """
    return float(value)



def fill_properties_by_contained(entities_types_dict, meta_entities):
    """
    Resolve containment hierarchies and inherit parent properties.

    Walks the entity containment tree (child -> parent -> grandparent, etc.)
    and copies missing attributes from parent entities to children. Also
    applies type conversion (Number -> float, String -> str) and flattens
    complex attributes via :func:`spread_attributes`.

    Parameters
    ----------
    entities_types_dict : dict[str, EntityType]
        A dictionary mapping entity type names to ``EntityType`` objects.
        Used to look up attribute type definitions for type conversion.
    meta_entities : list[dict]
        A list of raw entity dicts from the trial metadata, each potentially
        containing ``containedIn``, ``attributes``, and ``location`` fields.

    Returns
    -------
    list[dict]
        A new list of entity dicts (deep-copied) with inherited properties
        resolved and complex attributes flattened. The original
        ``meta_entities`` list is not modified.
    """

    handlerDict = dict(Number=handle_Number,
                       String=handle_String)

    # Build a cross-reference lookup: "TypeName : ItemName" → entity dict.
    # This allows O(1) parent lookups when walking the containment chain.
    xref_entities = {key_from_name(e): e for e in meta_entities if key_from_name(e) is not None}

    # Deep-copy to avoid mutating the original trial metadata.
    filled_entities = deepcopy(meta_entities)

    # Phase 1: Type-convert attributes and inherit from parents.
    for entity in filled_entities:
        if key_from_name(entity) is not None:
            # Look up the entity type schema to get attribute type definitions.
            device_type = entities_types_dict[entity["deviceTypeName"]]
            type_attrs = device_type._metadata.get("attributeTypes", [])
            attrs_names = [a.get("name", None) for a in type_attrs]
            attrs_names = [a for a in attrs_names if a is not None]
            # Map attribute name → type string (e.g. "height" → "Number")
            type_attrs_dict = dict((x['name'],x['type']) for x in type_attrs)

            # Apply type conversion to each attribute value using the
            # handler for its declared type (Number → float, String → str).
            entity_attrs = get_attrs(entity)
            for singleEntityAttrs in entity_attrs:
                entityName = singleEntityAttrs['name']
                singleEntityAttrs['value'] = handlerDict[type_attrs_dict[entityName]](singleEntityAttrs['value'])

            # Walk up the containment chain: child → parent → grandparent → ...
            # At each level, inherit missing attributes and location from parent.
            # Child's own values always take precedence (only copy if not found).
            parent = get_parent(xref_entities, entity)
            while parent is not None:
                parent_attrs = get_attrs(parent)
                # Always inherit location from parent (overwritten at each level,
                # so the immediate parent's location wins).
                entity["location"] = parent.get('location', {})
                for pa in parent_attrs:
                    pa_name = pa["name"]
                    # Only inherit attributes that are defined in the type schema
                    # and not already present on the child entity.
                    if pa_name in attrs_names:
                        found = [ea for ea in entity_attrs if pa_name == ea["name"]]
                        if len(found) == 0:
                            entity_attrs.append(pa)

                parent = get_parent(xref_entities, parent)

            entity["attributes"] = entity_attrs

    # Phase 2: Flatten complex attributes into top-level fields.
    # location → mapName, latitude, longitude
    # attributes list → individual keys
    # containedIn dict → containedInType, containedIn (name only)
    for entity in filled_entities:
        spread_attributes(entity)

    return filled_entities

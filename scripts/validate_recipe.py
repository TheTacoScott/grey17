#!/usr/bin/env python3
"""
Runs inside Docker container (system Python 3).
Validates a recipe.yaml against schema/recipe.schema.json.

Usage:
    python3 validate_recipe.py \
        --recipe /work/recipe/edit.recipe.yaml \
        --schema /work/schema/recipe.schema.json

Exits 0 on success, 1 if validation errors are found.
"""
import argparse
import json
import sys

import jsonschema
import yaml


def parse_args():
    p = argparse.ArgumentParser(description="Validate a grey17 recipe against its JSON Schema.")
    p.add_argument("--recipe", required=True, help="Path to recipe.yaml")
    p.add_argument("--schema", required=True, help="Path to recipe.schema.json")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.schema) as f:
        schema = json.load(f)

    with open(args.recipe) as f:
        recipe = yaml.safe_load(f)

    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(recipe), key=lambda e: list(e.absolute_path))

    if not errors:
        print("OK: recipe validates against schema ({})".format(args.recipe), flush=True)
        sys.exit(0)

    print("Schema validation errors in {}:".format(args.recipe), file=sys.stderr)
    for err in errors:
        path = " -> ".join(str(p) for p in err.absolute_path) if err.absolute_path else "(root)"
        print("  [{}] {}".format(path, err.message), file=sys.stderr)

    print("{} error(s) found.".format(len(errors)), file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()

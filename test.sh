#!/bin/bash
set -e

case "$1" in
  base)
    pytest -q tests/base
    ;;
  new)
    pytest -q tests/new
    ;;
  *)
    echo "Usage: ./test.sh {base|new}"
    exit 1
    ;;
esac

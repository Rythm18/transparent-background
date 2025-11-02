#!/bin/bash
set -e

case "$1" in
  base)
    # Run existing/base tests - should pass at base commit
    pytest -q tests/base
    ;;
  new)
    # Run newly added tests - expected to fail before solution
    pytest -q tests/new
    ;;
  *)
    echo "Usage: ./test.sh {base|new}"
    exit 1
    ;;
esac

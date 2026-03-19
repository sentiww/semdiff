#!/usr/bin/env bash

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo "This script must be sourced to affect the current shell session."
  echo "Use: source activate.sh"
  exit 1
fi

_semdiff_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
alias semdiff="bash ${_semdiff_repo_root}/src/semdiff.sh"

echo "Alias activated for this shell session: semdiff -> bash ${_semdiff_repo_root}/src/semdiff.sh"

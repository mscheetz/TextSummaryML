#!/usr/bin/env bash

export HSA_OVERRIDE_GFX_VERSION=10.3.0
exec uv run uvicorn server.src.api.app:app --reload
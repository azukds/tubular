from mcr.microsoft.com/devcontainers/python:3.12-bookworm

copy --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

RUN UV_TOOL_DIR=/opt/uv-tools UV_TOOL_BIN_DIR=/usr/local/bin uv tool install prek
#!/bin/bash
set -e
socat TCP-LISTEN:8009,fork,reuseaddr TCP:element:80 &
socat TCP-LISTEN:8008,fork,reuseaddr TCP:synapse:8008 &
sleep 1
exec "$@"

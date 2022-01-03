#!/usr/bin/env bash

parallel -a ./instance_ips ./ll_pull.sh

# The Compose way to run this recipe: compose.yaml describes one rank, this file starts the four in order.
# (start.sh / stop.sh / prepare.sh do the same without Compose.)  make prepare, then make up; make down to stop.
SHELL := /bin/bash
.SHELLFLAGS := -euo pipefail -c
.DEFAULT_GOAL := help

SITE := site.env
# A value from site.env (and, for the release pins, config.sh, which reads site.env itself).
site = $(shell set -a; . ./$(SITE); echo "$$$(1)")
pin = $(shell . ./config.sh >/dev/null; echo "$$$(1)")

WORKERS := $(call site,WORKERS)
HOST := $(call site,HOST)
PORT := $(call site,PORT)
WORLD := $(words head $(WORKERS))
TF_VERSION := $(call pin,TF_VERSION)
TF_ARCHIVE := $(call pin,TF_ARCHIVE)
TF_SHA256 := $(call pin,TF_SHA256)
TF_URL := $(call pin,TF_URL)
TF_NAME := tensorfold-$(TF_VERSION)-linux-aarch64
CONTAINER := glm53-fp4-tf

COMPOSE := docker compose --env-file $(SITE)
SSH := ssh -n -o BatchMode=yes

.PHONY: help prepare up down config logs

help: ## List the targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-8s %s\n", $$1, $$2}'

prepare: ## Fetch and check the TensorFold release, point dist/tensorfold at it, copy this folder to every worker
	mkdir -p dist
	[ -f dist/$(TF_ARCHIVE) ] || curl -fsSL -o dist/$(TF_ARCHIVE) $(TF_URL)
	echo "$(TF_SHA256)  dist/$(TF_ARCHIVE)" | sha256sum -c -
	[ -x dist/$(TF_NAME)/bin/tensorfold-native ] || tar xzf dist/$(TF_ARCHIVE) -C dist
	cd dist/$(TF_NAME) && sha256sum -c --quiet SHA256SUMS
	ln -sfn $(TF_NAME) dist/tensorfold
	for w in $(WORKERS); do rsync -a --delete --exclude .git -e "ssh -o BatchMode=yes" ./ $$w:$(CURDIR)/; echo "copied to $$w"; done

up: ## Start ranks 3..1 on the workers, then rank 0 here, and wait for the API
	ws=($(WORKERS)); for ((r=$${#ws[@]}; r>=1; r--)); do \
	  $(SSH) $${ws[r-1]} "cd $(CURDIR) && RANK=$$r WORLD=$(WORLD) $(COMPOSE) up -d --force-recreate" && echo "rank $$r started on $${ws[r-1]}"; \
	done
	RANK=0 WORLD=$(WORLD) $(COMPOSE) up -d --force-recreate
	@echo "waiting for http://$(HOST):$(PORT)/v1/models (make logs)"
	@for i in $$(seq 1 120); do \
	  curl -fsS -m 3 http://$(HOST):$(PORT)/v1/models >/dev/null 2>&1 && { echo "serving after ~$$((i*10)) s"; exit 0; }; \
	  [ -n "$$(docker ps -q -f name=^$(CONTAINER)$$)" ] || { echo "rank 0 exited:"; docker logs --tail 30 $(CONTAINER); exit 1; }; \
	  sleep 10; \
	done; echo "not serving after 20 min"; exit 1

down: ## Stop and remove every rank
	$(COMPOSE) down
	for w in $(WORKERS); do $(SSH) $$w "cd $(CURDIR) && $(COMPOSE) down"; done

config: ## Print rank 0's compose file with every value filled in
	RANK=0 WORLD=$(WORLD) $(COMPOSE) config

logs: ## Follow rank 0's log
	docker logs -f $(CONTAINER)

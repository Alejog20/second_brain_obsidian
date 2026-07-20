.PHONY: sync run status check report test

sync:
	cd worker && uv sync --extra dev

run:
	cd worker && uv run second-brain run $(ARGS)

status:
	cd worker && uv run second-brain status $(ARGS)

check:
	cd worker && uv run second-brain check $(ARGS)

report:
	cd worker && uv run second-brain report $(ARGS)

test:
	cd worker && uv run pytest -q

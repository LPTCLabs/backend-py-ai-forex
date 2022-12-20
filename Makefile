TOPTARGETS := install update test qa build build-docker
SUBDIRS := $(wildcard services/*/.) $(wildcard batch-tasks/*/.) $(wildcard lambda/*/.)

$(TOPTARGETS): $(SUBDIRS)
$(SUBDIRS):
	$(MAKE) -C $@ $(MAKECMDGOALS)

.PHONY: $(TOPTARGETS) $(SUBDIRS)

CC = gcc
CFLAGS = -O3 -Wall -Iinclude -Isrc/c -march=native
LDFLAGS = -L. -l_lightgbm -Wl,-rpath,. -lm

SRCDIR = src/c
BINDIR = bin

TARGETS = $(BINDIR)/benchmark $(BINDIR)/test_correctness $(BINDIR)/unit_tests

all: $(TARGETS)

$(BINDIR)/benchmark: $(SRCDIR)/benchmark.c $(SRCDIR)/common.h $(SRCDIR)/model.h $(SRCDIR)/model_util.h $(SRCDIR)/qs.h $(SRCDIR)/qs_conversion.h $(SRCDIR)/vqs_rs_impl.h
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BINDIR)/test_correctness: $(SRCDIR)/test_correctness.c $(SRCDIR)/common.h $(SRCDIR)/model.h $(SRCDIR)/model_util.h $(SRCDIR)/qs.h $(SRCDIR)/qs_conversion.h $(SRCDIR)/vqs_rs_impl.h
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BINDIR)/unit_tests: $(SRCDIR)/unit_tests.c $(SRCDIR)/common.h $(SRCDIR)/model.h $(SRCDIR)/model_util.h $(SRCDIR)/qs.h $(SRCDIR)/qs_conversion.h $(SRCDIR)/vqs_rs_impl.h
	mkdir -p $(BINDIR)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(BINDIR)/*

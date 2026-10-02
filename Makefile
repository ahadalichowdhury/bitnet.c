# bitnet.c — BitNet b1.58 inference engine for Apple Silicon (pure C11 + ARM NEON)
#
#   make            build the `bitnet` CLI, libbitnet.a/.dylib/.so and all test programs
#   make lib        static + shared library (build/libbitnet.a, build/libbitnet.{dylib,so})
#   make bitnet     build the CLI only
#   make test       run every verification suite (all module suites + public API)
#   make bench      ./bitnet --bench (TTFT, prefill/decode tok/s, memory)
#   make asan       sanitizer builds (ASan+UBSan) of the CLI and API test, then run them
#   make leaks      run the API test and a CLI generation under macOS `leaks`
#   make clean
#
# Model files (see tools/export_bitnet.py) are expected at MODEL / TOKENIZER.

# Fail a recipe if any command of a pipeline fails (test output is filtered by
# grep, which must not hide a crash that happens after some PASSED lines).
SHELL       := /bin/bash
.SHELLFLAGS := -o pipefail -c

ifeq ($(origin CC),default)
CC := clang
endif
ARCH := $(shell uname -m)
ifneq (,$(filter arm64 aarch64,$(ARCH)))
CPUFLAG := -mcpu=native
else ifneq (,$(filter x86_64 amd64,$(ARCH)))
# AVX2/FMA/F16C kernels; build with `make X86_SIMD=` for x86-64 CPUs without AVX2.
X86_SIMD ?= -mavx2 -mfma -mf16c
CPUFLAG := -march=native $(X86_SIMD)
else
CPUFLAG := -march=native
endif
CFLAGS   ?= -O3 $(CPUFLAG) -Wall -Wextra -std=c11
CPPFLAGS += -Iinclude -Isrc -Itests
DEPFLAGS  = -MMD -MP
SANFLAGS  = -O1 -g $(CPUFLAG) -Iinclude -Isrc -Itests -Wall -Wextra -std=c11 -fsanitize=address,undefined -fno-omit-frame-pointer

MODEL     ?= models/bitnet_2b4t.bitnet
TOKENIZER ?= models/hf/bitnet-b1.58-2B-4T/tokenizer.json
REF       ?= models/bitnet_2b4t.ref
REF_LOGITS?= models/bitnet_2b4t.ref_logits
GOLDEN    ?= tests/data/tokenizer_golden.txt

BUILD := build
OBJ   := $(BUILD)/obj

# Engine library (everything behind bitnet.h).
LIB_SRC := src/bitnet.c src/generate.c src/transformer.c src/threadpool.c src/model_loader.c src/tokenizer.c
LIB_OBJ := $(LIB_SRC:src/%.c=$(OBJ)/%.o)
LIB     := $(BUILD)/libbitnet.a
ifeq ($(shell uname -s),Darwin)
SHLIB       := $(BUILD)/libbitnet.dylib
SHLIB_FLAGS := -dynamiclib -install_name @rpath/libbitnet.dylib
else
SHLIB       := $(BUILD)/libbitnet.so
SHLIB_FLAGS := -shared -Wl,-soname,libbitnet.so
endif

TESTS := $(BUILD)/test_dot_product $(BUILD)/test_gemv $(BUILD)/test_bitlinear \
         $(BUILD)/test_loader $(BUILD)/test_tokenizer $(BUILD)/test_transformer \
         $(BUILD)/test_generate $(BUILD)/test_api

.PHONY: all lib tests test test-mock bench asan leaks clean

# The CLI is ./bitnet for the default build; variant builds (make BUILD=...)
# write $(BUILD)/bitnet so they never overwrite the main binary.
ifeq ($(BUILD),build)
CLI := bitnet
else
CLI := $(BUILD)/bitnet
.PHONY: bitnet
bitnet: $(CLI)
endif

all: $(CLI) lib tests

lib: $(LIB) $(SHLIB)
tests: $(TESTS)

# ---- objects / library ----------------------------------------------------

# Engine objects are position-independent (shared library) and hide every
# symbol except the BITNET_API functions of include/bitnet.h.
$(OBJ)/%.o: src/%.c Makefile | $(OBJ)
	$(CC) $(CPPFLAGS) $(CFLAGS) -fPIC -fvisibility=hidden $(DEPFLAGS) -c $< -o $@
$(OBJ)/%.o: tests/%.c Makefile | $(OBJ)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c $< -o $@
$(OBJ)/main.o: app/main.c Makefile | $(OBJ)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(DEPFLAGS) -c $< -o $@

$(OBJ) $(BUILD):
	@mkdir -p $@

$(LIB): $(LIB_OBJ)
	ar rcs $@ $^

$(SHLIB): $(LIB_OBJ)
	$(CC) $(CFLAGS) $(SHLIB_FLAGS) $^ -lpthread -lm -o $@

# The CLI binary: ./bitnet (or $(BUILD)/bitnet for variant builds)
$(CLI): $(OBJ)/main.o $(LIB)
	$(CC) $(CFLAGS) $^ -lpthread -lm -o $@

# ---- test programs ----------------------------------------------------------

$(BUILD)/test_dot_product: $(OBJ)/test_dot_product.o
	$(CC) $(CFLAGS) $^ -lpthread -lm -o $@
$(BUILD)/test_gemv: $(OBJ)/test_gemv.o
	$(CC) $(CFLAGS) $^ -lpthread -lm -o $@
$(BUILD)/test_bitlinear: $(OBJ)/test_bitlinear.o
	$(CC) $(CFLAGS) $^ -lpthread -lm -o $@
$(BUILD)/test_loader: $(OBJ)/test_loader.o $(OBJ)/model_loader.o
	$(CC) $(CFLAGS) $^ -lpthread -lm -o $@
$(BUILD)/test_tokenizer: $(OBJ)/test_tokenizer.o $(OBJ)/tokenizer.o $(OBJ)/threadpool.o
	$(CC) $(CFLAGS) $^ -lpthread -lm -o $@
$(BUILD)/test_transformer: $(OBJ)/test_transformer.o $(OBJ)/transformer.o $(OBJ)/threadpool.o \
                            $(OBJ)/model_loader.o $(OBJ)/tokenizer.o
	$(CC) $(CFLAGS) $^ -lpthread -lm -o $@
$(BUILD)/test_generate: $(OBJ)/test_generate.o $(OBJ)/generate.o $(OBJ)/transformer.o \
                         $(OBJ)/threadpool.o $(OBJ)/model_loader.o $(OBJ)/tokenizer.o
	$(CC) $(CFLAGS) $^ -lpthread -lm -o $@
$(BUILD)/test_api: $(OBJ)/test_api.o $(LIB)
	$(CC) $(CFLAGS) $^ -lpthread -lm -o $@

# ---- running ----------------------------------------------------------------

test: tests
	@echo "== dot product";  $(BUILD)/test_dot_product | grep -E "PASSED|Speedup"
	@echo "== gemv";                 $(BUILD)/test_gemv --quick | grep -E "PASSED"
	@echo "== bitlinear";            $(BUILD)/test_bitlinear --quick | grep -E "PASSED"
	@echo "== loader (real model)";  $(BUILD)/test_loader $(MODEL) --ref $(REF) | grep -E "PASSED"
	@echo "== tokenizer";            $(BUILD)/test_tokenizer $(TOKENIZER) $(GOLDEN) | grep -E "PASSED"
	@echo "== transformer";          $(BUILD)/test_transformer $(MODEL) $(TOKENIZER) $(REF_LOGITS) --quick | grep -E "PASSED|->"
	@echo "== generate";           $(BUILD)/test_generate $(MODEL) $(TOKENIZER) --quick | grep -E "PASSED|^  \["
	@echo "== public API";                   $(BUILD)/test_api $(MODEL) $(TOKENIZER)
	@echo "== all tests passed"

# Model-free suites + mock-model loader/transformer checks (used by CI).
MOCK := $(BUILD)/mock
test-mock: tests | $(BUILD)
	@mkdir -p $(MOCK)
	python3 tools/export_bitnet.py --mock --mock-save-hf $(MOCK)/hf --output $(MOCK)/mock.bitnet --ref $(MOCK)/mock.ref --max-seq-len 256
	python3 tools/reference_bitnet.py $(MOCK)/hf --tokens $$(python3 -c "import random; r=random.Random(3); print(','.join(str(r.randrange(1000)) for _ in range(100)))") --out $(MOCK)/mock.ref_logits \
	  2> $(MOCK)/reference.log || { cat $(MOCK)/reference.log; false; }
	$(BUILD)/test_dot_product | grep -E "PASSED"
	$(BUILD)/test_gemv --quick | grep -E "PASSED"
	$(BUILD)/test_bitlinear --quick | grep -E "PASSED"
	$(BUILD)/test_loader $(MOCK)/mock.bitnet --ref $(MOCK)/mock.ref --corrupt-tests --tmpdir $(MOCK) | grep -E "PASSED"
	$(BUILD)/test_generate --unit-only | grep -E "PASSED"
	@if [ -f $(TOKENIZER) ]; then $(BUILD)/test_tokenizer $(TOKENIZER) $(GOLDEN) --tmpdir $(MOCK) | grep -E "PASSED" && \
	  $(BUILD)/test_transformer $(MOCK)/mock.bitnet $(TOKENIZER) $(MOCK)/mock.ref_logits | grep -E "PASSED|->"; \
	  else echo "(tokenizer.json not found: tokenizer/transformer suites skipped)"; fi
	@echo "== mock tests passed"

bench: $(CLI)
	./$(CLI) --model $(MODEL) --tokenizer $(TOKENIZER) --bench

asan: | $(BUILD)
	$(CC) $(SANFLAGS) app/main.c $(LIB_SRC) -o $(BUILD)/bitnet-asan
	$(CC) $(SANFLAGS) tests/test_api.c $(LIB_SRC) -o $(BUILD)/test_api-asan
	ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
	  $(BUILD)/bitnet-asan --model $(MODEL) --tokenizer $(TOKENIZER) -p "Say hello in French." \
	  --temp 0 -n 24 --stats
	ASAN_OPTIONS=halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
	  $(BUILD)/test_api-asan $(MODEL) $(TOKENIZER)

leaks: $(CLI) $(BUILD)/test_api
	MallocStackLogging=1 leaks --atExit -- $(BUILD)/test_api $(MODEL) $(TOKENIZER) | grep -E "leaks for|PASSED"
	MallocStackLogging=1 leaks --atExit -- ./$(CLI) --model $(MODEL) --tokenizer $(TOKENIZER) \
	  -p "Name a color." --temp 0 -n 16 | grep -E "leaks for"

clean:
	rm -rf $(BUILD) $(CLI)

-include $(LIB_OBJ:.o=.d) $(OBJ)/main.d $(TESTS:$(BUILD)/%=$(OBJ)/%.d)

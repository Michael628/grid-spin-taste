# Main Makefile for stag_gamma project
# This orchestrates configuration, building and running the stag_gamma executable

.PHONY: configure build clean distclean run run-free run-l4444-free run-l6666-free run-l8888-free run-l4444 run-with-args help

# Default target
all: build

# Configure the build system
configure:
	@if [ ! -f configure ]; then \
		echo "Generating configure script..."; \
		autoreconf -fiv; \
	fi
	@if [ ! -f build/Makefile ]; then \
		echo "Running configure..."; \
		./configure; \
	fi

# Build the stag_gamma executable using the build directory Makefile
build: configure
	@echo "Building stag_gamma..."
	$(MAKE) -C build stag_gamma

# Clean build artifacts
clean:
	@if [ -f build/Makefile ]; then \
		$(MAKE) -C build clean; \
	fi

# Clean everything including configure-generated files
distclean: clean
	rm -f configure config.log config.status
	rm -rf autom4te.cache
	rm -f build/Makefile

# Free field run targets
run-l4444-free: build
	@echo "Running stag_gamma with free field parameters (4x4x4x4 lattice)..."
	cd test && ../build/stag_gamma params/param_free_l4444.xml --grid 4.4.4.4

run-l6666-free: build
	@echo "Running stag_gamma with free field parameters (6x6x6x6 lattice)..."
	cd test && ../build/stag_gamma params/param_free_l6666.xml --grid 6.6.6.6

run-l8888-free: build
	@echo "Running stag_gamma with free field parameters (8x8x8x8 lattice)..."
	cd test && ../build/stag_gamma params/param_free_l8888.xml --grid 8.8.8.8

# Alias for backward compatibility
run-free: run-l8888-free

# Run with param_l4444.xml (4x4x4x4 lattice)  
run-l4444: build
	@echo "Running stag_gamma with l4444 parameters (4x4x4x4 lattice)..."
	cd test && ../build/stag_gamma params/param_l4444.xml --grid 4.4.4.4

# Generic run target with configurable parameters
# Usage: make run-with-args PARAM=param_file.xml GRID=x.y.z.t
run-with-args: build
	@if [ -z "$(PARAM)" ] || [ -z "$(GRID)" ]; then \
		echo "Error: Both PARAM and GRID must be specified"; \
		echo "Usage: make run-with-args PARAM=param_file.xml GRID=x.y.z.t"; \
		echo "Example: make run-with-args PARAM=param_l4444.xml GRID=4.4.4.4"; \
		exit 1; \
	fi
	@echo "Running stag_gamma with $(PARAM) and grid $(GRID)..."
	cd test && ../build/stag_gamma params/$(PARAM) --grid $(GRID)

# Convenience alias for run-with-args
run: 
	@echo "Available run targets:"
	@echo ""
	@echo "Free field runs:"
	@echo "  make run-l4444-free - Run with free field parameters (4x4x4x4)"
	@echo "  make run-l6666-free - Run with free field parameters (6x6x6x6)"
	@echo "  make run-l8888-free - Run with free field parameters (8x8x8x8)"
	@echo "  make run-free       - Alias for run-l8888-free"
	@echo ""
	@echo "Interacting field runs:"
	@echo "  make run-l4444      - Run with interacting field parameters (4x4x4x4)"
	@echo ""
	@echo "Custom runs:"
	@echo "  make run-with-args PARAM=file.xml GRID=x.y.z.t - Run with custom parameters"
	@echo ""
	@echo "Example: make run-with-args PARAM=param_l4444.xml GRID=4.4.4.4"

# Help target
help:
	@echo "Available targets:"
	@echo "  configure       - Generate and run configure script"
	@echo "  build           - Compile the stag_gamma executable"
	@echo "  clean           - Clean build artifacts"
	@echo "  distclean       - Clean everything including configure files"
	@echo ""
	@echo "Free field runs:"
	@echo "  run-l4444-free  - Run with free field parameters (4x4x4x4)"
	@echo "  run-l6666-free  - Run with free field parameters (6x6x6x6)"
	@echo "  run-l8888-free  - Run with free field parameters (8x8x8x8)"
	@echo "  run-free        - Alias for run-l8888-free"
	@echo ""
	@echo "Interacting field runs:"
	@echo "  run-l4444       - Run with interacting field parameters (4x4x4x4)"
	@echo ""
	@echo "Custom runs:"
	@echo "  run-with-args   - Run with custom PARAM and GRID arguments"
	@echo "  run             - Show available run options"
	@echo "  help            - Show this help message"
	@echo ""
	@echo "Configuration options:"
	@echo "  ./configure --with-grid=PATH        - Specify Grid installation path"
	@echo "  ./configure --enable-optimization   - Enable compiler optimizations"
	@echo "  ./configure --enable-debug          - Enable debug symbols"
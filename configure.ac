AC_INIT([stag-gamma], [1.0], [support@example.com])
AC_PROG_CXX
AC_CONFIG_FILES([build/Makefile])

# Check for Grid installation
AC_ARG_WITH([grid],
    [AS_HELP_STRING([--with-grid=PATH], [path to Grid installation])],
    [grid_prefix="$withval"],
    [grid_prefix=""])

# Try to find Grid automatically if not specified
if test "x$grid_prefix" = "x"; then
    AC_MSG_NOTICE([Searching for Grid installation...])
    # Check common locations
    for dir in "$HOME/work/Grid/install-scalar" \
               "$HOME/Dropbox/Physics/projects/gridProjects/Grid/install-scalar" \
               "/usr/local" \
               "/opt/grid" \
               "/usr"; do
        AC_MSG_CHECKING([for grid-config in $dir/bin])
        if test -x "$dir/bin/grid-config"; then
            grid_prefix="$dir"
            AC_MSG_RESULT([found])
            break
        else
            AC_MSG_RESULT([not found])
        fi
    done
fi

# Verify Grid installation
if test "x$grid_prefix" = "x" || ! test -x "$grid_prefix/bin/grid-config"; then
    AC_MSG_ERROR([
Grid installation not found. Please specify the Grid installation path:
  ./configure --with-grid=PATH

Common locations to check:
  $HOME/work/Grid/install-scalar
  $HOME/Dropbox/Physics/projects/gridProjects/Grid/install-scalar
  /usr/local
  /opt/grid
])
fi

AC_MSG_NOTICE([Using Grid installation at: $grid_prefix])

# Test that grid-config works
AC_MSG_CHECKING([if grid-config works])
if "$grid_prefix/bin/grid-config" --version >/dev/null 2>&1; then
    AC_MSG_RESULT([yes])
    grid_version=$("$grid_prefix/bin/grid-config" --version 2>/dev/null || echo "unknown")
    AC_MSG_NOTICE([Grid version: $grid_version])
else
    AC_MSG_RESULT([no])
    AC_MSG_ERROR([grid-config at $grid_prefix/bin/grid-config is not working])
fi

# Add compiler optimization options
AC_ARG_ENABLE([optimization],
    [AS_HELP_STRING([--enable-optimization], [enable compiler optimizations (default: no)])],
    [optimization="$enableval"],
    [optimization="no"])

if test "x$optimization" = "xyes"; then
    AC_SUBST([OPTFLAGS], [-O3])
    AC_MSG_NOTICE([Compiler optimizations enabled (-O3)])
else
    AC_SUBST([OPTFLAGS], [-O0])
    AC_MSG_NOTICE([Compiler optimizations disabled (-O0)])
fi

# Add debug support
AC_ARG_ENABLE([debug],
    [AS_HELP_STRING([--enable-debug], [enable debug symbols (default: no)])],
    [debug="$enableval"],
    [debug="no"])

if test "x$debug" = "xyes"; then
    AC_SUBST([DEBUGFLAGS], [-g])
    AC_MSG_NOTICE([Debug symbols enabled (-g)])
else
    AC_SUBST([DEBUGFLAGS], [""])
fi

# Add A2A blocking size configuration
AC_ARG_WITH([a2a-blocking],
    [AS_HELP_STRING([--with-a2a-blocking=SIZE], [set A2A blocking size (default: 128)])],
    [a2a_blocking="$withval"],
    [a2a_blocking="128"])

# Validate that the blocking size is a positive integer
case "$a2a_blocking" in
    ''|*[[!0-9]]*)
        AC_MSG_ERROR([A2A blocking size must be a positive integer, got: $a2a_blocking])
        ;;
    *)
        if test "$a2a_blocking" -le 0; then
            AC_MSG_ERROR([A2A blocking size must be positive, got: $a2a_blocking])
        fi
        ;;
esac

AC_SUBST([A2A_BLOCKING_FLAGS], ["-DDEV_A2A_BLOCKING=$a2a_blocking"])
AC_MSG_NOTICE([A2A blocking size set to: $a2a_blocking])

AC_SUBST([GRID_PREFIX], [$grid_prefix])
AC_OUTPUT

echo ""
echo "Configuration summary:"
echo "  Grid installation: $grid_prefix"
echo "  Optimization:      $optimization"
echo "  Debug symbols:     $debug"
echo "  A2A blocking size: $a2a_blocking"
echo ""
echo "Now run 'make' to build the project."

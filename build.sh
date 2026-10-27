#!/bin/bash
# Builds libpyhxt.so here: pyhxt.c with HXT's core, predicates, tetMesh and tetBR from
# gmsh 4.13.1 (gmsh-4.13.1/, as gmsh builds them, but tetBR with HXT's own copy of TetGen's
# boundary recovery, HXT_WITHOUT_GMSH; predicates.c without unsafe math; its warning about
# TetGen's licence is expected). -ffp-contract=off throughout, so that a compiler that
# fuses multiply-adds by default (gcc on aarch64) gives the same numbers as gmsh's x86 wheel.
# Usage: ./build.sh [cc] [c++]   (gcc and g++ by default; both need OpenMP)
set -eu
CC=${1:-gcc}; CXX=${2:-g++}
HERE=$(cd "$(dirname "$0")" && pwd)
H=$HERE/gmsh-4.13.1/contrib/hxt
INC="-I$H/core/include -I$H/core/src -I$H/predicates/include -I$H/predicates/src -I$H/tetBR/include -I$H/tetBR/src -I$H/tetMesh/include -I$H/tetMesh/src"
CFLAGS="-O2 -DNDEBUG -fPIC -fopenmp -ffp-contract=off -DHXT_WITHOUT_GMSH"
OBJ=$(mktemp -d); trap 'rm -rf "$OBJ"' EXIT
pids=()
for f in $H/core/src/hxt_{bbox,mesh,message,sort}.c \
         $H/tetMesh/src/{hxt_edgeRemoval,hxt_smoothing,HXTSPR,hxt_tetColor,hxt_tetDelaunay,hxt_tetDelaunayReshape,hxt_tetFlag,hxt_tetMesh,hxt_tetNodalSize,hxt_tetOpti,hxt_tetQuality,hxt_tetRefine,hxt_tetRepair,hxt_tetSync,hxt_tetUtils,hxt_vertices}.c \
         $HERE/pyhxt.c; do
  $CC -std=gnu99 $CFLAGS $INC -c "$f" -o "$OBJ/$(basename "$f" .c).o" & pids+=($!)
done
$CC -std=gnu99 $CFLAGS -fno-unsafe-math-optimizations $INC -c $H/predicates/src/predicates.c -o "$OBJ/predicates.o" & pids+=($!)
$CXX $CFLAGS $INC -c $H/tetBR/src/hxt_boundary_recovery.cxx -o "$OBJ/hxt_boundary_recovery.o" & pids+=($!)
for p in "${pids[@]}"; do wait $p; done
$CXX -shared -fopenmp "$OBJ"/*.o -o "$HERE/libpyhxt.so.new"
mv "$HERE/libpyhxt.so.new" "$HERE/libpyhxt.so"   # never rewrite a library a run has loaded
echo "built $HERE/libpyhxt.so ($($CC --version | head -1))"

/* pyhxt: the switches pyhxt adds to HXT (gmsh 4.13.1), set per call by pyhxt_tetmesh.
 * With none set, HXT runs exactly as in gmsh. AGPL-3.0-or-later. */
#ifndef PYHXT_FLAGS_H
#define PYHXT_FLAGS_H
#ifdef __cplusplus
extern "C" {
#endif
extern int pyhxt_flags;
#ifdef __cplusplus
}
#endif
/* boundary recovery: bond the surface edges and triangles the empty mesh already has
 * before TetGen's random-order passes, which then only search for the missing ones */
#define PYHXT_PREBOND 1
/* boundary recovery: skip TetGen's global Delaunay recovery (a Lawson pass over every
 * face) after the boundary is recovered */
#define PYHXT_NO_RECOVER_DELAUNAY 2
/* empty mesh: insert the embedded (seed) points before the surface points; a point
 * inside each hole bounded by co-spherical points keeps its Delaunay from degenerating */
#define PYHXT_SEEDS_FIRST 4
/* HXT's reproducible option off: faster, but the mesh changes from run to run */
#define PYHXT_NONREPRODUCIBLE 8
#endif

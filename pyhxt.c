/* pyhxt: HXT's hxtTetMesh called as gmsh 4.13.1's meshGRegionHxt.cpp calls it, on arrays,
 * with pyhxt's switches (pyhxt_flags.h) on top. Built against gmsh 4.13.1's contrib/hxt
 * sources. AGPL-3.0-or-later. */
#include <stdio.h>
#include <stdint.h>
#include "hxt_tetMesh.h"
#include "hxt_tools.h"
#include "pyhxt_flags.h"

int pyhxt_flags = 0;
uint32_t pyhxt_hole_first = 0, pyhxt_num_holes = 0;
uint64_t* pyhxt_tri2tet = NULL;
static FILE *logfile = NULL;

static HXTStatus messageCallback(HXTMessage *msg)
{
  if(msg && logfile) {
    const char *level[] = {"Info", "Debug", "Warning", "Error", "Trace"};
    fprintf(logfile, "%s: %s\n", level[msg->level], msg->string);
    fflush(logfile);
  }
  return HXT_STATUS_OK;
}

/* gmsh's nodalSizesCallBack with no size field, point size or entity size: the
 * background size is MAX_LC, so a point keeps its interpolated size capped by the
 * model's characteristic length, or takes that length where it has none */
static HXTStatus nodalSizes(double *pts, uint32_t *volume, size_t n, void *data)
{
  double lcGlob = *(double *)data;
  HXT_INFO("Computing interpolated mesh sizes...");
  for(size_t i = 0; i < n; i++) {
    double s = pts[4 * i + 3];
    pts[4 * i + 3] = s > 0.0 ? (lcGlob < s ? lcGlob : s) : lcGlob;
  }
  HXT_INFO("Done computing interpolated mesh sizes");
  return HXT_STATUS_OK;
}

/* The version of this interface: 2 added nholes. */
int pyhxt_abi(void) { return 2; }

/* xyz: nv x 3, the surface's vertices; seeds: nseed x 3, embedded points, which follow
 * them as vertices nv, nv + 1, ..., the first nholes of them hole points (the volumes
 * holding them are left out); tri: ntri x 3 with one color each; volumes: nvol, each
 * bounded by the surfaces surfaces[offsets[i]:offsets[i+1]], or nvol 0 for every volume
 * HXT finds; flags: pyhxt_flags.h (0: as gmsh). On success *out holds HXT's mesh. */
int pyhxt_tetmesh(uint32_t nv, const double *xyz, uint32_t nseed, const double *seeds,
                  uint32_t nholes, uint64_t ntri, const uint32_t *tri,
                  const uint32_t *color, uint32_t nvol, const uint32_t *offsets,
                  const uint32_t *surfaces, int threads,
                  double sizeMin, double sizeMax, double lcGlob, double tolerance,
                  double qualityMin, int optimize, int flags, const char *logpath,
                  void **out)
{
  HXTMesh *m = NULL;
  *out = NULL;
  logfile = logpath ? fopen(logpath, "a") : NULL;
  hxtSetMessageCallback(messageCallback);
  HXTStatus status = hxtMeshCreate(&m);
  if(status != HXT_STATUS_OK) goto done;

  pyhxt_flags = flags;
  pyhxt_hole_first = nv;
  pyhxt_num_holes = nholes < nseed ? nholes : nseed;
  m->vertices.num = m->vertices.size = nv + nseed;
  if((status = hxtAlignedMalloc(&m->vertices.coord,
                                4 * ((size_t)nv + nseed) * sizeof(double))))
    goto done;
  for(uint32_t i = 0; i < nv + nseed; i++) {
    const double *x = i < nv ? xyz + 3 * (size_t)i : seeds + 3 * (size_t)(i - nv);
    m->vertices.coord[4 * i + 0] = x[0];
    m->vertices.coord[4 * i + 1] = x[1];
    m->vertices.coord[4 * i + 2] = x[2];
    m->vertices.coord[4 * i + 3] = 0.0;
  }
  if(nseed) {
    m->points.num = m->points.size = nseed;
    if((status = hxtAlignedMalloc(&m->points.node, nseed * sizeof(uint32_t))) ||
       (status = hxtAlignedMalloc(&m->points.color, nseed * sizeof(uint32_t))))
      goto done;
    for(uint32_t i = 0; i < nseed; i++) {
      m->points.node[i] = nv + i;
      m->points.color[i] = 0;
    }
  }
  m->triangles.num = m->triangles.size = ntri;
  if((status = hxtAlignedMalloc(&m->triangles.node, 3 * ntri * sizeof(uint32_t))) ||
     (status = hxtAlignedMalloc(&m->triangles.color, ntri * sizeof(uint32_t))))
    goto done;
  for(uint64_t i = 0; i < 3 * ntri; i++) m->triangles.node[i] = tri[i];
  for(uint64_t i = 0; i < ntri; i++) m->triangles.color[i] = color[i];

  m->brep.numVolumes = nvol;
  if(nvol == 0) goto options; /* HXT finds the volumes and fills the brep itself */
  if((status = hxtAlignedMalloc(&m->brep.numSurfacesPerVolume, nvol * sizeof(uint32_t))) ||
     (status = hxtAlignedMalloc(&m->brep.surfacesPerVolume,
                                offsets[nvol] * sizeof(uint32_t))))
    goto done;
  for(uint32_t i = 0; i < nvol; i++)
    m->brep.numSurfacesPerVolume[i] = offsets[i + 1] - offsets[i];
  for(uint32_t i = 0; i < offsets[nvol]; i++) m->brep.surfacesPerVolume[i] = surfaces[i];
  /* unused by hxtTetMesh, but in its input hash: gmsh gives the number of surfaces and
   * no curves on them (a volume's surfaces are distinct, and one volume per call here) */
  m->brep.numSurfaces = offsets[nvol];
  if((status = hxtAlignedMalloc(&m->brep.numCurvesPerSurface,
                                offsets[nvol] * sizeof(uint32_t))))
    goto done;
  for(uint32_t i = 0; i < offsets[nvol]; i++) m->brep.numCurvesPerSurface[i] = 0;

options:;
  HXTTetMeshOptions options = {
    threads, threads, threads, // default, Delaunay and improvement threads
    !(flags & PYHXT_NONREPRODUCIBLE), // reproducible
    1, // verbosity, as gmsh's at General.Verbosity 5
    1, // stat
    1, // refine
    optimize,
    tolerance,
    {NULL, NULL, qualityMin},
    {nodalSizes, &lcGlob, sizeMin, sizeMax, 1.0}};
  status = hxtTetMesh(m, &options);

done:
  pyhxt_flags = 0;
  pyhxt_num_holes = 0;
  if(status == HXT_STATUS_OK) {
    /* the adjacency (32 B a tet) and flags are not handed out: free them before the caller
     * allocates its copies, so that HXT's mesh and those copies are not all held at once */
    hxtAlignedFree(&m->tetrahedra.neigh);
    hxtAlignedFree(&m->tetrahedra.flag);
    *out = m;
  }
  else if(m)
    hxtMeshDelete(&m);
  if(logfile) fclose(logfile);
  logfile = NULL;
  return (int)status;
}

void pyhxt_counts(void *mesh, uint32_t *nv, uint64_t *ntet, uint64_t *ntri)
{
  HXTMesh *m = mesh;
  *nv = m->vertices.num;
  *ntet = m->tetrahedra.num;
  *ntri = m->triangles.num;
}

/* coords: nv x 3; tets: ntet x 4 and their colors; triangles: ntri x 3 */
void pyhxt_copy(void *mesh, double *coords, uint32_t *tets, uint32_t *tetcolor,
                  uint32_t *tris)
{
  HXTMesh *m = mesh;
  for(uint32_t i = 0; i < m->vertices.num; i++)
    for(int j = 0; j < 3; j++) coords[3 * i + j] = m->vertices.coord[4 * i + j];
  for(uint64_t i = 0; i < 4 * m->tetrahedra.num; i++) tets[i] = m->tetrahedra.node[i];
  for(uint64_t i = 0; i < m->tetrahedra.num; i++) tetcolor[i] = m->tetrahedra.color[i];
  for(uint64_t i = 0; i < 3 * m->triangles.num; i++) tris[i] = m->triangles.node[i];
}

void pyhxt_free(void *mesh)
{
  HXTMesh *m = mesh;
  if(m) hxtMeshDelete(&m);
}

/* colors: ntri, the surface number of each of HXT's triangles */
void pyhxt_copy_triangle_colors(void *mesh, uint32_t *colors)
{
  HXTMesh *m = mesh;
  for(uint64_t i = 0; i < m->triangles.num; i++)
    colors[i] = m->triangles.color ? m->triangles.color[i] : 0;
}
